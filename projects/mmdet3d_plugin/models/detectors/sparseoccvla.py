import time
import queue
import torch
import os
import numpy as np
import torch.nn as nn
from ..losses.map_loss import l2_loss
from mmdet.models import build_loss
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import get_dist_info
from mmcv.runner.fp16_utils import cast_tensor_type
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .utils import GridMask, pad_multiple, GpuPhotoMetricDistortion
from ..utils.misc import load_model,ClipBackbone
from transformers import AutoTokenizer, GenerationConfig
from ...datasets.utils.constants import IGNORE_INDEX, EGO_WAYPOINT_TOKEN
try:
    from ..dense_heads.llava_llama import add_special_token
except:
    pass
import mmcv
from ..dense_heads.bbox.utils import decode_points,encode_points,\
    EgoAttention,gen_sineembed_for_position,linear_relu_ln,CustomTransformerDecoderLayer,SinusoidalPosEmb
# from ..internvl_chat.internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM

import json
import torch.nn.functional as F
from ..utils.opus_sampling import sampling_3d
from mmcv.runner.checkpoint import load_checkpoint
from diffusers.schedulers import DDIMScheduler


@DETECTORS.register_module()
class SparseOccVLA(MVXTwoStageDetector):
    def __init__(self,
                 save_path = './opus_format/',
                 use_grid_mask=True,
                 data_aug=None,
                 stop_prev_grad=0,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 tokenizer=None,
                 plan_loss_cls=None,
                 plan_loss_reg=None,
                 lm_head=None,
                 frozen=True,
                 use_lora=False,
                 image_casual=True,
                 out_dims=4096,
                 num_global_query=0,
                 forecasting=False,
                 use_gen_token=False,
                 canbus_fu=False,
                 chat_config=None):
        super().__init__(pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder,
                         pts_fusion_layer, img_backbone, pts_backbone, img_neck,
                         pts_neck, pts_bbox_head, img_roi_head, img_rpn_head,
                         train_cfg, test_cfg, pretrained)
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.save_path = save_path
        self.memory = {}
        self.queue = queue.Queue()
        self.embed_dims = self.pts_bbox_head.embed_dims
        self.forecasting = forecasting
        self.num_frames = self.pts_bbox_head.transformer.num_frames
        self.eval_occ = False
        self.use_gen_token = use_gen_token
        self.canbus_fu = canbus_fu
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
        self.chat_cfg = chat_config
        if tokenizer is not None and 'InternVL' in tokenizer:
            self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                        model_max_length=2048,
                                        trust_remote_code=True,
                                        use_fast=False,
                                        )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model_name='internvl'
        elif tokenizer is not None:
            self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                        model_max_length=2048,
                                        padding_side="right",
                                        use_fast=False,
                                        )
            self.model_name='llava'
            self.tokenizer.pad_token = self.tokenizer.unk_token
        else:
            self.tokenizer = None

        if lm_head is not None:
            dtype = torch.float32 if train_cfg is not None else torch.float16
            lm_kwargs = {'use_gen_token':use_gen_token}
            self.lm_head = load_model(lm_head, use_lora, frozen ,dtype,lm_kwargs)

            self.can_bus_embed = nn.Sequential(nn.Linear(40,self.embed_dims),
                                               nn.ReLU(),
                                               nn.Linear(self.embed_dims,out_dims))
            self.projector = nn.Sequential(
                nn.Linear(self.embed_dims,out_dims))

            self.position_encoder = nn.Sequential(
                nn.Linear(3 * self.pts_bbox_head.num_refines[-1], self.embed_dims), 
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                )
        else:
            self.lm_head = None

        if plan_loss_cls is not None:
            self.waypoint_cls_loss = build_loss(plan_loss_cls)
            self.waypoint_reg_loss = build_loss(plan_loss_reg)

        if use_gen_token:
            add_special_token([EGO_WAYPOINT_TOKEN], tokenizer = self.tokenizer, model = self.lm_head)
            self.lm_head.config.waypoint_token_idx = self.tokenizer(EGO_WAYPOINT_TOKEN, add_special_tokens=False).input_ids[0]
        self.num_global_query = num_global_query
        self.image_casual = image_casual

        if num_global_query>0:
            self.global_attention = EgoAttention(self.embed_dims, num_layers=3, num_heads=8, mlp_ratio=4,drop=0.1)
            self.global_projector = nn.Sequential(
                nn.Linear(self.embed_dims,out_dims)
                )
        else:
            self.global_attention = None
        if self.forecasting:
            self.position_encoder_4d = nn.Sequential(
                nn.Linear(self.pts_bbox_head.num_refines[-1]*4,self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims,self.embed_dims))
            self.cls_branch_forecasting = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            # nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.pts_bbox_head.num_refines[-1] * 17)
            )
            self.reg_branch_forecasting = nn.Sequential(nn.Linear(self.embed_dims,self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims,self.pts_bbox_head.num_refines[-1]*3)
            )
            self.canbus_fu_embedding = nn.Sequential(
                nn.Linear(14,self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims,self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims,self.embed_dims),
            )

            self.res_layer = nn.Sequential(
                nn.Linear(self.embed_dims+256, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                )
            self.query_decoder = nn.Sequential(
                nn.Linear(out_dims,self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims,256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256,256))

            if self.use_gen_token: # Please ignore here.
                self.num_mode=6
                self.num_fu_traj=6
                self.ego_decoder = nn.Sequential(
                    nn.Linear(21+40+out_dims, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims,self.embed_dims),
                    )
                self.waypoint_cls = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims // 2),
                    nn.GELU(),
                    nn.Linear(self.embed_dims//2, self.embed_dims//2),
                    nn.GELU(),
                    nn.Linear(self.embed_dims//2, 18),
                )
                self.traj_encoder = nn.Sequential(
                        *linear_relu_ln(self.embed_dims, 1, 1,384),
                    nn.Linear(self.embed_dims,self.embed_dims))
                self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(self.embed_dims),
                nn.Linear(self.embed_dims, self.embed_dims * 4),
                nn.Mish(),
                nn.Linear(self.embed_dims * 4, self.embed_dims))
                self.diff_decoder = CustomTransformerDecoderLayer(num_poses=2,d_model=self.embed_dims,d_ffn=self.embed_dims)

                self.waypoint_loss = nn.L1Loss(reduction='none')
                plan_anchor = np.load('data/nuscenes/kmeans_plan_6.npy')
                plan_anchor_list = []
                for anchor in plan_anchor:
                    sort = np.argsort(anchor[:,-1,0])
                    plan_anchor_list.append(anchor[sort])
                plan_anchor = np.stack(plan_anchor_list)
                self.plan_anchor = nn.Parameter(torch.tensor(plan_anchor,dtype=torch.float32),
                requires_grad=False)

        self.pc_range = self.pts_bbox_head.pc_range
        self.planning_loss = l2_loss()
        self.iter = 30000 #preset
    
    def loss_fu_occ(self,gt_fu0_dict,gt_fu1_dict,src_tran,dst_tran,points_pred,cls_pred):
        B = len(src_tran)
        device = gt_fu0_dict['gt_points_list'][0].device
        num_voxel = torch.tensor([400,300,24],device=device)
        gt_points_list,gt_masks_list,gt_labels_list = \
        gt_fu0_dict['gt_points_list'],gt_fu0_dict['gt_masks_list'],gt_fu0_dict['gt_labels_list']
        voxel_semantics = torch.ones([B,400,300,24],dtype=torch.uint8,device=device) * 17
        mask_cameras = voxel_semantics.new_zeros(B,400,300,24).bool()
        pc_range = torch.tensor([-80,-60,-3,80,60,6.6],device=device)
        voxel_size = torch.tensor([0.4,0.4,0.4],device=device)
        for b in range(B):
            src2dst = torch.matmul(torch.inverse(torch.from_numpy(dst_tran[b])),\
            torch.from_numpy(src_tran[b])).to(device).float()
            gt_points = torch.matmul(src2dst[:3,:3], gt_points_list[b].transpose(0,1)).transpose(0,1) + src2dst[:3,3]
            warped_coords = torch.floor((gt_points-pc_range[None,:3])//voxel_size[None,:]).long()
            mask = torch.logical_and(warped_coords>=0,warped_coords<num_voxel[None,:]).all(-1)
            voxel_semantics[b,warped_coords[mask,0],warped_coords[mask,1],warped_coords[mask,2]] = gt_labels_list[b][mask].to(torch.uint8)
            mask_cameras[b,warped_coords[mask,0],warped_coords[mask,1],warped_coords[mask,2]] = gt_masks_list[b][mask]
        
        voxel_semantics[:,100:300,50:250,5:21] = gt_fu1_dict['voxel_semantic']
        mask_cameras[:,100:300,50:250,5:21] = gt_fu1_dict['mask_camera']
        gt_points_list,gt_masks_list,gt_labels_list = \
            self.pts_bbox_head.get_sparse_voxels_4d(voxel_semantics, mask_cameras)
        loss_cls, loss_pts = self.pts_bbox_head.loss_single(cls_pred,points_pred,gt_points_list,gt_masks_list,gt_labels_list,pc_range = self.pc_range)
        
        return loss_cls,loss_pts


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats
    @staticmethod
    def get_cls_target(
        
        reg_preds, 
        reg_target,
        reg_weight,
    ):

        dist = torch.linalg.norm(reg_target.unsqueeze(1) - reg_preds, dim=-1)
        dist = dist * reg_weight.unsqueeze(1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        # mode_idx = torch.zeros(bs,num_pred, dtype=torch.int64).to(reg_preds.device)
        return mode_idx

    @staticmethod
    def get_best_reg(
        reg_preds,
        cls_target,
        reg_target,
        reg_weight,
    ):
        bs, mode, ts, d = reg_preds.shape

        mode_idx = cls_target[..., None, None, None].repeat(1, 1, ts, d)
        best_reg = torch.gather(reg_preds, 1, mode_idx).squeeze(1)
        return best_reg

    def extract_feat(self, img, img_metas):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        assert img.dim() == 5

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        # move some augmentations to GPU
        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)


        if self.training and self.stop_prev_grad > 0:
            H, W = input_shape
            img = img.reshape(B, -1, 6, C, H, W)

            img_grad = img[:, :self.stop_prev_grad]
            img_nograd = img[:, self.stop_prev_grad:]

            all_img_feats = [self.extract_img_feat(img_grad.reshape(-1, C, H, W))]

            with torch.no_grad():
                self.eval()
                for k in range(img_nograd.shape[1]):
                    all_img_feats.append(self.extract_img_feat(img_nograd[:, k].reshape(-1, C, H, W)))
                self.train()

            img_feats = []
            for lvl in range(len(all_img_feats[0])):
                C, H, W = all_img_feats[0][lvl].shape[1:]
                img_feat = torch.cat([feat[lvl].reshape(B, -1, 6, C, H, W) for feat in all_img_feats], dim=1)
                img_feat = img_feat.reshape(-1, C, H, W)
                img_feats.append(img_feat)
        else:
            img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        return img_feats_reshaped

    def init_weights(self):
        if self.pts_bbox_head is not None:
            self.pts_bbox_head.init_weights()
        if self.forecasting:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_branch_forecasting[-1].bias,bias_init)


    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if self.lm_head is not None:
            # self.lm_head.model.image_casual = self.image_casual
            self.lm_head.num_global_query = self.num_global_query
            if self.lm_head.__class__.__name__ is 'PeftModelForCausalLM':
                self.lm_head.base_model.model.model.image_casual = self.image_casual
                self.lm_head.base_model.model.num_global_query = self.num_global_query

        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      img_metas=None,
                      img=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      input_ids=None,
                      vlm_labels=None,
                      can_bus=None,
                      gt_planning=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        B = img.shape[0]

        img_feats = self.extract_feat(img, img_metas)

        outs = self.pts_bbox_head(img_feats, img_metas, can_bus)

        if self.pts_bbox_head.signal_4d=='occ_4d':
            temporal2ego = [img_meta['temporal2ego'] for img_meta in img_metas]
            loss_inputs = [voxel_semantics,mask_camera,outs,temporal2ego]
            losses,gt_fu0_dict = self.pts_bbox_head.loss_4d(*loss_inputs)
        elif self.pts_bbox_head.signal_4d=='lidar':
            ego2lidar = [img_meta['ego2lidar'] for img_meta in img_metas]
            points = data['points']
            loss_inputs = [voxel_semantics,mask_camera,outs,ego2lidar,points]
            losses,gt_fu0_dict = self.pts_bbox_head.loss_lidar(*loss_inputs)
        else:
            loss_inputs = [voxel_semantics, mask_camera, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)


        if self.tokenizer is not None:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            
            vlm_labels = torch.nn.utils.rnn.pad_sequence(vlm_labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)            
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            vlm_labels = vlm_labels[:, :self.tokenizer.model_max_length]
            vlm_attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            input_ids = None
            vlm_labels = None
            vlm_attn_mask = None
        
        if self.lm_head:
            query_positions = outs['all_refine_pts'][-1].mean(-2)
            command = data['command']
            can_bus = torch.cat([can_bus.float(),data['command'][:,None]],dim=-1)
    
            can_bus_embed = self.can_bus_embed(can_bus.float())
            query_features = outs['query_features']
            query_positions = outs['all_refine_pts'][-1]
            global_queries = outs['global_query']
            B,N,P,C = query_positions.shape

            query_features_identity = query_features
            query_features_ = query_features + self.position_encoder(query_positions.reshape(B,N,P*C))
            query_features = self.projector(query_features_)

            query_features = torch.cat([can_bus_embed.unsqueeze(-2),query_features],dim=-2)
            if global_queries is not None:
                
                global_queries = self.global_attention(global_queries,query_features_)
                global_queries = self.global_projector(global_queries)
                query_features = torch.cat([query_features,global_queries],dim=-2)

            if not self.forecasting:
                query_features = query_features.to(self.lm_head.dtype)
                vlm_loss = self.lm_head(input_ids=input_ids, attention_mask=vlm_attn_mask,labels=vlm_labels,vit_embeds=query_features,use_cache=False,)
                losses.update(vlm_loss=vlm_loss[0])
            else:
                vlm_loss = self.lm_head(input_ids=input_ids, attention_mask=vlm_attn_mask,labels=vlm_labels,vit_embeds=query_features.to(self.lm_head.dtype),use_cache=False)
                losses.update(vlm_loss=vlm_loss[0])
                fu_time_stamp = [1,2,3,4,5,6]
                loss_inputs = [voxel_semantics,mask_camera,outs,temporal2ego]

                query_id = input_ids == self.lm_head.img_context_token_id
                query_features = vlm_loss['hidden_states'][query_id].reshape(B,1+N+self.num_global_query,-1).to(query_features.dtype)
                query_features = query_features[:,1:1+N]

                query_features = self.query_decoder(query_features)

                query_features = self.res_layer(torch.cat([query_features,query_features_identity],dim=-1))
                temporal_ego_states = data['temporal_ego_states']
                gt_fu_mask = data['gt_fu_mask'].all(-1).float()
                if self.use_gen_token:
                    ego_feature = vlm_loss['hidden_states'][index==32000]
                    ego_feature = self.ego_decoder(torch.concat([ego_feature,can_bus.float(),temporal_ego_states],dim=-1))
                
                canbus_fu = np.stack([img_metas[i]['canbus_fu'] for i in range(B)],0)
                canbus_fu = torch.from_numpy(canbus_fu).to(query_features.device)
                canbus_fu_stamp = torch.arange(6, dtype=torch.float32, device=query_features.device)[None,:,None].repeat(B,1,1)
                canbus_fu = torch.concat([canbus_fu,canbus_fu_stamp],dim=-1).float()
                canbus_fu_embedding = self.canbus_fu_embedding(canbus_fu)

                query_positions = query_positions.detach()
                query_positions_4d = torch.cat([query_positions,torch.ones_like(query_positions[:,:,:,:1])*0],dim=-1)
                query_feature = query_features + self.position_encoder_4d(query_positions_4d.flatten(-2,-1))

                gt_planning = gt_planning.squeeze(1)[...,:2].float()
                gt_planning_diff=torch.cat([gt_planning[:,:1,:],torch.diff(gt_planning,dim=-2)],dim=-2).float()
                if self.use_gen_token:

                    cls_weight = gt_fu_mask.any(-1)
                    bs_ids = torch.arange(B,device=query_feature.device)
                    # ego_feature = self.waypoint_attention(ego_feature.unsqueeze(1), query_feature).squeeze(1)
                    waypoint_cls = self.waypoint_cls(ego_feature).reshape(B,3,self.num_mode)
                    waypoint_cls = waypoint_cls[bs_ids,command]
                    
                    anchor_traj_all = self.plan_anchor.data.unsqueeze(0).repeat(B,1,1,1,1)
                    anchor_traj = anchor_traj_all[bs_ids,command]
                    
                    cls_target = self.get_cls_target(anchor_traj,gt_planning,gt_fu_mask)

                    losses['waypoint_cls_loss'] = self.waypoint_cls_loss(waypoint_cls,cls_target,weight=cls_weight)*5
                    anchor_traj_diff = torch.concat([anchor_traj[:,:,:1,:],anchor_traj[:,:,1:]-anchor_traj[:,:,:-1]],dim=-2)
                    timesteps = torch.randint(0, 40,(B,), device=ego_feature.device) 
                    noise = torch.randn_like(anchor_traj_diff)
                    waypoint_pred_norm = self.norm_odo(anchor_traj_diff)
                    noisy_traj_points = self.diffusion_scheduler.add_noise(original_samples=waypoint_pred_norm,noise=noise,timesteps=timesteps).float()
                    noisy_traj_points = torch.clamp(noisy_traj_points,min=-1, max=1)
                    noisy_traj_points = self.denorm_odo(noisy_traj_points)
                    traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
                    traj_pos_embed = traj_pos_embed.flatten(-2)

                    merged_ego_states = torch.cat([can_bus,temporal_ego_states],dim=-1)
                    traj_feature = self.traj_encoder(traj_pos_embed)
                    time_embed = self.time_mlp(timesteps)
                    waypoint_pred = self.diff_decoder(traj_feature,noisy_traj_points, query_feature,ego_feature.unsqueeze(1),time_embed.unsqueeze(1),None)
                    best_reg = self.get_best_reg(waypoint_pred,cls_target,gt_planning,gt_fu_mask)

                    waypoint_diff_loss = self.waypoint_reg_loss(best_reg,gt_planning_diff,weight=gt_fu_mask.unsqueeze(-1))
                    losses['waypoint_diff_loss'] = waypoint_diff_loss

                occ_forecasting_list = []
                for t in fu_time_stamp:
                    weight = 1 if (t-1)*200<self.iter else 0.2

                    if t>1:
                        query_positions = query_positions.detach()
                        query_positions_4d = torch.cat([query_positions,torch.ones_like(query_positions[:,:,:,:1])*(t-1)*0.5],dim=-1)
                        query_feature = query_features + self.position_encoder_4d(query_positions_4d.flatten(-2,-1))
                        # print(t,'  ',self.res_connect,'  ',query_features.isnan().sum(),'----------',query_features.shape, ' --------', canbus_fu_embedding.shape)
                    query_feature = query_feature + canbus_fu_embedding[:,t-1:t]

                    query_positions_offset = self.reg_branch_forecasting(query_feature).reshape(B,N,-1,3)*0.5
                    query_cls = self.cls_branch_forecasting(query_feature).reshape(B,N,-1,17)
                    query_positions = self.refine_points(query_positions,query_positions_offset)

                    src_trans = [np.eye(4) for _ in temporal2ego]
                    dst_trans = [temp_trans[t-1] for temp_trans in temporal2ego]
                    gt_fu1_dict = {'voxel_semantic':voxel_semantics[:,t],'mask_camera':mask_camera[:,t]}
                    loss_cls,loss_pts = self.loss_fu_occ(gt_fu0_dict,gt_fu1_dict,src_trans,dst_trans,query_positions,query_cls)
                    
                    losses[f'cls_loss_fu{t}'] = loss_cls * weight 
                    losses[f'pts_loss_fu{t}'] = loss_pts * weight 

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False,**data):
        can_bus=data['can_bus'][0]
        outs = self.pts_bbox_head(x, img_metas,can_bus)
        results = [{}]
        mmcv.mkdir_or_exist(self.save_path)
        self.feature_distill=False
    
        pred_dict = {'cls_scores':outs['all_cls_scores'][-1],'refine_pts':outs['all_refine_pts'][-1]}
        occ_result = self.pts_bbox_head.get_occ(pred_dict, img_metas[0])
        results[0]['occ_pred'] = occ_result[0].cpu().to(torch.uint8)

        if self.lm_head:
            can_bus = torch.cat([data['can_bus'][0],data['command'][0][:,None]],dim=-1)
            
            can_bus_embed = self.can_bus_embed(can_bus)
            
            query_features = outs['query_features']
            query_positions = outs['all_refine_pts'][-1]
            global_queries = outs['global_query']
            B,N,P,C = query_positions.shape

            query_features_identity = query_features

            query_features_ = query_features + self.position_encoder(query_positions.reshape(B,N,P*C))
            query_features = self.projector(query_features_)
            query_features = torch.cat([can_bus_embed.unsqueeze(-2),query_features,],dim=-2)

            if global_queries is not None:
                # global_queries = self.global_query_embedding.weight[None,:,:].repeat(B,1,1)
                global_queries = self.global_attention(global_queries,query_features_)
                global_queries = self.global_projector(global_queries)
                query_features = torch.cat([query_features,global_queries],dim=-2)

            generated_text = []
            token = img_metas[0]['sample_idx']  #InternVL
            conv_save = []
            question_list = []
            gt_answer_list = []
            conv_path = "./data/nuscenes/conv/val/"
            conv_path = os.path.join(conv_path, token + ".json")
            desc_path = "./data/nuscenes/desc/val/"
            desc_path = os.path.join(desc_path, token + ".json")
            with open(conv_path, 'r') as f:
                conv_gt = json.load(f)
                # add the questions to the question_list and the gt_answer to the gt_answer_list
                for i in range(len(conv_gt)):
                    question_list.append(conv_gt[i]['question'])
                    gt_answer_list.append(conv_gt[i]['answer'])
            with open(desc_path, 'r') as f:
                desc_gt = json.load(f)
                desc = desc_gt['description']
                action = desc_gt['action']
                question_list = question_list + ["What action should be taken in the current driving scenario?"]
                gt_answer_list = gt_answer_list + [action]
            question_list = ["Can you provide a summary of the current driving scenario based on the input images?"] \
                            + question_list
            gt_answer_list = [desc] + gt_answer_list

            question_list = question_list + ["Please try to imagine the future scene."]
            gt_answer_list = gt_answer_list + [
                "Sure, I can try to envision the future scene based on the understanding of the driving scenario."]

            assert len(question_list) == len(gt_answer_list)
            question_list[0] = "<image>\n" + question_list[0]

            history_list = []
            conv = []
            self.lm_head.num_iamge_token = query_features.shape[1]
            for i, question in enumerate(question_list):
                if self.eval_occ:
                    break 
                response = self.lm_head.chat(
                    tokenizer=self.tokenizer,
                    pixel_values = query_features,
                    visual_features = query_features,
                    question=question,
                    generation_config = self.chat_cfg,
                    verbose=False,
                    # history=history_list
                )
                conv.append({'from': 'human', 'value': question})
                conv.append({'from': 'gpt', 'value': response})
                generated_text.append({
                    "Q": question,
                    "A": response,
                    "G": gt_answer_list[i]
                })

            with open(os.path.join(self.save_path,img_metas[0]['sample_idx']), 'w') as file:
                json.dump(generated_text, file,indent=1)
            results[0]['generated_text'] = generated_text

            if self.forecasting:

                temporal_ego_states = data['temporal_ego_states'][0]
                # input_ids = data['input_ids'][0][0].unsqueeze(0)
                fu_time_stamp = [1,2,3,4,5,6]
                # vlm_attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
                # vlm_labels = torch.ones_like(input_ids)
                vlm_loss,index = self.lm_head.chat(
                        tokenizer=self.tokenizer,
                        pixel_values = query_features,
                        visual_features = query_features,
                        question=question,
                        generation_config = self.chat_cfg,
                        verbose=False,
                        generate=False,
                        # history=history_list
                    )
                bs_ids = torch.arange(B)
                command = data['command'][0]
                query_features = vlm_loss['hidden_states'][index==self.lm_head.img_context_token_id].reshape(B,1+N+self.num_global_query,-1).to(query_features.dtype)
                ego_feature,query_features = query_features[:,0],query_features[:,1:1+N]
                # ego_feature = vlm_loss['hidden_states'][index==-3].float()
                # query_features = vlm_loss['hidden_states'][index==-200].float().reshape(B,N,-1)
                
                query_features = self.query_decoder(query_features)
                query_features = self.res_layer(torch.cat([query_features,query_features_identity],dim=-1))                
                canbus_fu = np.stack([img_metas[i]['canbus_fu'] for i in range(B)],0)
                canbus_fu = torch.from_numpy(canbus_fu).to(query_features.device)
                canbus_fu_stamp = torch.arange(6, dtype=torch.float32, device=query_features.device)[None,:,None].repeat(B,1,1)
                canbus_fu = torch.concat([canbus_fu,canbus_fu_stamp],dim=-1).float()
                canbus_fu_embedding = self.canbus_fu_embedding(canbus_fu)

                query_positions = query_positions.detach()
                query_positions_4d = torch.cat([query_positions,torch.ones_like(query_positions[:,:,:,:1])*0],dim=-1)
                query_feature = query_features + self.position_encoder_4d(query_positions_4d.flatten(-2,-1))

                if self.use_gen_token:
                    ego_feature = vlm_loss['hidden_states'][index==32000]
                    
                    ego_feature = self.ego_decoder(torch.concat([ego_feature,can_bus.float(),temporal_ego_states],dim=-1))
                    device=ego_feature.device
                    waypoint_cls = self.waypoint_cls(ego_feature).reshape(B,3,self.num_mode)
                    waypoint_cls = waypoint_cls[bs_ids,command]
                    anchor_traj_all = self.plan_anchor.data.unsqueeze(0).repeat(B,1,1,1,1)
                    anchor_traj = anchor_traj_all[bs_ids,command]
                    anchor_traj_diff = torch.concat([anchor_traj[:,:,:1,:],anchor_traj[:,:,1:]-anchor_traj[:,:,:-1]],dim=-2)
                    trunc_timesteps = torch.ones((B,), device=device, dtype=torch.long) * 8
                    noise = torch.randn_like(anchor_traj_diff)
                    img = self.norm_odo(anchor_traj_diff)
                    
                    step_num=2
                    
                    self.diffusion_scheduler.set_timesteps(1000, device=device)
                    step_ratio = 10 / step_num
                    roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
                    roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

                    img = self.diffusion_scheduler.add_noise(original_samples=img,noise=noise,timesteps=trunc_timesteps).float()
                
                    for k in roll_timesteps[:]:
                        img = torch.clamp(img,min=-1, max=1)
                        noisy_traj_points = self.denorm_odo(img)
                        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
                        traj_pos_embed = traj_pos_embed.flatten(-2)
                        traj_feature = self.traj_encoder(traj_pos_embed)
                        time_embed = self.time_mlp(k.unsqueeze(0))
                        waypoint_pred = self.diff_decoder(traj_feature,noisy_traj_points, query_feature,ego_feature.unsqueeze(1),time_embed.unsqueeze(1),None)
                        
                        x_start = waypoint_pred
                        x_start = self.norm_odo(x_start)
                        img=self.diffusion_scheduler.step(
                            model_output=x_start,
                            timestep=k,
                            sample=img
                        ).prev_sample
                        
                        waypoint_index = torch.argmax(waypoint_cls,-1)
                        waypoint_pred = torch.gather(waypoint_pred,1,waypoint_index[:,None,None,None].repeat(1,1,self.num_fu_traj,2)).squeeze(1)

                occ_pred_list = []
                
                pred_dict = {'cls_scores':outs['all_cls_scores'][-1],'refine_pts':outs['all_refine_pts'][-1]}
                occ_pred_list.append(self.pts_bbox_head.get_occ(pred_dict, img_metas[0], rescale=rescale))
                
                for t in fu_time_stamp:
                    if t>1:
                        query_positions = query_positions.detach()
                        query_positions_4d = torch.cat([query_positions,torch.ones_like(query_positions[:,:,:,:1])*(t-1)*0.5],dim=-1)                
                        query_feature = query_features + self.position_encoder_4d(query_positions_4d.flatten(-2,-1))
                    
                    query_feature = query_feature + canbus_fu_embedding[:,t-1:t]
                    
                    query_positions_offset = self.reg_branch_forecasting(query_feature).reshape(B,N,-1,3)*0.5
                    query_cls = self.cls_branch_forecasting(query_feature).reshape(B,N,-1,17)
                    query_positions = self.refine_points(query_positions,query_positions_offset)
                    pred_dict = {'cls_scores':query_cls,'refine_pts':query_positions}
                    occ_pred_list.append(self.pts_bbox_head.get_occ(pred_dict, img_metas[0], rescale=rescale))
                results[0]['occ_forecasting'] = torch.cat(occ_pred_list,0).cpu()[[0,2,4,6]].to(torch.uint8)
                if self.use_gen_token:
                    results[0]['traj_forecasting'] = waypoint_pred.squeeze(0).cpu().numpy().tolist()
                results[0]['token'] = img_metas[0]['sample_idx']
        
        return results
    
    def simple_test(self, img_metas, img=None, rescale=False, **data):
        world_size = get_dist_info()[1]
        if 'input_ids' in data:
            data['input_ids'] = data['input_ids'][0]
        if world_size == 1:  # online
            return self.simple_test_online(img_metas, img, rescale, **data)
        
        else:  # offline
            return self.simple_test_offline(img_metas, img, rescale,**data)

    def simple_test_offline(self, img_metas, img=None, rescale=False,**data):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        return self.simple_test_pts(img_feats, img_metas, rescale=rescale,**data)

    def simple_test_online(self, img_metas, img=None, rescale=False,**data):
        self.fp16_enabled = False
        assert len(img_metas) == 1  # batch_size = 1

        B, N, C, H, W = img.shape
        img = img.reshape(B, N//6, 6, C, H, W)

        img_filenames = img_metas[0]['filename']
        num_frames = len(img_filenames) // 6
        # assert num_frames == img.shape[1]

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len(img_filenames))]

        img_feats_list, img_metas_list = [], []

        # extract feature frame by frame
        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                item = img_metas[0][k]
                if isinstance(item, list) and (len(item) == 6 * num_frames):
                    img_metas_curr[0][k] = [item[j] for j in img_indices]
                else:
                    img_metas_curr[0][k] = item

            if img_filenames[img_indices[0]] in self.memory:
                # found in memory
                img_feats_curr = self.memory[img_filenames[img_indices[0]]]
            else:
                # extract feature and put into memory
                img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
                self.memory[img_filenames[img_indices[0]]] = img_feats_curr
                self.queue.put(img_filenames[img_indices[0]])
                while self.queue.qsize() >= 16:  # avoid OOM
                    pop_key = self.queue.get()
                    self.memory.pop(pop_key)

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)

        # reorganize
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)

        # run occupancy predictor
        return self.simple_test_pts(img_feats, img_metas, rescale=rescale,**data)
    

    def refine_points(self, points_proposal, points_delta):
        B, Q, N = points_delta.shape[:3]
        points_delta = points_delta.reshape(B, Q, N, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    def norm_odo(self, gt_ego_fut_trajs):
        # bs, ego_fut_ts, _ = gt_ego_fut_trajs.shape
        odo_info_fut_y = gt_ego_fut_trajs[..., 0:1]
        odo_info_fut_x = gt_ego_fut_trajs[..., 1:2]

        odo_info_fut_x = odo_info_fut_x / 3
        odo_info_fut_x = odo_info_fut_x.clamp(-1, 1)
        odo_info_fut_y = (odo_info_fut_y+0.5) / 8.1
        odo_info_fut_y = odo_info_fut_y.clamp(0, 1)
        odo_info_fut_y = odo_info_fut_y * 2 - 1
        odo_info_fut = torch.cat([odo_info_fut_y, odo_info_fut_x], dim=-1)
        # odo_info_fut = odo_info_fut.reshape(-1,self.ego_fut_ts, 2)
        return odo_info_fut
    
    def denorm_odo(self, noisy_traj_points):
        # bs, ego_fut_ts, _ = noisy_traj_points.shape
        odo_info_fut_y = noisy_traj_points[..., 0:1]
        odo_info_fut_x = noisy_traj_points[..., 1:2]

        odo_info_fut_x = odo_info_fut_x * 3
        # odo_info_fut_x = odo_info_fut_x.clamp(-1, 1)
        odo_info_fut_y = (odo_info_fut_y+1) / 2 * 8.1 - 0.5
        # odo_info_fut_y = odo_info_fut_y.clamp(-1, 1)
        odo_info_fut = torch.cat([odo_info_fut_y, odo_info_fut_x], dim=-1)
        return odo_info_fut