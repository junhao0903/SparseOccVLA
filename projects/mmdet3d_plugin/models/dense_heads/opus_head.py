import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops import knn, Voxelization
from mmdet.core import multi_apply
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from .bbox.utils import decode_points,resort_points,purify_points
import numpy as np
# import torch_scatter
@HEADS.register_module()

class OPUSHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 transformer=None,
                 pc_range=[],
                 empty_label=17,
                 voxel_size=[],
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_pts=dict(type='L1Loss'),
                 init_cfg=None,
                 pts_resort=False,
                 signal_4d='occ_4d',
                 **kwargs):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.empty_label = empty_label
        self.loss_cls = build_loss(loss_cls)
        self.loss_pts = build_loss(loss_pts)
        self.transformer = build_transformer(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims
        self.pts_resort = pts_resort
        self.voxel_generator = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points=10, 
            max_voxels=50000,
            deterministic=False
        )

        # prepare scene
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)

        self.signal_4d = signal_4d
        if self.signal_4d=='occ_4d':
            self.points_scale_branch = nn.Sequential(
            nn.Linear(39,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,3),

        )
        elif signal_4d=='lidar':
            for name,param in self.named_parameters():
                if 'cls_branch' in name:
                    param.requires_grad = False


        self._init_layers()

    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0, 1)

    def init_weights(self):
        self.transformer.init_weights()
    def purify_points(self,points,ego2lidar):
        points = points[:,:3]
        ego2lidar = torch.from_numpy(ego2lidar).to(points)
        lidar2ego = torch.linalg.inv(ego2lidar)
        points_pad = torch.cat([points,torch.ones_like(points[:,:1])],dim=-1)
        points = torch.matmul(points_pad,lidar2ego.transpose(0,1))
        dis = torch.norm(points,dim=-1,p=2)
        mask = dis>5
        # mask = (points[:,0]<40) & (points[:,0]>-40) & (points[:,1]>-40) & (points[:,1]<40) & (points[:,2]<5.4) & (points[:,2]>-1)
        points = points[mask,:3]
        
        _, voxels, num_pts = self.voxel_generator(points)
        points = voxels[:,[2,1,0]]*self.voxel_size[None,:] + self.pc_range[None,:3]
        return points.contiguous()
    def forward(self, mlvl_feats, img_metas,can_bus=None):
        B, Q, = mlvl_feats[0].shape[0], self.num_query
        init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1)
        if self.signal_4d=='occ_4d':
            for b in range(B):
                can_bus = can_bus.clone()
                if self.training:
                    can_bus[b,[10,23,36]]*=img_metas[b]['num_fu_frame_real']//6
            points_scale = self.points_scale_branch(can_bus)
            points_scale = (torch.tanh(points_scale) +1) / 2 * (1.5-0.8) +0.8
            init_points = init_points * points_scale[:,None,None,:]

        query_feat = init_points.new_zeros(B, Q, self.embed_dims)
        if self.pts_resort:
            init_points,query_feat = resort_points(init_points.squeeze(-2),query_feat)
            init_points = init_points.unsqueeze(-2)
        cls_scores, refine_pts,query_feat,global_query = self.transformer(
            init_points,
            query_feat,
            mlvl_feats,
            img_metas=img_metas,
        )

        return dict(init_points=init_points,
                    all_cls_scores=cls_scores,
                    all_refine_pts=refine_pts,
                    query_features=query_feat,
                    global_query=global_query)

    def get_dis_weight(self, pts):
        max_dist = torch.sqrt(
            self.scene_size[0] ** 2 + self.scene_size[1] ** 2)
        centers = (self.pc_range[:3] + self.pc_range[3:]) / 2
        dist = (pts - centers[None, ...])[..., :2]
        dist = torch.norm(dist, dim=-1)
        return dist / max_dist + 1
    
    def discretize(self, pts, clip=True, decode=False):
        loc = torch.floor((pts - self.pc_range[:3]) / self.voxel_size)
        if clip:
            loc[..., 0] = loc[..., 0].clamp(0, self.voxel_num[0] - 1)
            loc[..., 1] = loc[..., 1].clamp(0, self.voxel_num[1] - 1)
            loc[..., 2] = loc[..., 2].clamp(0, self.voxel_num[2] - 1)

        return loc.long() if not decode else \
            (loc + 0.5) * self.voxel_size + self.pc_range[:3]

    @torch.no_grad()
    def _get_target_single(self, refine_pts, gt_points,gt_masks, gt_labels):
        # knn to apply Chamfer distance
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()
        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()
        gt_paired_pts = refine_pts[gt_paired_idx]
        pred_paired_pts = gt_points[pred_paired_idx]

        # cls assignment
        refine_pts_labels = gt_labels[pred_paired_idx]
        cls_weights = self.train_cfg.get('cls_weights', [1] * self.num_classes)
        cls_weights = refine_pts.new_tensor(cls_weights)
        label_weights = cls_weights * \
            self.get_dis_weight(pred_paired_pts)[..., None]

        # gt side assignment
        empty_dist_thr = self.train_cfg.get('empty_dist_thr', 0.2)
        empty_weights = self.train_cfg.get('empty_weights', 5)

        gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        dist = torch.norm(gt_points - gt_paired_pts, dim=-1)
        mask = (dist > empty_dist_thr) & gt_masks
        gt_pts_weights[mask] = empty_weights

        rare_classes = self.train_cfg.get('rare_classes', [0,2,5,8])
        rare_weights = self.train_cfg.get('rare_weights', 10)
        for cls_idx in rare_classes:
            mask = (gt_labels == cls_idx) & gt_masks
            gt_pts_weights[mask] = gt_pts_weights[mask].clamp(min=rare_weights)
        if True:
            dis = torch.norm(pred_paired_pts - refine_pts,dim=-1,p=2,keepdim=True)
            label_weights = label_weights * torch.clamp(1/dis,max=1,min=0.4)
            gt_pts_weights[(gt_labels>=15) | (gt_labels==11)] *=0.5       
             
        return (refine_pts_labels, gt_paired_idx, pred_paired_idx, label_weights, 
                gt_pts_weights)
    def _get_target_single_pts(self, refine_pts, gt_points,gt_masks):
        # knn to apply Chamfer distance
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()
        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()
        gt_paired_pts = refine_pts[gt_paired_idx]
        pred_paired_pts = gt_points[pred_paired_idx]

        # cls assignment
        # refine_pts_labels = gt_labels[pred_paired_idx]

        # gt side assignment
        empty_dist_thr = self.train_cfg.get('empty_dist_thr', 0.2)
        empty_weights = self.train_cfg.get('empty_weights', 5)

        gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        dist = torch.norm(gt_points - gt_paired_pts, dim=-1)
        mask = (dist > empty_dist_thr) & gt_masks
        gt_pts_weights[mask] = empty_weights

        rare_classes = self.train_cfg.get('rare_classes', [0,2,5,8])
        rare_weights = self.train_cfg.get('rare_weights', 10)
        for cls_idx in rare_classes:
            mask = gt_masks
            gt_pts_weights[mask] = gt_pts_weights[mask].clamp(min=rare_weights)
        if True:
            distance = torch.norm(pred_paired_pts - refine_pts,dim=-1,p=2,keepdim=True)
            # label_weights = label_weights * torch.clamp(1/dis,max=1,min=0.4)
            # gt_pts_weights[(gt_labels>=15) | (gt_labels==11)] *=0.5       
            distance = torch.clamp(distance,max=5)
            labels = torch.exp(- (distance ** 2) / (2 * 0.2 ** 2))
        return (distance, gt_paired_idx, pred_paired_idx,gt_pts_weights)
    
    def get_targets(self):
        # To instantiate the abstract method
        pass

    def loss_single(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_masks_list,
                    gt_labels_list,
                    pc_range=None):
        num_imgs = cls_scores.size(0) # B
        cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        
        refine_pts = decode_points(refine_pts, self.pc_range if pc_range is None else pc_range)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights,
         gt_pts_weights) = multi_apply(
             self._get_target_single, refine_pts_list, gt_points_list, 
              gt_masks_list,gt_labels_list)
        
        gt_paired_pts, pred_paired_pts= [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        # concatenate all results from different samples
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        cls_weights = torch.cat(cls_weights)
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)

        # calculate loss cls
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 weight=cls_weights,
                                 avg_factor=cls_scores.shape[0])
        # calculate loss pts
        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts, 
                                  pred_paired_pts,
                                  avg_factor=pred_pts.shape[0])

        return loss_cls, loss_pts

    def loss_single_pts(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_masks_list,
                    pc_range=None):
        num_imgs = refine_pts.size(0) # B
        cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        
        refine_pts = decode_points(refine_pts, self.pc_range if pc_range is None else pc_range)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        (labels_list,gt_paired_idx_list, pred_paired_idx_list,gt_pts_weights) = multi_apply(
             self._get_target_single_pts, refine_pts_list, gt_points_list, 
              gt_masks_list)
        
        gt_paired_pts, pred_paired_pts= [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        # concatenate all results from different samples
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        # cls_weights = torch.cat(cls_weights)
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)

        # calculate loss cls
        loss_cls = self.loss_pts(cls_scores,
                                 labels,
                                 weight=torch.clamp(1/labels,max=10),
                                 avg_factor=cls_scores.shape[0])
        # loss_cls = F.binary_cross_entropy_with_logits(cls_scores, labels)
        # calculate loss pts
        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts, 
                                  pred_paired_pts,
                                  avg_factor=pred_pts.shape[0])

        return loss_cls,loss_pts
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, mask_camera, preds_dicts):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores'] # 6 ,B,2k4,32,17
        all_refine_pts = preds_dicts['all_refine_pts']

        num_dec_layers = len(all_cls_scores)
        gt_points_list,gt_masks_list, gt_labels_list = \
            self.get_sparse_voxels(voxel_semantics, mask_camera)
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_pts = multi_apply(
            self.loss_single, all_cls_scores, all_refine_pts, 
            all_gt_points_list,all_gt_masks_list, all_gt_labels_list)

        loss_dict = dict()
        # loss of init_points
        if init_points is not None:
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)
            _, init_loss_pts = self.loss_single(
                pseudo_scores, init_points, gt_points_list, 
                 gt_masks_list, gt_labels_list)
            loss_dict['init_loss_pts'] = init_loss_pts

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1] * self.train_cfg.get('cls_weight',1)
        loss_dict['loss_pts'] = losses_pts[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls[:-1], losses_pts[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i * self.train_cfg.get('cls_weight',1)
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i * self.train_cfg.get('reg_weight',1)
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def loss_4d(self, voxel_semantics, mask_camera, preds_dicts, temporal2ego):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores'] # 6 ,B,2k4,32,17
        all_refine_pts = preds_dicts['all_refine_pts']

        num_dec_layers = len(all_cls_scores)
        voxel_semantics,mask_cameras = self.get_sparse_voxels_stack(voxel_semantics[:,0],voxel_semantics[:,1:],temporal2ego,mask_camera)

        gt_points_list,gt_masks_list, gt_labels_list = \
            self.get_sparse_voxels_4d(voxel_semantics, mask_cameras)
        # gt_masks_list = [torch.ones_like(gt_points[:,0],dtype=torch.bool) for gt_points in gt_points_list]

        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_pts = multi_apply(
            self.loss_single, all_cls_scores, all_refine_pts, 
            all_gt_points_list,all_gt_masks_list, all_gt_labels_list)

        loss_dict = dict()
        # loss of init_points
        if init_points is not None:
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)
            _, init_loss_pts = self.loss_single(
                pseudo_scores, init_points, gt_points_list, 
                gt_masks_list, gt_labels_list)
            loss_dict['init_loss_pts'] = init_loss_pts

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1] * self.train_cfg.get('cls_weight',1)
        loss_dict['loss_pts'] = losses_pts[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls[:-1], losses_pts[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i * self.train_cfg.get('cls_weight',1)
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i * self.train_cfg.get('reg_weight',1)
            num_dec_layer += 1
        gt_fu0_dict = {'gt_points_list':gt_points_list,'gt_masks_list':gt_masks_list,'gt_labels_list':gt_labels_list}

        return loss_dict, gt_fu0_dict

    def loss_lidar(self, voxel_semantics, mask_camera, preds_dicts, ego2lidar,points):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores'] # 6 ,B,2k4,32,17
        all_refine_pts = preds_dicts['all_refine_pts']

        num_dec_layers = len(all_cls_scores)
        # voxel_semantics,mask_cameras = self.get_sparse_voxels_stack(voxel_semantics[:,0],voxel_semantics[:,1:],temporal2ego,mask_camera)

        # gt_points_list,gt_masks_list, gt_labels_list = \
        #     self.get_sparse_voxels_4d(voxel_semantics, mask_cameras)
        points = [self.purify_points(pts,e2l) for pts,e2l in zip(points,ego2lidar)]
        gt_points_list = points
        gt_masks_list = [torch.ones_like(gt_points[:,0],dtype=torch.bool) for gt_points in gt_points_list]
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        # all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        loss_cls,losses_pts = multi_apply(
            self.loss_single_pts,all_cls_scores,  all_refine_pts, 
            all_gt_points_list,all_gt_masks_list,)

        loss_dict = dict()
        # loss of init_points
        if init_points is not None:
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)
            _,init_loss_pts = self.loss_single_pts(
                 pseudo_scores,init_points, gt_points_list, 
                gt_masks_list)
            loss_dict['init_loss_pts'] = init_loss_pts

        # loss from the last decoder layer
        loss_dict['loss_cls'] = loss_cls[-1]
        loss_dict['loss_pts'] = losses_pts[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(loss_cls[:-1],losses_pts[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i * self.train_cfg.get('cls_weight',1)
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i * self.train_cfg.get('reg_weight',1)
            num_dec_layer += 1
        gt_fu0_dict = {'gt_points_list':gt_points_list,'gt_masks_list':gt_masks_list}

        return loss_dict, gt_fu0_dict


    
    def get_occ(self, pred_dict, img_metas=None, rescale=False):
        # all_cls_scores = pred_dicts['all_cls_scores']
        # all_refine_pts = pred_dicts['all_refine_pts']
        cls_scores = pred_dict['cls_scores'].sigmoid()
        refine_pts = pred_dict['refine_pts']

        batch_size = refine_pts.shape[0]
        ctr_dist_thr = self.test_cfg.get('ctr_dist_thr', 3.)
        score_thr = self.test_cfg.get('score_thr', 0.)

        result_list = []
        for i in range(batch_size):
            refine_pts, cls_scores = refine_pts[i], cls_scores[i]
            refine_pts = decode_points(refine_pts, self.pc_range)

            # filter weak points by distance and score
            centers = refine_pts.mean(dim=1, keepdim=True)
            ctr_dists = torch.norm(refine_pts - centers, dim=-1)
            mask_dist = ctr_dists < ctr_dist_thr
            mask_score = (cls_scores > score_thr).any(dim=-1)
            mask = mask_dist & mask_score
            refine_pts = refine_pts[mask]
            cls_scores = cls_scores[mask]

            pts = torch.cat([refine_pts, cls_scores], dim=-1)
            pts_infos, voxels, num_pts = self.voxel_generator(pts)
            voxels = torch.flip(voxels, [1]).long()
            pts, scores = pts_infos[..., :3], pts_infos[..., 3:]
            scores = scores.sum(dim=1) / num_pts[..., None]

            if self.test_cfg.get('padding', True):
                occ = scores.new_zeros((self.voxel_num[0], self.voxel_num[1], 
                                        self.voxel_num[2], self.num_classes))
                occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = scores
                occ = occ.permute(3, 0, 1, 2).unsqueeze(0)
                # padding
                dilated_occ = F.max_pool3d(occ, 3, stride=1, padding=1)
                eroded_occ = -F.max_pool3d(-dilated_occ, 3, stride=1, padding=1)
                # repalce with original occ prediction
                original_mask = (occ > score_thr).any(dim=1, keepdim=True)
                original_mask = original_mask.expand_as(eroded_occ)
                eroded_occ[original_mask] = occ[original_mask]
                # sparse dense occ
                eroded_occ = eroded_occ.squeeze(0).permute(1, 2, 3, 0)
                voxels = torch.nonzero((eroded_occ > score_thr).any(dim=-1))
                scores = eroded_occ[voxels[:, 0], voxels[:, 1], voxels[:, 2], :]
            occ_pred = torch.ones(self.voxel_num.tolist(),device=eroded_occ.device, dtype=torch.long)*17
            occ_pred[voxels[:,0], voxels[:,1], voxels[:,2]] = \
                              eroded_occ[voxels[:,0],voxels[:,1],voxels[:,2],:].argmax(-1)
            # labels = scores.argmax(dim=-1)
            result_list.append(occ_pred)

        return torch.stack(result_list,0)
    
    def get_sparse_voxels_4d(self, voxel_semantics, mask_camera):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()
        scene_size = [160,120,9.6]
        pc_range = [-80,-60,-3,80,60,6.6]
        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * scene_size[0] + pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * scene_size[1] + pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * scene_size[2] + pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, H, Z)
        coors = torch.stack([xx, yy, zz], dim=-1) # actual space

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.empty_label
            gt_points.append(coors[mask])
            gt_masks.append(mask_camera[i][mask]) # camera mask and not empty
            gt_labels.append(voxel_semantics[i][mask])
        
        return gt_points,gt_masks, gt_labels


    def get_sparse_voxels(self, voxel_semantics, mask_camera):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, W, Z)
        coors = torch.stack([xx, yy, zz], dim=-1) # actual space

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.empty_label
            gt_points.append(coors[mask])
            gt_masks.append(mask_camera[i][mask]) # camera mask and not empty
            gt_labels.append(voxel_semantics[i][mask])
        
        return gt_points, gt_masks, gt_labels
    


    @torch.no_grad()
    def get_sparse_voxels_stack(self,voxel_semantic,temporal_semantics,temporal_fu2curs,mask_camera):
        
        fu_frame = [1,2,3,4,5,6]
        num_fu_frame = len(fu_frame)
        
        temporal_fu2curs = [torch.tensor(np.stack(fu2cur),device=voxel_semantic.device).float() for fu2cur in temporal_fu2curs]
        B, W, H, Z = voxel_semantic.shape
        stack_semantics = voxel_semantic.new_ones(B,400,300,24) * 17
        stack_mask_camera = mask_camera.new_zeros(B,400,300,24)
        # stack_stamp = voxel_semantic.new_zeros(B,400,300,24,num_fu_frame+1)
        device = voxel_semantic.device
        new_voxel_num = torch.tensor([400,300,24],device=voxel_semantic.device)
        pc_range = torch.tensor([-80,-60,-3,80,60,6.6],device=voxel_semantic.device)
        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, W, Z)
        coors = torch.stack([xx, yy, zz], dim=-1)

        for b in range(B):
            for t in [6,5,4,3,2,1,0]:
                cur_mask_camera = mask_camera[b,t].flatten()
                if t>0:
                    cur_semantic,fu2cur = (temporal_semantics[b,t-1],
                                       temporal_fu2curs[b][t-1])     
                else:
                    cur_semantic,fu2cur = voxel_semantic[b],torch.eye(4,device=device)
                cur_points = coors.reshape(-1,3)
                cur_semantic = cur_semantic.flatten()
                cur_points = torch.matmul(cur_points,fu2cur[:3,:3].transpose(0,1)) + fu2cur[:3,3]
                warped_coords = torch.floor((cur_points - pc_range[None,:3])//self.voxel_size[None,:]).long()
                mask = torch.logical_and(warped_coords>=0,warped_coords<new_voxel_num[None,:]).all(-1)

                # stack_stamp[b, warped_coords[mask, 0], warped_coords[mask, 1], warped_coords[mask, 2], t] = 1
                # if mask_moving and t>0:
                #     mask = mask * torch.logical_or(cur_semantic<2,cur_semantic>10)

                stack_semantics[b,warped_coords[mask,0],warped_coords[mask,1],warped_coords[mask,2]] = cur_semantic[mask]
                stack_mask_camera[b,warped_coords[mask,0],warped_coords[mask,1],warped_coords[mask,2]] = cur_mask_camera[mask]
        return stack_semantics,stack_mask_camera
