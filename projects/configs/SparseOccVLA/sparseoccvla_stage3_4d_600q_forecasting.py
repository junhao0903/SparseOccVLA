_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'
occ_root = 'data/nuscenes/gts/'
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

voxel_size = [0.4, 0.4, 0.4]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
occ_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]

embed_dims = 1024
num_layers = 6
num_query = 600
num_frames = 8
num_levels = 4
num_points = 4
num_groups=4
num_refines = [2, 4, 16, 32, 56, 112]

num_gpus = 8
batch_size = 2
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 12
llm_path = './ckpts/InternVL2-2B'
# llm_path = None

img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=True)
img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=num_levels)
chat_cfg = dict(
        num_beams=1,
        max_new_tokens=1000,
        min_new_tokens=1,
        do_sample=False,
        temperature=0.)

collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command', 'can_bus']
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
model = dict(
    type='SparseOccVLA',
    save_path='v./results_planning_only/',  #save path for vlm models.
    frozen=False,
    use_lora=False,
    use_grid_mask=False,
    stop_prev_grad=0,
    forecasting=True,
    num_global_query=12,
    out_dims=2048,
    use_gen_token=False,
    chat_config=chat_cfg,
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    tokenizer=llm_path,
    lm_head=llm_path, # set to None if don't use llm head
    img_backbone=img_backbone,
    img_neck = img_neck,
    plan_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5,
            ),
    plan_loss_reg=dict(type='L1Loss', loss_weight=1.0),
    pts_bbox_head=dict(
        type='OPUSHead',
        num_classes=len(occ_names),
        in_channels=embed_dims,
        num_query=num_query,
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        signal_4d='occ_4d',
        transformer=dict(
            type='OPUSTransformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_global_query=12,
            num_layers=num_layers,
            num_groups=num_groups,
            num_levels=num_levels,
            num_out_points=32,
            num_classes=len(occ_names),
            num_refines=num_refines,
            scales=[0.5],
            pc_range=point_cloud_range),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_pts=dict(type='SmoothL1Loss', beta=0.2, loss_weight=0.5)),
        
    train_cfg=dict(
        pts=dict(
            cls_weights=[
                10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1],
            )
        ),
        test_cfg=dict(
        pts=dict(
            score_thr=0.35,
            padding=True)
    )
            )

dataset_type = 'CustomNuScenesDataset4D'
data_root = './data/nuscenes/'

file_client_args = dict(backend='disk')

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
    #     with_label=True, with_bbox_depth=True),
    dict(type='Load4DOccFromFile', occ_root=occ_root), 
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    # dict(type='ResizeMultiview3D', img_scale=(256, 704), keep_ratio=False, multiscale_mode='value'),
    dict(type='LoadAnnoatationVQA_InternVL', 
         base_vqa_path='./data/nuscenes/proxy/train/', 
         base_desc_path='./data/nuscenes/desc/train/',
         base_conv_path='./data/nuscenes/conv/train/',
         base_key_path='./data/nuscenes/keywords/train/',
         llm_path=llm_path,
         max_length=3072,
         img_length=613, 
         ignore_type=['v1','v2','v3','planning'],
         lane_objs_info="./data/nuscenes/lane_obj_train.pkl"),
    # dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='PadMultiViewImage', size_divisor=32),
    # dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='OPUSFormatBundle3D', class_names=class_names),

    # dict(type='Collect3D', keys=['img', 'voxel_semantics', 'mask_camera'], meta_keys=(
    #     'filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar','sample_idx'))

    dict(type='Collect3D', keys=['input_ids', 'vlm_labels', 'img', 'voxel_semantics','mask_camera','can_bus','gt_planning','command','gt_fu_mask','temporal_ego_states'],
             meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'img_norm_cfg', 'sample_idx',
              'lidar2img','img_timestamp','ego2lidar','temporal2ego','num_fu_frame_real','canbus_fu'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),

    dict(type='PadMultiViewImage', size_divisor=32),

    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='OPUSFormatBundle3D',
                # collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
                dict(type='Collect3D', keys=[ 'img','can_bus','command','temporal_ego_states'], 
                meta_keys=('sample_idx', 'filename', 'ori_shape', 'img_shape','pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 
            'box_type_3d', 'img_norm_cfg', 'scene_token','lidar2img', 'img_timestamp', 'ego2lidar','canbus_fu'))
            
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_train.pkl',
        seq_split_num=1, # streaming video training
        seq_mode=True, # streaming video training
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR',
        historical_can_bus=True),
    val=dict(
        type=dataset_type, 
        eval_mode=['lane', 'det'],
        pipeline=test_pipeline, 
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_val.pkl',
        classes=class_names, 
        modality=input_modality),
    test=dict(
        type=dataset_type, 
        eval_mode=['lane', 'det'],
        pipeline=test_pipeline, 
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_val.pkl', 
        classes=class_names, 
        modality=input_modality,
        historical_can_bus=True),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        seq_split_num=2,
        warmup_split_num=10, # lane det and vlm need short term temporal fusion in the early stage of training
        num_iters_to_seq=num_iters_per_epoch,
    ),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )


optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
        'lm_head':dict(lr_mult=0.05,decay_mult=0.01)
    }),
    weight_decay=0.01
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
bf16 = dict(enabled=True)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)

find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=1)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
# runner = dict(type='EpochBasedRunner',max_epochs=num_epochs)

custom_hooks = [dict(type='IterAccumulateHook')]

# load_from='ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
# load_from='work_dirs/mask_eva_lane_det_vlm_sparse_stage2_4d_450q_forecasting_res_above_planning_distill_thre/latest.pth'
load_from='ckpts/stage1_4d_600q.pth'
# revise_keys = [('backbone', 'img_backbone')]
resume_from=None
log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)