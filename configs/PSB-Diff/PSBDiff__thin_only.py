# -*- coding: utf-8 -*-

task = 'semantic'
object_size = 128

model = dict(
    type='SegRefinerSemantic',
    task=task,
    step=6,
    denoise_model=dict(
        type='DenoiseUNet',
        in_channels=4,
        out_channels=1,
        model_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_strides=(16, 32),
        learn_time_embd=True,
        channel_mult=(1, 1, 2, 2, 4, 4),
        dropout=0.0
    ),
    diffusion_cfg=dict(
        betas=dict(type='linear', start=0.8, stop=0, num_timesteps=6),
        diff_iter=False
    ),
    loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    loss_texture=dict(type='TextureL1Loss', loss_weight=1.0),

    loss_centroid=dict(type='CentroidPreservationLoss', loss_weight=0.1),
    test_cfg=dict()
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=True),
    dict(type='FilterAnnotations', by_box=False, by_mask=True, keep_empty=False, min_gt_mask_area=0),
    dict(type='LoadPatchData', object_size=object_size, patch_size=object_size),
    dict(type='Resize', img_scale=(object_size, object_size), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=[
        'object_img', 'object_gt_masks', 'object_coarse_masks',
        'patch_img', 'patch_gt_masks', 'patch_coarse_masks'
    ])
]

dataset_type = 'HRCollectionDataset'
data_root = '/home/ubt1234/syc/1_lunwen3/3_R1/SegRefiner/data'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    persistent_workers=False,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_root=data_root,
        collection_datasets=['thin'],
        collection_json=data_root + '/collection_hr.json'
    ),
    val=dict(),
    test=dict()
)

optimizer = dict(type='AdamW', lr=4e-5, weight_decay=0, eps=1e-8, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

max_iters = 40000
runner = dict(type='IterBasedRunner', max_iters=max_iters)

lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[20000, 35000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,
    warmup_iters=10
)

checkpoint_config = dict(by_epoch=False, interval=5000, save_last=True, max_keep_ckpts=20)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ]
)
workflow = [('train', 5000)]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None

opencv_num_threads = 0
mp_start_method = 'fork'

auto_scale_lr = dict(enable=False, base_batch_size=16)