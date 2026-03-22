_base_ = [
    '../_base_/default_runtime.py'
]

object_size = 256
task='semantic'

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
        dropout=0.0),
    diffusion_cfg=dict(
        betas=dict(
            type='linear',
            start=0.8,
            stop=0,
            num_timesteps=6),
        diff_iter=False),
    test_cfg=dict(
        model_size=object_size,
        fine_prob_thr=0.9,
        iou_thr=0.3,
        batch_max=32,
        centroid_correction=dict(
            enabled=True,
            max_offset=6,
            use_multiprocessing=False,
            num_workers=4
        )))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCoarseMasks', test_mode=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'coarse_masks'])
]

data = dict(
    train=dict(),
    val=dict(),
    test=dict(
        pipeline=test_pipeline,
        type='BigDataset',
        data_root='/home/ubt1234/syc/1_lunwen3/3_R1/SegRefiner/data/unet/renamed_images',
        test_mode=True),
    test_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=1,
        shuffle=False))

total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)
