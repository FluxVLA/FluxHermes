"""PI0.5 ALOHA FP16 TensorRT deployment template.

Create the representative NPZ before executing. Required keys are exactly the
input names below. The language length may vary within the declared profile.
"""

deployment = dict(
    name='pi05-aloha-fp16',
    work_dir='../../work_dirs/pi05-aloha-fp16',
    pipeline=[
        dict(type='PrepareModel'),
        dict(type='Rewrite', transforms=[dict(type='ExportONNX')]),
        dict(type='LowerPrecision'),
        dict(type='ConvertBackend'),
        dict(type='ValidateBackend'),
        dict(type='PackageBundle'),
    ],
    codebase_config=dict(
        type='fluxvla',
        task='predict_action',
        device='cuda:0',
        source=dict(
            type='fluxvla',
            fluxvla_root='$FLUXVLA_ROOT',
            config='configs/pi05/pi05_paligemma_aloha_full_finetune.py',
            checkpoint='$PI05_CHECKPOINT',
            model_section='model',
            sample_inputs='$PI05_DEPLOY_SAMPLE',
            device='cuda:0',
            strict=True,
            bindings=dict(
                images='images',
                lang_tokens='lang_tokens',
                states='states',
                lang_masks='lang_masks',
                noise='noise',
            ),
        ),
    ),
    inputs=[
        dict(
            name='images',
            dtype='float32',
            shape=(1, 6, 224, 224),
            semantic='two normalized RGB camera views in NCHW layout',
        ),
        dict(
            name='lang_tokens',
            dtype='int64',
            shape=(1, -1),
            min_shape=(1, 8),
            opt_shape=(1, 48),
            max_shape=(1, 64),
            semantic='tokenized instruction',
        ),
        dict(
            name='states',
            dtype='float32',
            shape=(1, 1, 32),
            semantic='normalized and padded robot state',
        ),
        dict(
            name='lang_masks',
            dtype='bool',
            shape=(1, -1),
            min_shape=(1, 8),
            opt_shape=(1, 48),
            max_shape=(1, 64),
            semantic='valid language-token mask',
        ),
        dict(
            name='noise',
            dtype='float32',
            shape=(1, 50, 32),
            semantic='fixed diffusion noise for deterministic parity checks',
        ),
    ],
    outputs=[
        dict(
            name='actions',
            dtype='float32',
            shape=(1, 50, 32),
            semantic='normalized action chunk',
        ),
    ],
    ir_config=dict(
        type='onnx',
        opset=18,
        check=True,
        external_data=True,
        do_constant_folding=True,
    ),
    optimization=dict(
        fp16_conversion=True,
        keep_io_types=True,
    ),
    backend_config=dict(
        type='tensorrt',
        precision='fp16',
        workspace_size_gib=16,
        strict_types=False,
        strongly_typed='auto',
        timing_cache='../../work_dirs/timing/pi05-aloha.timing',
        log_level='warning',
    ),
    validation=dict(
        atol=2e-2,
        rtol=2e-2,
        min_cosine_similarity=0.995,
        fail_on_error=True,
    ),
    package=dict(
        output_dir='../../artifacts/pi05-aloha-fp16',
        include_onnx=False,
        assets=[],
        runtime=dict(action_output='actions', ),
    ),
)
