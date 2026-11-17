"""Direct original PI05 LIBERO deployment: saved training config + safetensors.

Set FLUXVLA_ROOT, PI05_CONFIG, PI05_CHECKPOINT, PI05_DEPLOY_SAMPLE (NPZ),
PI05_WORK_DIR and PI05_BUNDLE_DIR. Shapes match the two-view, horizon-10 case.
BF16 parameters and autocast follow the original LiberoEvalRunner.
"""

deployment = dict(
    name='pi05-libero-bf16',
    work_dir='$PI05_WORK_DIR',
    pipeline=[
        dict(type='PrepareModel'),
        dict(type='Rewrite', transforms=[dict(type='ExportONNX')]),
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
            config='$PI05_CONFIG',
            checkpoint='$PI05_CHECKPOINT',
            model_section='model',
            sample_inputs='$PI05_DEPLOY_SAMPLE',
            device='cuda:0',
            strict=True,
            parameter_dtype='bfloat16',
            autocast_dtype='bfloat16',
            bindings=dict(
                images='images',
                img_masks='img_masks',
                lang_tokens='lang_tokens',
                lang_masks='lang_masks',
                noise='noise',
                # Ordinary PI05 reads only states batch/device/dtype, all equal
                # to noise here. This binding is NOT suitable for PI0.
                # Verified against original six-input full-checkpoint inference.
                states='noise',
            ),
        ),
    ),
    inputs=[
        dict(name='images', dtype='float32', shape=(1, 6, 224, 224)),
        dict(name='img_masks', dtype='bool', shape=(1, 2)),
        dict(name='lang_tokens', dtype='int64', shape=(1, 32)),
        dict(name='lang_masks', dtype='bool', shape=(1, 32)),
        dict(name='noise', dtype='float32', shape=(1, 10, 32)),
    ],
    outputs=[dict(name='actions', dtype='float32', shape=(1, 10, 32))],
    ir_config=dict(
        type='onnx',
        opset=18,
        target_opset=22,
        legacy_bf16_compat=True,
        dynamo=False,
        check=True,
        external_data=True,
        do_constant_folding=False,
    ),
    backend_config=dict(
        type='tensorrt',
        precision='bf16',
        strongly_typed=True,
        allow_tf32=False,
        workspace_size_gib=2,
        log_level='warning',
    ),
    validation=dict(
        atol=0.02,
        rtol=0.02,
        min_cosine_similarity=0.999,
        fail_on_error=True,
    ),
    package=dict(
        output_dir='$PI05_BUNDLE_DIR',
        include_onnx=False,
        assets=[],
        runtime=dict(action_output='actions'),
    ),
)
