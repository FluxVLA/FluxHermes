"""GR00T ALOHA deployment recipe template.

GR00T action sampling needs a model-family recipe that makes stochastic inputs
explicit before export. Point ``factory`` at that recipe after implementing it
with ``PreparedModel``; see docs/model_adapter.md.
"""

deployment = dict(
    name='gr00t-aloha-fp16',
    work_dir='../../work_dirs/gr00t-aloha-fp16',
    pipeline=[
        dict(type='PrepareModel'),
        dict(type='Rewrite', transforms=[dict(type='ExportONNX')]),
        dict(type='LowerPrecision'),
        dict(type='ConvertBackend'),
        dict(type='ValidateBackend'),
        dict(type='PackageBundle'),
    ],
    codebase_config=dict(
        type='python',
        task='custom',
        device='cuda:0',
        source=dict(
            type='python',
            factory='deployment_recipes.gr00t:build_prepared_model',
            python_paths=['$FLUXHERMES_RECIPE_ROOT'],
            device='cuda:0',
            kwargs=dict(
                fluxvla_root='$FLUXVLA_ROOT',
                config='configs/gr00t/gr00t_eagle_3b_aloha_rtc_inference.py',
                checkpoint='$GR00T_CHECKPOINT',
                sample_inputs='$GR00T_DEPLOY_SAMPLE',
            ),
        ),
    ),
    inputs=[
        dict(
            name='image_0',
            dtype='float32',
            shape=(1, 3, 224, 224),
            semantic='normalized primary camera image',
        ),
        dict(
            name='image_1',
            dtype='float32',
            shape=(1, 3, 224, 224),
            semantic='normalized wrist camera image',
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
            name='lang_masks',
            dtype='bool',
            shape=(1, -1),
            min_shape=(1, 8),
            opt_shape=(1, 48),
            max_shape=(1, 64),
            semantic='valid language-token mask',
        ),
        dict(
            name='states',
            dtype='float32',
            shape=(1, 1, 64),
            semantic='normalized and padded robot state',
        ),
        dict(
            name='noise',
            dtype='float32',
            shape=(1, 50, 64),
            semantic='explicit action-head diffusion noise',
        ),
    ],
    outputs=[
        dict(
            name='actions',
            dtype='float32',
            shape=(1, 50, 64),
            semantic='normalized action chunk',
        ),
    ],
    ir_config=dict(
        type='onnx',
        opset=18,
        check=True,
        external_data=True,
    ),
    optimization=dict(
        fp16_conversion=True,
        keep_io_types=True,
    ),
    backend_config=dict(
        type='tensorrt',
        precision='fp16',
        workspace_size_gib=16,
        strongly_typed='auto',
        timing_cache='../../work_dirs/timing/gr00t-aloha.timing',
        log_level='warning',
    ),
    validation=dict(
        atol=2e-2,
        rtol=2e-2,
        min_cosine_similarity=0.995,
        fail_on_error=True,
    ),
    package=dict(
        output_dir='../../artifacts/gr00t-aloha-fp16',
        runtime=dict(action_output='actions'),
    ),
)
