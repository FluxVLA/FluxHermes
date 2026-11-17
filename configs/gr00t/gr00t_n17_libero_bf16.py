"""FluxVLAEngine N1.7 LIBERO-10 training checkpoint; BF16 native evaluation.

GROOT_N17_CONFIG is the saved training config.json; GROOT_N17_CHECKPOINT is
its original .safetensors file, not an NVIDIA checkpoint directory. Source:
https://huggingface.co/limxdynamics/FluxVLAEngine/tree/main/gr00t_n17_native_libero_10_full_finetune_bs64
The old saved backbone config reuses FluxVLA's native N1.7 architecture config.
Required env: FLUXVLA_ROOT, GROOT_N17_CONFIG, GROOT_N17_CHECKPOINT,
GROOT_N17_DEPLOY_SAMPLE, GROOT_N17_WORK_DIR, GROOT_N17_BUNDLE_DIR.
Deployment token-length range: 138..180.
Set validation.fail_on_error=False explicitly for diagnostic failed bundles;
the default validation and unverified-runtime gates remain enabled.
"""

import os


_isolate_export = os.environ.get(
    'GROOT_N17_ISOLATE_EXPORT', 'true').strip().lower()
if _isolate_export not in {'true', 'false'}:
    raise ValueError('GROOT_N17_ISOLATE_EXPORT must be true or false.')

deployment = dict(
    name='gr00t-n17-libero-bf16', work_dir='$GROOT_N17_WORK_DIR',
    codebase_config=dict(
        type='python', task='custom', device='cuda:0',
        isolate_export=_isolate_export == 'true',
        source=dict(
            type='python',
            factory='fluxhermes.adapters.groot_n17:prepare_groot_n17',
            kwargs=dict(
                fluxvla_root='$FLUXVLA_ROOT', config='$GROOT_N17_CONFIG',
                checkpoint='$GROOT_N17_CHECKPOINT',
                sample_inputs='$GROOT_N17_DEPLOY_SAMPLE'))),
    inputs=[
        dict(name='images', dtype='float32', shape=(1, 512, 1536)),
        dict(name='lang_tokens', dtype='int64', shape=(1, -1),
             min_shape=(1, 138), opt_shape=(1, 148), max_shape=(1, 180)),
        dict(name='position_ids', dtype='int64', shape=(3, 1, -1),
             min_shape=(3, 1, 138), opt_shape=(3, 1, 148), max_shape=(3, 1, 180)),
        dict(name='states', dtype='float32', shape=(1, 1, 132)),
        dict(name='embodiment_ids', dtype='int32', shape=(1,)),
        dict(name='noise', dtype='float32', shape=(1, 40, 132)),
    ],
    outputs=[dict(name='actions', dtype='float32', shape=(1, 40, 132))],
    ir_config=dict(
        type='onnx', opset=18, target_opset=22, legacy_bf16_compat=True,
        normalize_full_slices=True, dynamo=False, check=True,
        external_data=True, do_constant_folding=False),
    optimization=dict(simplify=False, fp16_conversion=False),
    quantization=dict(mode='none'),
    backend_config=dict(
        type='tensorrt', precision='bf16', strongly_typed=True,
        allow_tf32=False, workspace_size_gib=2, log_level='warning',
        file_descriptor_limit=4096,
        builder_optimization_level=0, tactic_dram_size_gib=8),
    validation=dict(
        enabled=True, atol=0.1, rtol=0.02, min_cosine_similarity=0.999,
        fail_on_error=True),
    package=dict(
        output_dir='$GROOT_N17_BUNDLE_DIR', include_onnx=False, assets=[],
        runtime=dict(action_output='actions')),
)
