"""Compile an explicitly configured PI0.5 SDK policy on the target GPU.

Install fluxvla-export independently, then set PI05_DEPLOYMENT_JSON,
PI05_CHECKPOINT and PI05_EXPORT_INPUTS to matching JSON/safetensors artifacts.
Relative paths are relative to this Hermes config, not a training checkout.
The SDK JSON owns architecture, preprocessing IDs, dimensions and precision;
this example copies only its public policy tensor boundary into Hermes.
"""

import os
from pathlib import Path

from fluxhermes.adapters.fluxvla_export import tensor_specs_from_sdk
from fluxvla_export.config import load_config as load_sdk_config


def _required_path(name):
    value = os.environ.get(name)
    if not value or not value.strip():
        raise ValueError(f'{name} must name an explicit PI0.5 artifact.')
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).parent / path
    return path.resolve()


_sdk_json = _required_path('PI05_DEPLOYMENT_JSON')
_checkpoint = _required_path('PI05_CHECKPOINT')
_inputs = _required_path('PI05_EXPORT_INPUTS')
_sdk = load_sdk_config(_sdk_json, recipe='pi05-v1')
_policy_specs = tensor_specs_from_sdk(_sdk.input_specs)
_noise = next(spec for spec in _policy_specs if spec.name == 'noise')

deployment = dict(
    name='pi05-export-fp32',
    work_dir='../../work_dirs/pi05-export-fp32',
    codebase_config=dict(
        type='fluxvla',
        task='export_package',
        device='cuda:0',
        source=dict(
            type='fluxvla_export',
            recipe='pi05-v1',
            deployment_config=str(_sdk_json),
            checkpoint=str(_checkpoint),
            inputs=str(_inputs))),
    inputs=[spec.to_dict() for spec in _policy_specs],
    outputs=[
        dict(
            name='actions',
            dtype='float32',
            shape=_noise.shape,
            semantic=_sdk.preprocessing.action_semantics)
    ],
    ir_config=dict(
        # The SDK selects legacy/opset18 for FP32 or Dynamo/opset22 for BF16.
        type='onnx',
        check=True,
        external_data=True,
        do_constant_folding=False),
    optimization=dict(simplify=False, fp16_conversion=False),
    quantization=dict(mode='none'),
    backend_config=dict(
        type='tensorrt',
        precision='fp32',
        strongly_typed=True,
        tf32=False,
        workspace_size_gib=4,
        log_level='warning'),
    validation=dict(
        enabled=True,
        atol=1e-5,
        rtol=1e-4,
        min_cosine_similarity=0.9999,
        fail_on_error=True),
    package=dict(
        output_dir='../../artifacts/pi05-export-fp32',
        include_onnx=False,
        assets=[],
        runtime=dict(action_output='actions')),
)
