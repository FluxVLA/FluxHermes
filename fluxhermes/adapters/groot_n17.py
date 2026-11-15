"""Thin N1.7 boundary; all neural computation stays in the original FluxVLA."""
from __future__ import annotations

import copy
from pathlib import Path

from ..contracts import TensorSpec
from ..errors import ConfigurationError
from ..plugins import add_python_paths
from .base import PreparedModel

_CONTRACT = (
    ('images', 'float32', (1, 512, 1536)),
    ('lang_tokens', 'int64', (1, -1)),
    ('position_ids', 'int64', (3, 1, -1)),
    ('states', 'float32', (1, 1, 132)),
    ('embodiment_ids', 'int32', (1, )),
    ('noise', 'float32', (1, 40, 132)),
)
_NAMES = tuple(item[0] for item in _CONTRACT)


def _check_specs(deployment):
    specs = deployment.inputs
    if tuple(s.name for s in specs) != _NAMES:
        raise ConfigurationError(f'N1.7 input order must be {_NAMES}.')
    for spec, (name, dtype, shape) in zip(specs, _CONTRACT):
        expected = TensorSpec(name, dtype, shape, spec.min_shape, spec.opt_shape,
                              spec.max_shape)
        if spec.dtype != expected.dtype or spec.shape != expected.shape:
            raise ConfigurationError(f'{name}: expected {dtype} {shape}.')
        for profile in (spec.min_shape, spec.opt_shape, spec.max_shape):
            if profile is not None and any(d < 1 or (s != -1 and s != d)
                                           for s, d in zip(shape, profile)):
                raise ConfigurationError(f'{name}: invalid fixed geometry/profile.')
    tokens, positions = specs[1], specs[2]
    for field in ('min_shape', 'opt_shape', 'max_shape'):
        if getattr(tokens, field)[1] != getattr(positions, field)[2]:
            raise ConfigurationError(
                'lang_tokens and position_ids profiles must share L.')
    if (tokens.min_shape[1] < 138 or tokens.max_shape[1] > 180):
        raise ConfigurationError('N1.7 supported trimmed length range is 138..180.')
    if len(deployment.outputs) != 1 or deployment.outputs[0] != TensorSpec(
            'actions', 'float32', (1, 40, 132)):
        raise ConfigurationError('Expected actions float32[1,40,132] output.')
    return specs


def _check_inputs(inputs, specs):
    import numpy as np
    if set(inputs) != set(_NAMES):
        raise ConfigurationError(f'N1.7 inputs must contain exactly {_NAMES}.')
    for spec in specs:
        value = inputs[spec.name]
        dtype = str(value.dtype).removeprefix('torch.')
        shape = tuple(value.shape)
        if dtype != spec.dtype or len(shape) != len(spec.shape):
            raise ConfigurationError(
                f'{spec.name}: expected {spec.dtype} rank {len(spec.shape)}.')
        for i, (actual, expected) in enumerate(zip(shape, spec.shape)):
            if (expected != -1 and actual != expected) or (
                    expected == -1
                    and not spec.min_shape[i] <= actual <= spec.max_shape[i]):
                raise ConfigurationError(
                    f'{spec.name}: shape {shape} outside contract.')
        finite = (
            value.isfinite().all().item()
            if hasattr(value, 'isfinite') else np.isfinite(value).all())
        if not finite:
            raise ConfigurationError(f'{spec.name}: non-finite values.')
    if inputs['lang_tokens'].shape[1] != inputs['position_ids'].shape[2]:
        raise ConfigurationError('lang_tokens and position_ids must share L.')


def _verify_attention(vlm, mode):
    found = []
    for name, module in vlm.named_modules():
        if 'attention' in type(module).__name__.lower():
            config = getattr(module, 'config', None)
            actual = getattr(config, '_attn_implementation', None)
            if actual != mode:
                raise RuntimeError(f'Attention {name}: expected {mode}, got {actual}.')
            found.append(name)
    if not found:
        raise RuntimeError('No actual VLM attention modules found to verify.')
    return found


def _native_model_config(cfg, root):
    """Restore published native configs without their training-machine assets."""
    from mmengine.config import Config
    model_cfg = copy.deepcopy(cfg.get('inference_model', cfg.model))
    if model_cfg.get('type') != 'GrootN17VLA':
        raise ConfigurationError('Expected a FluxVLA GrootN17VLA training config.')
    flat = 'vlm_backbone' not in model_cfg
    removed_model = {}
    if flat:
        if 'vla_head' in model_cfg:
            raise ConfigurationError(
                'Incomplete N1.7 component config: missing vlm_backbone.')
        for name, expected in dict(
                assembly_runtime='native',
                load_mode='native_safe',
                processor_runtime='native').items():
            if model_cfg.get(name) != expected:
                raise ConfigurationError(f'Unsupported legacy model.{name}.')
            removed_model[name] = model_cfg.pop(name)
        locator = model_cfg.pop('backbone_model_path', None)
        removed_model['backbone_model_path'] = locator
    else:
        locator = model_cfg['vlm_backbone'].get('model_name')
    architecture_source = None
    if flat or 'model_config' not in model_cfg['vlm_backbone']:
        if not isinstance(locator, str) or Path(locator).name != 'Cosmos-Reason2-2B':
            raise ConfigurationError(
                'Missing backbone model_config; expected the saved '
                'Cosmos-Reason2-2B config.')
        # Reuse FluxVLA's architecture declaration, not pretrained model files.
        architecture_source = root / (
            'configs/gr00tn17/gr00tn17_qwen3vl_2b_libero_10_full_finetune.py')
        canonical = Config.fromfile(str(architecture_source))
        if flat:
            model_cfg['vlm_backbone'] = copy.deepcopy(canonical.model.vlm_backbone)
            model_cfg['vla_head'] = copy.deepcopy(canonical.model.vla_head)
            model_cfg['vlm_backbone'].update(
                use_flash_attention=model_cfg.get('use_flash_attention'),
                qwen3_runtime=model_cfg.get('qwen3_runtime'))
        else:
            model_cfg['vlm_backbone']['model_config'] = copy.deepcopy(
                canonical.model.vlm_backbone.model_config)
    backbone = model_cfg['vlm_backbone']
    if (backbone.get('type') != 'GrootN17Qwen3Backbone'
            or type(backbone.get('use_flash_attention')) is not bool):
        raise ConfigurationError(
            'Expected native Qwen3 backbone and boolean attention flag.')
    legacy_values = dict(
        transformers_loading_kwargs=dict(local_files_only=True, trust_remote_code=True),
        trainable_params_fp32=False,
        tune_llm=False,
        tune_top_llm_layers=0,
        tune_visual=False)
    removed = {}
    for name in ('model_name', *legacy_values):
        if name not in backbone:
            continue
        value = backbone[name]
        if name == 'model_name':
            valid = isinstance(value, str) and Path(value).name == 'Cosmos-Reason2-2B'
        else:
            expected = legacy_values[name]
            if isinstance(expected, dict):
                valid = isinstance(value, dict) and value == expected
            elif isinstance(expected, bool):
                # These flags controlled training, not the BF16 evaluation dtype.
                valid = type(value) is bool
            else:
                valid = type(value) is type(expected) and value == expected
        if not valid:
            raise ConfigurationError(
                f'Unsupported saved vlm_backbone.{name}: {value!r}')
        removed[name] = backbone.pop(name)
    model_cfg.update(
        model_path=None,
        processor_path=None,
        load_metadata=False,
        load_pretrained_weights=False)
    return model_cfg, dict(
        architecture_source=str(architecture_source) if architecture_source else None,
        removed_legacy_model_options=removed_model,
        removed_legacy_backbone_options=removed,
        disabled_pretrained_loading=True)


def _native_checkpoint_keys(state_dict):
    """Adapt the old backbone attribute name in memory; retain original tensors."""
    result, renamed = {}, 0
    for key, value in state_dict.items():
        target = key
        for prefix in ('vlm_backbone.', 'n17_backbone.'):
            if key.startswith(prefix + 'model.'):
                target = prefix + 'vlm.' + key[len(prefix + 'model.'):]
                break
        if target in result:
            raise ConfigurationError(f'Conflicting checkpoint aliases for {target}.')
        result[target] = value
        renamed += target != key
    return result, renamed


def _set_attention(vlm, mode):
    vlm.set_attn_implementation(mode)
    return _verify_attention(vlm, mode)


def _make_wrapper(model, torch):

    class Boundary(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.model = model
            self.register_buffer(
                'image_grid_thw',
                torch.tensor([[[1, 16, 16], [1, 16, 16]]], dtype=torch.int64))

        def forward(self, images, lang_tokens, position_ids, states, embodiment_ids,
                    noise):
            vlm, head = self.model.vlm_backbone.vlm, self.model.vla_head
            had_local = '_sample_initial_actions' in head.__dict__
            previous = head.__dict__.get('_sample_initial_actions')

            def positions_hook(module, args, kwargs):
                if kwargs.get('position_ids') is not None:
                    raise RuntimeError('Original VLM already supplies position_ids.')
                return args, dict(kwargs, position_ids=position_ids)

            handle = vlm.register_forward_pre_hook(positions_hook, with_kwargs=True)
            try:
                head._sample_initial_actions = lambda *args, **kwargs: noise.clone()
                with torch.autocast(images.device.type, dtype=torch.bfloat16):
                    return self.model.predict_action(
                        images=images,
                        lang_tokens=lang_tokens,
                        lang_masks=torch.ones_like(lang_tokens),
                        image_grid_thw=self.image_grid_thw,
                        states=states.to(torch.bfloat16),
                        embodiment_ids=embodiment_ids)
            finally:
                handle.remove()
                if had_local:
                    head._sample_initial_actions = previous
                else:
                    del head._sample_initial_actions

    return Boundary().eval()


def _reference_runner(wrapper, specs, torch, attention_mode='sdpa'):

    def run(inputs):
        _check_inputs(inputs, specs)
        vlm = wrapper.model.vlm_backbone.vlm
        try:
            if attention_mode != 'sdpa':
                _set_attention(vlm, attention_mode)
            else:
                _verify_attention(vlm, 'sdpa')
            with torch.inference_mode():
                return wrapper(*(inputs[name] for name in _NAMES))
        finally:
            if attention_mode != 'sdpa':
                _set_attention(vlm, 'sdpa')

    return run


def prepare_groot_n17(*, deployment, device, fluxvla_root, config, checkpoint,
                      sample_inputs):
    """Load the native FluxVLAEngine training checkpoint, not official GR00T shards.

    NPZ is prepared externally using the original backbone's
    _trim_common_left_padding and vlm.model.get_rope_index. No RoPE/trim
    arithmetic is reimplemented here. This shared model is not thread-safe.
    """
    import numpy as np
    import torch
    specs = _check_specs(deployment)
    if torch.device(device).type != 'cuda':
        raise ConfigurationError('N1.7 deployment factory requires CUDA.')
    root = Path(fluxvla_root).expanduser().resolve()
    if not (root / 'fluxvla').is_dir():
        raise ConfigurationError(f'Invalid FluxVLA root: {root}')
    config_path = Path(config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    sample_path = Path(sample_inputs).expanduser()
    if not sample_path.is_absolute() and deployment.config_path:
        sample_path = deployment.config_path.parent / sample_path
    for path in (config_path, checkpoint_path, sample_path):
        if not path.exists():
            raise ConfigurationError(f'N1.7 source path not found: {path}')
    if not checkpoint_path.is_file() or checkpoint_path.suffix != '.safetensors':
        raise ConfigurationError('Expected a FluxVLAEngine training .safetensors file, '
                                 'not an official directory.')
    with np.load(sample_path, allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}
    _check_inputs(inputs, specs)
    add_python_paths([str(root)])
    from fluxvla.engines import build_vla_from_cfg
    from mmengine.config import Config
    cfg = Config.fromfile(str(config_path))
    model_cfg, config_normalization = _native_model_config(cfg, root)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_vla_from_cfg(model_cfg)
    unknown = set(
        model.extra_cfg) - {'state_dropout_prob', 'reproject_vision', 'load_bf16'}
    if unknown:
        raise ConfigurationError(f'Unknown N1.7 model options: {sorted(unknown)}')
    head_options = set(vars(model._native_n17_config())) | {
        'type', 'tune_diffusion_model', 'tune_projector', 'tune_vlln'
    }
    unknown = set(model_cfg['vla_head']) - head_options
    if unknown:
        raise ConfigurationError(f'Unknown N1.7 vla_head options: {sorted(unknown)}')
    from safetensors.torch import load_file
    state_dict = load_file(str(checkpoint_path), device='cpu')
    checkpoint_dtypes = sorted({str(value.dtype) for value in state_dict.values()})
    state_dict, renamed_checkpoint_keys = _native_checkpoint_keys(state_dict)
    # Native loading owns outer n17_* remapping and lazy/meta materialization.
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    # Match FluxVLA's BF16 evaluation; the input checkpoint stays untouched.
    model.eval().to(device, dtype=torch.bfloat16)
    vlm = model.vlm_backbone.vlm
    reference_attention = ('flash_attention_2' if
                           model_cfg['vlm_backbone']['use_flash_attention'] else 'sdpa')
    attention_modules = _verify_attention(vlm, reference_attention)
    if reference_attention != 'sdpa':
        _set_attention(vlm, 'sdpa')
    wrapper = _make_wrapper(model, torch).to(device)
    inputs = {name: torch.from_numpy(inputs[name]).to(device) for name in _NAMES}
    return PreparedModel(
        module=wrapper,
        inputs=inputs,
        output_names=('actions', ),
        dynamic_axes={
            'lang_tokens': {
                1: 'sequence_length'
            },
            'position_ids': {
                2: 'sequence_length'
            }
        },
        reference_runner=_reference_runner(wrapper, specs, torch, reference_attention),
        metadata=dict(
            adapter='groot_n17',
            fluxvla_root=str(root),
            config=str(config_path),
            checkpoint=str(checkpoint_path),
            sample_inputs=str(sample_path),
            implementation=(
                'Config.fromfile -> build_vla_from_cfg -> native strict load_state_dict'
            ),
            config_normalization=config_normalization,
            checkpoint_dtypes=checkpoint_dtypes,
            renamed_checkpoint_keys=renamed_checkpoint_keys,
            runtime_parameter_dtype='bfloat16',
            host_boundary=('Original _trim_common_left_padding + '
                           'get_rope_index in external samples'),
            static_image_grid_thw=[[1, 16, 16], [1, 16, 16]],
            image_geometry=[1, 512, 1536],
            batch_size=1,
            dynamic_length=[specs[1].min_shape[1], specs[1].max_shape[1]],
            reference_attention=reference_attention,
            export_attention='sdpa',
            attention_modules=attention_modules,
            postprocessing='Original predict_action only; no new mathematics'))
