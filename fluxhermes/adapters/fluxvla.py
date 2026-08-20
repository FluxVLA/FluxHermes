from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from ..errors import ConfigurationError, DependencyError
from ..plugins import add_python_paths, load_symbol
from .base import ModelAdapter, PreparedModel


class FluxVLAAdapter(ModelAdapter):
    """Load a FluxVLA model and wrap ``predict_action`` for ONNX export.

    The adapter deliberately keeps model-family-specific preprocessing out of
    FluxHermes. Representative graph inputs come from an NPZ file or a Python
    sample factory, and ``bindings`` describes how flat engine inputs are
    reconstructed into ``predict_action`` keyword arguments.
    """

    def __init__(self, source: Dict[str, Any], deployment: Any) -> None:
        self.source = source
        self.deployment = deployment

    def _deployment_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        config_path = getattr(self.deployment, 'config_path', None)
        if not path.is_absolute() and config_path is not None:
            path = config_path.parent / path
        return path.resolve()

    def _import_stack(self):
        root = Path(
            self.source.get('fluxvla_root', os.getenv('FLUXVLA_ROOT', ''))
        ).expanduser()
        if not str(root):
            raise ConfigurationError(
                'source.fluxvla_root or FLUXVLA_ROOT must be configured.')
        root = root.resolve()
        if not (root / 'fluxvla').is_dir():
            raise ConfigurationError(f'Invalid FluxVLA root: {root}')
        add_python_paths([str(root), *self.source.get('python_paths', ())])
        try:
            import torch
            from fluxvla.engines import build_vla_from_cfg
            from mmengine import Config
        except Exception as error:
            raise DependencyError(
                'FluxVLA export requires the FluxVLA Python environment: '
                f'{error}') from error
        return root, torch, Config, build_vla_from_cfg

    def _load_model(self, device: str):
        root, torch, Config, build_vla_from_cfg = self._import_stack()
        config_path = Path(self.source.get('config', '')).expanduser()
        if not config_path.is_absolute():
            config_path = root / config_path
        if not config_path.is_file():
            raise ConfigurationError(f'FluxVLA config not found: {config_path}')
        cfg = Config.fromfile(str(config_path))
        section = str(self.source.get('model_section', 'model'))
        if not hasattr(cfg, section):
            raise ConfigurationError(
                f'FluxVLA config has no model section named {section!r}.')
        model = build_vla_from_cfg(getattr(cfg, section))
        checkpoint = Path(self.source.get('checkpoint', '')).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        if not checkpoint.exists():
            raise ConfigurationError(f'Checkpoint not found: {checkpoint}')
        state_dict = self._load_state_dict(checkpoint, torch)
        missing, unexpected = model.load_state_dict(
            state_dict, strict=bool(self.source.get('strict', True)))
        if not self.source.get('strict', True) and (missing or unexpected):
            print(
                '[FluxHermes] non-strict checkpoint load: '
                f'missing={len(missing)}, unexpected={len(unexpected)}',
                file=sys.stderr,
            )
        model.eval().to(device)
        return model, cfg, checkpoint, torch

    @staticmethod
    def _load_state_dict(checkpoint: Path, torch: Any) -> Mapping[str, Any]:
        if checkpoint.is_dir() or checkpoint.suffix == '.safetensors':
            try:
                from safetensors.torch import load_file
            except Exception as error:
                raise DependencyError(
                    'Loading safetensors checkpoints requires safetensors.') from error
            files = [checkpoint] if checkpoint.is_file() else sorted(
                checkpoint.glob('*.safetensors'))
            if not files:
                raise ConfigurationError(
                    f'No safetensors files found in {checkpoint}')
            state_dict: Dict[str, Any] = {}
            for path in files:
                state_dict.update(load_file(str(path), device='cpu'))
            return state_dict
        loaded = torch.load(str(checkpoint), map_location='cpu')
        if isinstance(loaded, dict) and 'model' in loaded:
            return loaded['model']
        return loaded

    def _load_inputs(self, device: str, torch: Any) -> Mapping[str, Any]:
        sample_factory = self.source.get('sample_factory')
        if sample_factory:
            factory = load_symbol(sample_factory)
            samples = factory(
                deployment=self.deployment,
                device=device,
                **dict(self.source.get('sample_kwargs', {})),
            )
            return OrderedDict((spec.name, samples[spec.name])
                               for spec in self.deployment.inputs)
        sample_path = self._deployment_path(
            str(self.source.get('sample_inputs', '')))
        if not sample_path.is_file():
            raise ConfigurationError(
                'FluxVLA export needs source.sample_inputs (.npz) or '
                'source.sample_factory.')
        try:
            import numpy as np
        except Exception as error:
            raise DependencyError('Loading NPZ samples requires numpy.') from error
        with np.load(sample_path, allow_pickle=False) as archive:
            missing = [
                spec.name for spec in self.deployment.inputs
                if spec.name not in archive
            ]
            if missing:
                raise ConfigurationError(
                    f'Sample input file is missing: {", ".join(missing)}')
            return OrderedDict((spec.name,
                                torch.from_numpy(archive[spec.name]).to(device))
                               for spec in self.deployment.inputs)

    def _make_wrapper(self, model: Any, torch: Any) -> Any:
        input_names = tuple(spec.name for spec in self.deployment.inputs)
        output_names = tuple(spec.name for spec in self.deployment.outputs)
        bindings = dict(self.source.get('bindings', {}))
        if not bindings:
            bindings = {name: name for name in input_names}

        class PredictActionWrapper(torch.nn.Module):

            def __init__(self, wrapped_model: Any) -> None:
                super().__init__()
                self.wrapped_model = wrapped_model

            def forward(self, *values):
                flat = dict(zip(input_names, values))
                kwargs: Dict[str, Any] = {}
                for argument, binding in bindings.items():
                    if isinstance(binding, str):
                        kwargs[argument] = flat[binding]
                    elif isinstance(binding, (list, tuple)):
                        kwargs[argument] = [flat[name] for name in binding]
                    elif isinstance(binding, dict):
                        names = binding.get('inputs', ())
                        container = binding.get('container', 'list')
                        items = [flat[name] for name in names]
                        kwargs[argument] = (
                            tuple(items) if container == 'tuple' else items)
                    else:
                        raise TypeError(
                            f'Unsupported binding for {argument}: {binding!r}')
                result = self.wrapped_model.predict_action(**kwargs)
                if isinstance(result, dict):
                    values_out = tuple(result[name] for name in output_names)
                    return values_out[0] if len(values_out) == 1 else values_out
                return result

        return PredictActionWrapper(model).eval()

    def _calibration_source(self, device: str):
        reference = self.source.get('calibration_factory')
        directory = self.source.get('calibration_inputs')
        if not reference and not directory:
            return None

        if directory:
            root = self._deployment_path(str(directory))

            def load_npz_directory() -> Iterable[Mapping[str, Any]]:
                try:
                    import numpy as np
                except Exception as error:
                    raise DependencyError(
                        'NPZ calibration loading requires numpy.') from error
                files = sorted(root.glob('*.npz'))
                if not files:
                    raise ConfigurationError(
                        f'No calibration NPZ files found in {root}')
                names = tuple(spec.name for spec in self.deployment.inputs)
                for path in files:
                    with np.load(path, allow_pickle=False) as archive:
                        missing = [name for name in names if name not in archive]
                        if missing:
                            raise ConfigurationError(
                                f'{path} is missing calibration inputs: {missing}')
                        yield {name: archive[name] for name in names}

            return load_npz_directory

        def load() -> Iterable[Mapping[str, Any]]:
            factory = load_symbol(reference)
            return factory(
                deployment=self.deployment,
                device=device,
                **dict(self.source.get('calibration_kwargs', {})),
            )

        return load

    def prepare(self, device: str) -> PreparedModel:
        model, _, checkpoint, torch = self._load_model(device)
        inputs = self._load_inputs(device, torch)
        dynamic_axes: Dict[str, Dict[int, str]] = {}
        for spec in self.deployment.inputs + self.deployment.outputs:
            axes = {
                index: f'{spec.name}_dim_{index}'
                for index, dim in enumerate(spec.shape) if dim == -1
            }
            if axes:
                dynamic_axes[spec.name] = axes
        return PreparedModel(
            module=self._make_wrapper(model, torch),
            inputs=inputs,
            output_names=tuple(spec.name for spec in self.deployment.outputs),
            dynamic_axes=dynamic_axes,
            calibration_source=self._calibration_source(device),
            metadata={
                'adapter': 'fluxvla',
                'config': str(self.source.get('config')),
                'checkpoint': str(checkpoint),
                'model_section': str(self.source.get('model_section', 'model')),
            },
        )
