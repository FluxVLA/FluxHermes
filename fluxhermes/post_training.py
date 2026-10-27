from __future__ import annotations

import copy
import importlib.util
import time
from pathlib import Path
from typing import Any, Dict, Mapping

from .backend.tensorrt import TensorRTWrapper
from .builders import build_tensorrt_engine
from .errors import DependencyError, ValidationError
from .exporters import export_onnx
from .model_families import detect_model_family
from .plugins import add_python_paths
from .utils import sha256_file, write_json
from .validation import compare_outputs


def _fluxvla_root(config_path: Path) -> Path:
    for candidate in config_path.parents:
        if (candidate / 'fluxvla').is_dir():
            return candidate
    raise FileNotFoundError(
        f'Could not find the FluxVLA project root above {config_path}.')


def _resolve_checkpoint(path: Path) -> Path:
    checkpoint = path.expanduser().resolve()
    if checkpoint.suffix == '.pt':
        safetensors = checkpoint.with_suffix('.safetensors')
        if safetensors.is_file():
            checkpoint = safetensors
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Training checkpoint not found: {checkpoint}')
    return checkpoint


def _load_checkpoint(path: Path, torch: Any) -> Mapping[str, Any]:
    if path.suffix == '.safetensors':
        try:
            from safetensors.torch import load_file
        except Exception as error:
            raise DependencyError(
                'Loading the training checkpoint requires safetensors.') from error
        return load_file(str(path), device='cpu')
    loaded = torch.load(str(path), map_location='cpu')
    if isinstance(loaded, Mapping) and 'model' in loaded:
        loaded = loaded['model']
    if not isinstance(loaded, Mapping):
        raise TypeError(f'Checkpoint does not contain a state dict: {path}')
    return loaded


def _checkpoint_state_for_model(
    model: Any,
    state: Mapping[str, Any],
) -> Dict[str, Any]:
    target = model.state_dict()
    if set(state) == set(target) and all(
            tuple(state[name].shape) == tuple(value.shape)
            for name, value in target.items()):
        return dict(state)

    candidate_fn = getattr(model, '_mapped_name_candidates', None)
    name_mapping = getattr(model, 'name_mapping', None)
    if not callable(candidate_fn) or not name_mapping:
        missing = sorted(set(target) - set(state))
        unexpected = sorted(set(state) - set(target))
        raise RuntimeError('Checkpoint does not exactly match the export model; '
                           f'missing={missing[:8]}, unexpected={unexpected[:8]}')

    remapped = {}
    used = set()
    for name, target_value in target.items():
        candidates = [name]
        candidates.extend(mapped for _, mapped in candidate_fn(name))
        matches = [
            candidate for candidate in dict.fromkeys(candidates) if candidate in state
            and tuple(state[candidate].shape) == tuple(target_value.shape)
        ]
        if len(matches) != 1:
            raise RuntimeError(f'Checkpoint tensor {name!r} resolved to {matches}; '
                               'expected exactly one shape-compatible source tensor.')
        source = matches[0]
        if source in used:
            raise RuntimeError(
                f'Checkpoint source tensor {source!r} resolves to more than '
                'one model tensor.')
        remapped[name] = state[source]
        used.add(source)
    unexpected = sorted(set(state) - used)
    if unexpected:
        raise RuntimeError('Checkpoint contains tensors unused by the export model: '
                           f'{unexpected[:8]}')
    return remapped


def _save_outputs(path: Path, outputs: Mapping[str, Any]) -> None:
    try:
        import numpy as np
    except Exception as error:
        raise DependencyError('Saving accelerated outputs requires numpy.') from error
    arrays = {}
    for name, value in outputs.items():
        arrays[name] = value.detach().float().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _onnx_identity(path: Path, output_dir: Path) -> Dict[str, Any]:
    artifacts = [path]
    artifacts.extend(
        candidate for candidate in sorted(path.parent.glob(path.name + '.*'))
        if candidate.is_file())
    return {
        'graph':
        str(path.relative_to(output_dir)),
        'files': [{
            'file': str(artifact.relative_to(output_dir)),
            'size_bytes': artifact.stat().st_size,
            'sha256': sha256_file(artifact),
        } for artifact in artifacts],
    }


class PostTrainingPipeline:
    """Compile a completed FluxVLA checkpoint on the current CUDA GPU."""

    def __init__(
        self,
        training_config: str | Path,
        checkpoint: str | Path,
        output_dir: str | Path,
    ) -> None:
        self.training_config = Path(training_config).expanduser().resolve()
        self.checkpoint = _resolve_checkpoint(Path(checkpoint))
        self.output_dir = Path(output_dir).expanduser().resolve()
        if not self.training_config.is_file():
            raise FileNotFoundError(
                f'Training config not found: {self.training_config}')

    def _check_environment(self) -> None:
        missing = [
            name for name in ('torch', 'onnx', 'tensorrt')
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise DependencyError('Post-training TensorRT conversion requires: '
                                  f'{", ".join(missing)}.')

    def _load_model(self):
        root = _fluxvla_root(self.training_config)
        add_python_paths([str(root)])
        try:
            import fluxvla.models  # noqa: F401
            import torch
            from fluxvla.engines import build_vla_from_cfg
            from mmengine import Config
        except Exception as error:
            raise DependencyError(
                f'Loading the FluxVLA training model failed: {error}') from error
        if not torch.cuda.is_available():
            raise DependencyError(
                'Post-training TensorRT conversion requires the training CUDA GPU.')

        config = Config.fromfile(str(self.training_config))
        model_config = copy.deepcopy(config.model)
        family_type = detect_model_family(model_config)
        family_type.configure_model(model_config)
        family = family_type()
        dtype = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
        }[family.export_dtype]
        device = torch.device('cuda', torch.cuda.current_device())
        model = build_vla_from_cfg(model_config).eval()
        state = _checkpoint_state_for_model(
            model,
            _load_checkpoint(self.checkpoint, torch),
        )
        model.load_state_dict(state, strict=True)
        del state
        model.to(device=device, dtype=dtype).eval()
        torch.cuda.empty_cache()
        return device, model_config, family, model

    def _execute(self) -> Dict[str, Any]:
        started = time.perf_counter()
        self._check_environment()
        device, model_config, family, model = self._load_model()
        inputs = family.make_representative_inputs(
            model,
            model_config,
            device,
        )
        deployment = family.prepare(model, model_config, inputs)

        reference = deployment.run_reference()
        onnx_dir = self.output_dir / 'onnx'
        engine_dir = self.output_dir / 'engines'
        onnx_files: Dict[str, Path] = {}
        onnx_metadata: Dict[str, Dict[str, Any]] = {}
        engine_files: Dict[str, Path] = {}
        engine_metadata: Dict[str, Dict[str, Any]] = {}

        for name, graph in deployment.graphs.items():
            onnx_path = export_onnx(
                graph,
                onnx_dir / f'{name}.onnx',
                graph.export_options,
            )
            engine_path = engine_dir / f'{name}.plan'
            options = dict(graph.build_options)
            options['timing_cache'] = str(engine_dir / 'timing.cache')
            engine_metadata[name] = build_tensorrt_engine(
                onnx_path,
                engine_path,
                graph.input_specs,
                options,
            )
            engine_metadata[name]['sha256'] = sha256_file(engine_path)
            onnx_files[name] = onnx_path
            onnx_metadata[name] = _onnx_identity(onnx_path, self.output_dir)
            engine_files[name] = engine_path

        wrapper = TensorRTWrapper(engine_files, device=str(device))
        executor = family.create_executor(wrapper.engines, deployment)
        candidate = executor.infer(inputs, synchronize=True)
        validation_options = family.validation_options
        validation = compare_outputs(
            reference,
            candidate,
            atol=float(validation_options['atol']),
            rtol=float(validation_options['rtol']),
            min_cosine_similarity=float(validation_options['min_cosine_similarity']),
        )
        if not validation['passed']:
            raise ValidationError(
                f'Post-training TensorRT parity validation failed: {validation}')

        output_path = self.output_dir / 'accelerated_outputs.npz'
        _save_outputs(output_path, candidate)
        elapsed = time.perf_counter() - started
        first_engine = next(iter(engine_metadata.values()))
        target = {
            name: first_engine[name]
            for name in (
                'device',
                'compute_capability',
                'cuda_version',
                'tensorrt_version',
            )
        }
        return {
            'status': 'complete',
            'family': family.name,
            'training_config': str(self.training_config),
            'training_config_sha256': sha256_file(self.training_config),
            'checkpoint': str(self.checkpoint),
            'checkpoint_sha256': sha256_file(self.checkpoint),
            'target': target,
            'contract': deployment.metadata['contract'],
            'onnx': {
                name: str(path.relative_to(self.output_dir))
                for name, path in onnx_files.items()
            },
            'onnx_metadata': onnx_metadata,
            'engines': {
                name: str(path.relative_to(self.output_dir))
                for name, path in engine_files.items()
            },
            'engine_metadata': engine_metadata,
            'validation': validation,
            'accelerated_outputs': str(output_path.relative_to(self.output_dir)),
            'accelerated_outputs_sha256': sha256_file(output_path),
            'elapsed_seconds': elapsed,
        }

    def execute(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.output_dir / 'result.json'
        write_json(
            result_path,
            {
                'status': 'building',
                'training_config': str(self.training_config),
                'checkpoint': str(self.checkpoint),
            },
        )
        try:
            result = self._execute()
        except Exception as error:
            write_json(
                result_path,
                {
                    'status': 'failed',
                    'training_config': str(self.training_config),
                    'checkpoint': str(self.checkpoint),
                    'error': f'{type(error).__name__}: {error}',
                },
            )
            raise
        write_json(result_path, result)
        return result_path


def convert_after_training(
    training_config: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
) -> Path:
    """Public entry point for a training runner's completion callback."""
    return PostTrainingPipeline(
        training_config,
        checkpoint,
        output_dir,
    ).execute()
