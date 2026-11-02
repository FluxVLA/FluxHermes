from __future__ import annotations

import copy
import gc
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from .backend import BackendIR
from .backend.tensorrt import TensorRTWrapper, build_tensorrt_artifacts
from .bundle import verify_bundle
from .contracts import BundleManifest, TensorSpec
from .errors import DependencyError, ValidationError
from .exporters import export_onnx
from .model_families import detect_model_family
from .utils import sha256_file, write_json
from .validation import compare_outputs


def _resolve_checkpoint(path: Path) -> Path:
    checkpoint = path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Training checkpoint not found: {checkpoint}')
    return checkpoint


def _load_checkpoint(path: Path, torch: Any) -> Mapping[str, Any]:
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
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
    missing = sorted(set(target) - set(state))
    unexpected = sorted(set(state) - set(target))
    mismatched = sorted(
        name for name in set(target) & set(state)
        if tuple(state[name].shape) != tuple(target[name].shape))
    if missing or unexpected or mismatched:
        raise RuntimeError(
            'Training checkpoint does not exactly match the export model; '
            f'missing={missing[:8]}, unexpected={unexpected[:8]}, '
            f'shape_mismatch={mismatched[:8]}')
    return dict(state)


def _save_outputs(path: Path, outputs: Mapping[str, Any]) -> None:
    import numpy as np
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


def _tensor_specs(tensors: Mapping[str, Any]) -> tuple[TensorSpec, ...]:
    return tuple(
        TensorSpec(
            name=name,
            dtype=str(value.dtype).removeprefix('torch.'),
            shape=tuple(value.shape),
        ) for name, value in tensors.items())


def _fluxvla_identity(package_file: str) -> Dict[str, Any]:
    root = Path(package_file).resolve().parents[1]
    commit = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--short'],
        text=True,
    ).splitlines()
    return {
        'root': str(root),
        'commit': commit,
        'dirty': bool(status),
        'changes': status,
    }


def _representative_config(training_config: Mapping[str, Any]) -> Mapping[str, Any]:
    if 'eval' in training_config and 'dataset' in training_config['eval']:
        return training_config['eval']['dataset']
    if ('train_dataloader' in training_config
            and 'dataset' in training_config['train_dataloader']):
        return training_config['train_dataloader']['dataset']
    raise KeyError(
        'Training config requires eval.dataset or train_dataloader.dataset.')


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

    def _load_model(self):
        import fluxvla
        import fluxvla.models  # noqa: F401
        import torch
        from fluxvla.engines import build_vla_from_cfg
        from mmengine import Config
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
        return device, config, family, model, _fluxvla_identity(fluxvla.__file__)

    def _execute(self) -> Dict[str, Any]:
        import torch

        started = time.perf_counter()
        device, config, family, model, fluxvla_identity = self._load_model()
        config_dict = config.to_dict()
        inputs = family.make_representative_inputs(
            model,
            _representative_config(config_dict),
            device,
        )
        deployment = family.prepare(model, config.model, inputs)
        deployment_metadata = dict(deployment.metadata)

        deployment.run_reference()
        torch.cuda.synchronize(device)
        reference_started = time.perf_counter()
        reference = deployment.run_reference()
        torch.cuda.synchronize(device)
        reference_seconds = time.perf_counter() - reference_started
        onnx_dir = self.output_dir / 'onnx'
        engine_dir = self.output_dir / 'engines'
        onnx_files: Dict[str, Path] = {}
        onnx_metadata: Dict[str, Dict[str, Any]] = {}
        engine_files: Dict[str, Path] = {}
        engine_metadata: Dict[str, Dict[str, Any]] = {}
        backend_graphs: Dict[str, BackendIR] = {}

        for name, graph in deployment.graphs.items():
            onnx_path = export_onnx(
                graph,
                onnx_dir / f'{name}.onnx',
                graph.export_options,
            )
            options = dict(graph.build_options)
            options['timing_cache'] = str(engine_dir / 'timing.cache')
            backend_graphs[name] = BackendIR(
                path=onnx_path,
                inputs=graph.input_specs,
                options=options,
            )
            onnx_files[name] = onnx_path
            onnx_metadata[name] = _onnx_identity(onnx_path, self.output_dir)

        del graph
        del deployment
        del model
        gc.collect()
        torch.cuda.empty_cache()

        backend_result = build_tensorrt_artifacts(
            backend_graphs,
            engine_dir,
            {},
        )
        engine_files = dict(backend_result.artifacts)
        engine_metadata = dict(backend_result.metadata['artifacts'])
        for name, engine_path in engine_files.items():
            engine_metadata[name]['sha256'] = sha256_file(engine_path)

        wrapper = TensorRTWrapper(engine_files, device=str(device))
        executor = family.create_executor(wrapper.engines, deployment_metadata)
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

        inference_samples = []
        for _ in range(5):
            inference_started = time.perf_counter()
            executor.infer(inputs, synchronize=True)
            inference_samples.append(time.perf_counter() - inference_started)
        median_inference_seconds = statistics.median(inference_samples)
        benchmark = {
            'reference_seconds': reference_seconds,
            'tensorrt_median_seconds': median_inference_seconds,
            'tensorrt_samples_seconds': inference_samples,
            'speedup': reference_seconds / median_inference_seconds,
        }

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
        source = {
            'training_config': str(self.training_config),
            'training_config_sha256': sha256_file(self.training_config),
            'checkpoint': str(self.checkpoint),
            'checkpoint_sha256': sha256_file(self.checkpoint),
            'fluxvla': fluxvla_identity,
        }
        bundle_files = {
            str(path.relative_to(self.output_dir)): sha256_file(path)
            for directory in (onnx_dir, engine_dir)
            for path in directory.rglob('*') if path.is_file()
        }
        bundle_files[str(output_path.relative_to(self.output_dir))] = sha256_file(
            output_path)
        manifest = BundleManifest(
            schema_version=3,
            name=f'{family.name}-post-training',
            created_at=datetime.now(timezone.utc).isoformat(),
            backend='tensorrt',
            precision=family.export_dtype,
            backend_files=tuple(
                str(path.relative_to(self.output_dir))
                for path in engine_files.values()),
            backend_metadata={
                'artifacts': engine_metadata,
                **target,
            },
            source=source,
            inputs=_tensor_specs(inputs),
            outputs=_tensor_specs(candidate),
            runtime={
                'model_family': family.name,
                'action_output': 'actions',
            },
            validation=validation,
            files=bundle_files,
            metadata={
                'contract': deployment_metadata['contract'],
                'benchmark': benchmark,
            },
            backend_artifacts={
                name: str(path.relative_to(self.output_dir))
                for name, path in engine_files.items()
            },
        )
        manifest_path = self.output_dir / 'manifest.json'
        write_json(manifest_path, manifest.to_dict())
        verify_bundle(self.output_dir)
        return {
            'status': 'complete',
            'family': family.name,
            **source,
            'target': target,
            'contract': deployment_metadata['contract'],
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
            'benchmark': benchmark,
            'accelerated_outputs': str(output_path.relative_to(self.output_dir)),
            'accelerated_outputs_sha256': sha256_file(output_path),
            'bundle_manifest': str(manifest_path.relative_to(self.output_dir)),
            'elapsed_seconds': elapsed,
        }

    def execute(self) -> Path:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                'Post-training output directory must be empty: '
                f'{self.output_dir}')
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
