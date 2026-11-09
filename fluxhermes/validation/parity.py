from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from ..adapters.base import PreparedModel
from ..backend.base import BaseBackendWrapper
from ..errors import ValidationError
from ..runtime.tensorrt import TensorRTEngine


def _numpy(value: Any):
    try:
        import numpy as np
    except Exception as error:
        raise ValidationError('Parity validation requires numpy.') from error
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'float'):
        value = value.float()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def compare_outputs(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    atol: float,
    rtol: float,
    min_cosine_similarity: float,
) -> Dict[str, Any]:
    try:
        import numpy as np
    except Exception as error:
        raise ValidationError('Parity validation requires numpy.') from error
    metrics: Dict[str, Any] = {}

    def structural_error(message, **details):
        error = ValidationError(message)
        error.report = {
            'passed': False,
            'outputs': metrics,
            'error': {
                'type': 'ValidationError',
                'message': message,
                **details
            }
        }
        raise error

    if not (np.isfinite([atol, rtol, min_cosine_similarity]).all() and atol >= 0
            and rtol >= 0 and -1 <= min_cosine_similarity <= 1):
        structural_error(
            'Parity thresholds must be finite; tolerances nonnegative; '
            'cosine in [-1,1].'
        )
    if not reference or set(reference) != set(candidate):
        structural_error(
            f'Output names differ or are empty: '
            f'{sorted(reference)} != {sorted(candidate)}',
            expected_names=sorted(reference),
            actual_names=sorted(candidate))
    passed = True
    for name in reference:
        expected = _numpy(reference[name])
        actual = _numpy(candidate[name])
        if expected.shape != actual.shape:
            structural_error(
                f'{name}: shape differs: {expected.shape} != {actual.shape}',
                output=name,
                expected_shape=list(expected.shape),
                actual_shape=list(actual.shape))
        if not (np.isfinite(expected).all() and np.isfinite(actual).all()):
            metrics[name] = dict(
                shape=list(expected.shape),
                max_abs_error=None,
                max_rel_error=None,
                cosine_similarity=None,
                finite=False,
                passed=False,
                reason='non-finite reference or candidate')
            passed = False
            continue
        delta = np.abs(expected - actual)
        scale = np.maximum(np.abs(expected), 1e-12)
        max_abs = float(delta.max(initial=0.0))
        max_rel = float((delta / scale).max(initial=0.0))
        flat_expected = expected.reshape(-1)
        flat_actual = actual.reshape(-1)
        denominator = float(np.linalg.norm(flat_expected) * np.linalg.norm(flat_actual))
        cosine = (
            float(np.dot(flat_expected, flat_actual) /
                  denominator) if denominator > 0 else 1.0)
        finite = bool(np.isfinite(actual).all())
        close = bool(np.allclose(expected, actual, atol=atol, rtol=rtol))
        tensor_passed = finite and close and cosine >= min_cosine_similarity
        passed = passed and tensor_passed
        metrics[name] = {
            'shape': list(expected.shape),
            'max_abs_error': max_abs,
            'max_rel_error': max_rel,
            'cosine_similarity': cosine,
            'finite': finite,
            'passed': tensor_passed,
        }
    return {
        'passed': passed,
        'atol': atol,
        'rtol': rtol,
        'min_cosine_similarity': min_cosine_similarity,
        'outputs': metrics,
    }


def validate_engine(
    prepared: PreparedModel,
    engine_path: str,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    engine = TensorRTEngine(engine_path)
    return validate_backend(prepared, engine, options)


def validate_backend(
    prepared: PreparedModel,
    backend: BaseBackendWrapper | TensorRTEngine,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        return _validate_backend(prepared, backend, options)
    except Exception as error:
        if not hasattr(error, 'report'):
            error.report = {
                'passed': False,
                'error': {
                    'type': type(error).__name__,
                    'message': str(error)
                }
            }
        raise


def _validate_backend(prepared, backend, options):
    try:
        import torch
    except Exception as error:
        raise ValidationError('Backend validation requires torch.') from error
    thresholds = {
        'atol': float(options.get('atol', 1e-2)),
        'rtol': float(options.get('rtol', 1e-2)),
        'min_cosine_similarity': float(options.get('min_cosine_similarity', 0.995)),
    }

    def compare_sample(inputs):
        # Give each runner its own storage: native in-place ops must not affect
        # the backend inputs or the representative/sample tensors.
        with torch.inference_mode():
            reference = prepared.run_reference(_clone_inputs(inputs))
        candidate = backend.infer(_clone_inputs(inputs), synchronize=True)
        return compare_outputs(reference, candidate, **thresholds)

    if 'samples_manifest' not in options:
        result = compare_sample(prepared.inputs)
    else:
        manifest = Path(options['samples_manifest']).expanduser().resolve()
        samples = _read_samples_manifest(manifest)
        result = {
            'passed': True,
            **thresholds,
            'sample_count': len(samples),
            'samples': [],
        }
        for sample in samples:
            entry = {'id': sample['id']}
            try:
                path = Path(sample['inputs']).expanduser()
                if not path.is_absolute():
                    path = manifest.parent / path
                inputs, info = _load_sample(path, prepared.inputs)
                entry['inputs'] = info
                entry.update(compare_sample(inputs))
            except Exception as error:
                # Runtime failures are never numerical failures or skipped cases.
                # Preserve the actual exception and the completed report prefix.
                entry.update(
                    passed=False,
                    error={
                        'type': type(error).__name__,
                        'message': str(error),
                    })
                result['samples'].append(entry)
                result['passed'] = False
                error.report = result
                raise
            result['samples'].append(entry)
            result['passed'] = result['passed'] and entry['passed']
    if not result['passed'] and bool(options.get('fail_on_error', True)):
        error = ValidationError(f'Backend parity validation failed: {result}')
        error.report = result
        raise error
    return result


def _clone_inputs(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    from copy import deepcopy

    return {
        name: value.clone() if hasattr(value, 'clone') else deepcopy(value)
        for name, value in inputs.items()
    }


def _read_samples_manifest(path: Path):
    with path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or set(manifest) != {'samples'}:
        raise ValidationError('Samples manifest must contain only "samples".')
    samples = manifest['samples']
    if not isinstance(samples, list) or not samples:
        raise ValidationError('Samples manifest requires a nonempty samples list.')
    seen = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != {'id', 'inputs'}:
            raise ValidationError(
                f'Sample {index} must contain exactly "id" and "inputs".')
        sample_id = sample['id']
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValidationError(f'Sample {index} requires a nonempty string id.')
        if sample_id in seen:
            raise ValidationError(f'Duplicate sample id: {sample_id!r}')
        seen.add(sample_id)
        if not isinstance(sample['inputs'], str) or not sample['inputs'].strip():
            raise ValidationError(f'Sample {sample_id!r} requires an inputs path.')
    return samples


def _load_sample(path: Path, templates: Mapping[str, Any]):
    import numpy as np
    import torch

    inputs, info = {}, {}
    with np.load(path, allow_pickle=False) as arrays:
        if (len(arrays.files) != len(set(arrays.files))
                or set(arrays.files) != set(templates)):
            raise ValidationError(
                f'{path}: input names differ: {sorted(arrays.files)} != '
                f'{sorted(templates)}')
        for name, template in templates.items():
            array = arrays[name]
            if not isinstance(template, torch.Tensor):
                raise ValidationError(f'{name}: prepared input must be a Torch tensor.')
            # Conversion follows the prepared contract, not strict NPZ dtype rejection.
            inputs[name] = torch.from_numpy(array.copy()).to(
                device=template.device, dtype=template.dtype)
            info[name] = {
                'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
                'shape': list(array.shape),
                'source_dtype': str(array.dtype),
                'dtype': str(inputs[name].dtype),
                'device': str(inputs[name].device),
            }
    return inputs, info
