from __future__ import annotations

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
    if set(reference) != set(candidate):
        raise ValidationError(
            f'Output names differ: {sorted(reference)} != {sorted(candidate)}')
    metrics: Dict[str, Any] = {}
    passed = True
    for name in reference:
        expected = _numpy(reference[name])
        actual = _numpy(candidate[name])
        if expected.shape != actual.shape:
            raise ValidationError(
                f'{name}: shape differs: {expected.shape} != {actual.shape}')
        delta = np.abs(expected - actual)
        scale = np.maximum(np.abs(expected), 1e-12)
        max_abs = float(delta.max(initial=0.0))
        max_rel = float((delta / scale).max(initial=0.0))
        flat_expected = expected.reshape(-1)
        flat_actual = actual.reshape(-1)
        denominator = float(
            np.linalg.norm(flat_expected) * np.linalg.norm(flat_actual))
        cosine = (
            float(np.dot(flat_expected, flat_actual) / denominator)
            if denominator > 0 else 1.0)
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
        import torch
    except Exception as error:
        raise ValidationError('Backend validation requires torch.') from error
    with torch.inference_mode():
        reference = prepared.run_reference()
    candidate = backend.infer(prepared.inputs, synchronize=True)
    result = compare_outputs(
        reference,
        candidate,
        atol=float(options.get('atol', 1e-2)),
        rtol=float(options.get('rtol', 1e-2)),
        min_cosine_similarity=float(
            options.get('min_cosine_similarity', 0.995)),
    )
    if not result['passed'] and bool(options.get('fail_on_error', True)):
        raise ValidationError(f'Backend parity validation failed: {result}')
    return result
