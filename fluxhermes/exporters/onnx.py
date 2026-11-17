from __future__ import annotations

import inspect
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

from ..adapters.base import PreparedModel
from ..errors import ArtifactError, ConfigurationError, DependencyError
from .compat import EXPORT_LOCK, legacy_bf16_symbolics, normalize_full_slices


def _walk_tensor_protos(message):
    """Visit tensors in attributes, nested graphs, sparse values/indices, etc."""
    if message.DESCRIPTOR.full_name == 'onnx.TensorProto':
        yield message
        return
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            children = value if field.label == field.LABEL_REPEATED else (value, )
            for child in children:
                yield from _walk_tensor_protos(child)


def _restore_external_references(onnx, source, converted):
    """Restore only uniquely named tensors with identical non-reference fields."""
    source_tensors = list(_walk_tensor_protos(source))
    external = [
        value for value in source_tensors
        if value.data_location == onnx.TensorProto.EXTERNAL
    ]
    counts = Counter(value.name for value in source_tensors)
    by_name = {}
    for value in external:
        if not value.name or counts[value.name] != 1:
            raise ArtifactError(
                f'External tensor name is empty or ambiguous: {value.name!r}.')
        by_name[value.name] = []
    for value in _walk_tensor_protos(converted):
        if value.name in by_name:
            by_name[value.name].append(value)
    pairs = []
    for original in external:
        matches = by_name[original.name]
        if len(matches) != 1:
            raise ArtifactError(
                f'Conversion has {len(matches)} matches for external tensor '
                f'{original.name!r}; expected exactly one.')
        current = matches[0]
        before, after = onnx.TensorProto(), onnx.TensorProto()
        before.CopyFrom(original)
        after.CopyFrom(current)
        for tensor in (before, after):
            tensor.ClearField('external_data')
            tensor.ClearField('data_location')
        if before != after:
            raise ArtifactError(f'Conversion changed external tensor {original.name}; '
                                'its storage reference was not reassigned.')
        pairs.append((current, original))
    # Validate every match before changing any tensor or saving a graph.
    for current, original in pairs:
        current.CopyFrom(original)


def _load_convert(onnx, output: Path, target_opset):
    """Preserve all external references without loading external weight bytes."""
    model = onnx.load(str(output), load_external_data=False)
    if target_opset is not None:
        converted = onnx.version_converter.convert_version(model, target_opset)
        _restore_external_references(onnx, model, converted)
        model = converted
        onnx.save_model(model, str(output))
    return model


def export_onnx(
    prepared: PreparedModel,
    output_path: str | Path,
    options: Dict[str, Any],
) -> Path:
    """Export a prepared source model with stable tensor names.

    ``legacy_bf16_compat=True`` opts into validated Torch 2.6/2.8 opset18
    symbolics. ``target_opset=22`` runs the official ONNX version converter,
    preserving external-data references. Either option mandates full checking,
    even when ordinary ``check`` is disabled. No precision fallback is applied.
    """
    normalize_views = options.get('normalize_full_slices', False)
    if type(normalize_views) is not bool:
        raise ConfigurationError('normalize_full_slices must be a boolean.')
    if normalize_views and bool(options.get('dynamo', False)):
        raise ConfigurationError('normalize_full_slices requires dynamo=False.')
    compat = options.get('legacy_bf16_compat', False)
    if not isinstance(compat, bool):
        raise ConfigurationError('legacy_bf16_compat must be a boolean.')
    target_opset = options.get('target_opset')
    if target_opset is not None and (isinstance(target_opset, bool)
                                     or not isinstance(target_opset, int)):
        raise ConfigurationError('target_opset must be an integer.')
    if compat and bool(options.get('dynamo', False)):
        raise ConfigurationError('legacy_bf16_compat requires dynamo=False.')
    try:
        import torch
    except Exception as error:
        raise DependencyError('ONNX export requires PyTorch.') from error
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    input_names = list(prepared.inputs)
    arguments = tuple(prepared.inputs[name] for name in input_names)
    kwargs: Dict[str, Any] = {
        'input_names': input_names,
        'output_names': list(prepared.output_names),
        'opset_version': int(options.get('opset', 18)),
        'do_constant_folding': bool(options.get('do_constant_folding', True)),
        'dynamic_axes': prepared.dynamic_axes or None,
        'verbose': bool(options.get('verbose', False)),
    }
    signature = inspect.signature(torch.onnx.export)
    if 'dynamo' in signature.parameters:
        kwargs['dynamo'] = bool(options.get('dynamo', False))
    if 'external_data' in signature.parameters:
        kwargs['external_data'] = bool(options.get('external_data', True))
    if target_opset is not None and target_opset < kwargs['opset_version']:
        raise ConfigurationError('target_opset must not be lower than export opset.')
    context = (
        legacy_bf16_symbolics(torch, kwargs['opset_version'])
        if compat else nullcontext())
    view_context = normalize_full_slices(torch) if normalize_views else nullcontext({})
    with EXPORT_LOCK, context, view_context as view_stats, torch.inference_mode():
        torch.onnx.export(prepared.module, arguments, str(output), **kwargs)
    if normalize_views:
        prepared.metadata['jit_view_normalization'] = dict(view_stats)
    if not output.is_file():
        raise RuntimeError(f'PyTorch did not create the ONNX file: {output}')
    if bool(options.get('check', True)) or compat or target_opset is not None:
        try:
            import onnx
        except Exception as error:
            raise DependencyError('ONNX checking requires the onnx package.') from error
        # Keep large weight files external throughout conversion/serialization.
        model = _load_convert(onnx, output, target_opset)

        def info(value):
            tensor = value.type.tensor_type
            return dict(
                name=value.name,
                dtype=onnx.TensorProto.DataType.Name(tensor.elem_type),
                shape=[
                    dim.dim_param if dim.dim_param else
                    dim.dim_value if dim.HasField('dim_value') else None
                    for dim in tensor.shape.dim
                ])

        actual_names = [value.name for value in model.graph.input]
        pruned = sorted(set(input_names) - set(actual_names))
        prepared.metadata['onnx_io'] = dict(
            inputs=[info(value) for value in model.graph.input],
            outputs=[info(value) for value in model.graph.output],
            pruned_inputs=pruned,
            opsets={item.domain: item.version
                    for item in model.opset_import})
        if compat:
            if actual_names != input_names:
                raise ArtifactError(f'Unexpected ONNX input names: {actual_names}; '
                                    f'pruned={pruned}')
            if tuple(value.name
                     for value in model.graph.output) != prepared.output_names:
                raise ArtifactError('Unexpected ONNX output names.')
        onnx.checker.check_model(
            str(output), full_check=compat or target_opset is not None)
    return output
