"""Backend-independent, checked ONNX artifacts and portable external weights.

Only external-data locations are rewritten during staging. Tensor bytes and
ONNX dtypes, including BF16, are preserved; backend kernel support is checked
by the backend, not by these packaging utilities. ONNX is imported lazily.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

from ..errors import ArtifactError, DependencyError
from ..utils import sha256_file


def require_onnx():
    try:
        import onnx
    except ImportError as error:
        raise DependencyError(
            'ONNX artifact validation requires optional dependency onnx.') from error
    return onnx


def iter_model_messages(message: Any) -> Iterator[Any]:
    """Traverse tensors/types in subgraphs, sparse values and node attributes.

    Message-vs-container inspection supports protobuf 5 and 7 without using
    their incompatible FieldDescriptor.label / is_repeated APIs.
    """
    from google.protobuf.message import Message

    yield message
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        if isinstance(value, Message):
            yield from iter_model_messages(value)
        else:
            for child in value:
                yield from iter_model_messages(child)


def _relative_location(location: str) -> PurePosixPath:
    relative = PurePosixPath(location)
    if (not location or '\\' in location or ':' in location or '\x00' in location
            or relative.is_absolute()
            or any(part in ('.', '..') for part in relative.parts)
            or str(relative) != location or location == '.'):
        raise ArtifactError(f'Unsafe ONNX external-data location: {location!r}')
    return relative


def _external_fields(tensor: Any) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for entry in tensor.external_data:
        if entry.key not in {'location', 'offset', 'length', 'checksum'}:
            raise ArtifactError(
                f'{tensor.name}: unsupported external-data key {entry.key!r}.')
        if entry.key in fields:
            raise ArtifactError(
                f'{tensor.name}: duplicate external-data key {entry.key!r}.')
        fields[entry.key] = entry.value
    if 'location' not in fields:
        raise ArtifactError(f'{tensor.name}: external tensor has no location.')
    return fields


def _range_value(fields: Mapping[str, str], key: str, default: int, name: str) -> int:
    if key not in fields:
        return default
    value = fields[key]
    if not value.isascii() or not value.isdecimal():
        raise ArtifactError(
            f'{name}: external-data {key} must be a nonnegative integer, got {value!r}.'
        )
    return int(value)


def checked_model(graph_path: Path) -> Tuple[Any, Dict[str, Path]]:
    """Validate declared external data before running the full ONNX checker."""
    onnx = require_onnx()
    graph_path = Path(graph_path).resolve(strict=True)
    if not graph_path.is_file():
        raise ArtifactError(f'ONNX graph is not a regular file: {graph_path}')
    model = onnx.load_model(str(graph_path), load_external_data=False)
    root = graph_path.parent
    external: Dict[str, Path] = {}
    verified_checksums = {}
    for message in iter_model_messages(model):
        if not isinstance(message, onnx.TensorProto):
            continue
        if message.data_location != onnx.TensorProto.EXTERNAL:
            if message.external_data:
                raise ArtifactError(
                    f'{message.name}: external-data fields require '
                    'EXTERNAL data_location.'
                )
            continue
        if message.HasField('raw_data'):
            raise ArtifactError(
                f'{message.name}: external tensor must not also contain raw_data.')
        fields = _external_fields(message)
        relative = _relative_location(fields['location'])
        candidate = root.joinpath(*relative.parts)
        try:
            path = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise ArtifactError(
                f'{message.name}: ONNX external data is missing: {candidate}'
            ) from error
        if not path.is_relative_to(root) or not path.is_file():
            raise ArtifactError(
                f'{message.name}: external data escapes the graph directory '
                f'or is not a file: {candidate}'
            )
        if path == graph_path:
            raise ArtifactError(
                f'{message.name}: external data cannot reference the graph itself.')
        size = path.stat().st_size
        offset = _range_value(fields, 'offset', 0, message.name)
        length = _range_value(fields, 'length', size - offset, message.name)
        if offset > size or length < 0 or offset + length > size:
            raise ArtifactError(
                f'{message.name}: external-data range offset={offset}, '
                f'length={length} exceeds {path} ({size} bytes).'
            )
        if 'checksum' in fields:
            if path not in verified_checksums:
                digest = hashlib.sha1()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                verified_checksums[path] = digest.hexdigest()
            if fields['checksum'].lower() != verified_checksums[path]:
                raise ArtifactError(
                    f'{message.name}: ONNX external-data SHA1 checksum mismatch: {path}'
                )
        external[fields['location']] = path
    try:
        onnx.checker.check_model(str(graph_path), full_check=True)
    except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as error:
        raise ArtifactError(f'Invalid ONNX graph {graph_path}: {error}') from error
    return model, external


def stage_graph(graph_path: Path,
                output_dir: Path) -> Tuple[Tuple[Path, ...], Dict[str, Any]]:
    """Flatten external files safely, rewriting locations only (not tensor data)."""
    onnx = require_onnx()
    graph_path = Path(graph_path).resolve(strict=True)
    original_hash = sha256_file(graph_path)
    model, external = checked_model(graph_path)
    source_hashes = {location: sha256_file(path) for location, path in external.items()}
    if sha256_file(graph_path) != original_hash:
        raise ArtifactError(f'ONNX source changed during validation: {graph_path}')
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ArtifactError(
            f'ONNX artifact output directory already exists: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = output_dir.resolve(strict=True)
    locations = {
        location: f'external_{index:04d}.data'
        for index, location in enumerate(sorted(external))
    }
    files = [output_dir / 'model.onnx']
    for location, filename in locations.items():
        destination = output_dir / filename
        with external[location].open('rb') as source, destination.open('xb') as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if sha256_file(destination) != source_hashes[location]:
            raise ArtifactError(
                f'External weights changed during copy: {external[location]}')
        files.append(destination)
    for message in iter_model_messages(model):
        if isinstance(message, onnx.TensorProto
                      ) and message.data_location == onnx.TensorProto.EXTERNAL:
            for entry in message.external_data:
                if entry.key == 'location':
                    entry.value = locations[entry.value]
    with files[0].open('xb') as target:
        target.write(model.SerializeToString())
    checked_model(files[0])
    metadata = {
        'graph_file':
        'model.onnx',
        'artifact_layout':
        'flat',
        'source_graph_sha256':
        original_hash,
        'artifacts': {
            path.name: {
                'sha256': sha256_file(path),
                'size_bytes': path.stat().st_size
            }
            for path in files
        },
        'external_data': [{
            'source_location': location,
            'file': filename,
            'sha256': source_hashes[location]
        } for location, filename in locations.items()],
    }
    return tuple(files), metadata


def validate_backend_files(files: Sequence[Path],
                           metadata: Mapping[str, Any] | None = None) -> Path:
    """Check one flat staged graph, complete external files and optional hashes.

    Metadata describes artifacts only. Provider, device and backend identity
    are intentionally not interpreted by this backend-independent validator.
    """
    if not files:
        raise ArtifactError(
            'ONNX artifacts require exactly one graph and its external-data files.')
    resolved = tuple(Path(path).resolve(strict=True) for path in files)
    if len(set(resolved)) != len(resolved):
        raise ArtifactError('Duplicate ONNX backend artifact paths.')
    graphs = [path for path in resolved if path.suffix.lower() == '.onnx']
    if len(graphs) != 1:
        raise ArtifactError('ONNX artifacts require exactly one .onnx graph.')
    graph = graphs[0]
    if any(path.parent != graph.parent for path in resolved):
        raise ArtifactError('ONNX backend artifacts must use the staged flat layout.')
    _, external = checked_model(graph)
    expected = {graph, *external.values()}
    if set(resolved) != expected:
        raise ArtifactError(
            'ONNX backend file coverage differs: '
            f'missing={sorted(str(path) for path in expected - set(resolved))}, '
            f'extra={sorted(str(path) for path in set(resolved) - expected)}'
        )
    if metadata is not None:
        if metadata.get('artifact_layout') != 'flat' or metadata.get(
                'graph_file') != graph.name:
            raise ArtifactError('ONNX metadata must identify the flat graph layout.')
        artifacts = metadata.get('artifacts')
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
                path.name
                for path in resolved
        }:
            raise ArtifactError(
                'ONNX artifact checksums must cover every backend file exactly.')
        for path in resolved:
            record = artifacts[path.name]
            if (not isinstance(record, Mapping)
                    or record.get('size_bytes') != path.stat().st_size
                    or record.get('sha256') != sha256_file(path)):
                raise ArtifactError(f'ONNX artifact integrity mismatch: {path}')
    return graph
