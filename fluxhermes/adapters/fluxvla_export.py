"""Installed FluxVLA SDK boundary; no training checkout or path injection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..contracts import TensorSpec
from ..errors import ArtifactError, ConfigurationError, DependencyError
from .base import ModelAdapter

if TYPE_CHECKING:
    from fluxvla_export import PreparedDeployment

_SPEC_FIELDS = (
    'name', 'dtype', 'shape', 'min_shape', 'opt_shape', 'max_shape')
_SOURCE_FIELDS = {'recipe', 'deployment_config', 'checkpoint', 'inputs'}


def tensor_specs_from_sdk(specs: Sequence[Any]) -> tuple[TensorSpec, ...]:
    """Translate SDK tensor boundaries without changing order or precision.

    Hermes-only semantic descriptions are intentionally not part of equality.
    Both static dimensions and all three dynamic profile shapes are preserved.
    """
    converted = []
    for spec in specs:
        try:
            fields = {name: getattr(spec, name) for name in _SPEC_FIELDS}
            converted.append(TensorSpec(**fields))
        except (AttributeError, TypeError, ValueError) as error:
            raise ConfigurationError(
                f'Invalid FluxVLA SDK tensor specification: {error}'
            ) from error
    names = tuple(spec.name for spec in converted)
    if len(names) != len(set(names)):
        raise ConfigurationError(
            f'FluxVLA SDK tensor names must be unique: {names}.')
    return tuple(converted)


def _validate_specs(
    sdk_specs: Sequence[Any],
    declared: Sequence[TensorSpec],
    *,
    owner: str,
) -> None:
    actual = tensor_specs_from_sdk(sdk_specs)
    actual_names = tuple(spec.name for spec in actual)
    declared_names = tuple(spec.name for spec in declared)
    if actual_names != declared_names:
        raise ConfigurationError(
            f'{owner}: names/order {declared_names} do not match '
            f'FluxVLA SDK {actual_names}.')
    for sdk_spec, hermes_spec in zip(actual, declared):
        for field in _SPEC_FIELDS[1:]:
            expected = getattr(sdk_spec, field)
            supplied = getattr(hermes_spec, field)
            if supplied != expected:
                raise ConfigurationError(
                    f'{owner}.{sdk_spec.name}: {field} {supplied!r} does not '
                    f'match FluxVLA SDK {expected!r}.')


class FluxVLAExportAdapter(ModelAdapter):
    """Return the SDK's native multi-graph deployment, not a model wrapper.

    The SDK owns model construction, artifact integrity, computation, graph
    splitting and runtime execution. Hermes only checks its declared policy
    boundary and refuses unvalidated global precision changes. The native
    metadata (including precision_policy and runtime) is preserved unchanged.
    """

    def __init__(self, source: Mapping[str, Any], deployment: Any) -> None:
        if not isinstance(source, Mapping):
            raise ConfigurationError(
                'export_package source must be a mapping.')
        if any(not isinstance(name, str) for name in source):
            raise ConfigurationError(
                'export_package source field names must be strings.')
        self.source = dict(source)
        self.deployment = deployment
        missing = sorted(_SOURCE_FIELDS - set(self.source))
        unknown = sorted(set(self.source) - _SOURCE_FIELDS - {'type'})
        if missing or unknown:
            raise ConfigurationError(
                'export_package source requires only recipe, '
                'deployment_config, checkpoint and inputs; '
                f'missing={missing}, unknown={unknown}. '
                'Training checkout paths and Python path injection are not '
                'supported.')
        if self.source.get('type', 'fluxvla_export') != 'fluxvla_export':
            raise ConfigurationError(
                'export_package source.type must be '
                'fluxvla_export if supplied.')
        recipe = self.source['recipe']
        if not isinstance(recipe, str) or not recipe.strip():
            raise ConfigurationError(
                'source.recipe must be a non-empty string.')

    def _artifact_path(self, field: str) -> Path:
        value = self.source[field]
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ConfigurationError(
                f'source.{field} must be an explicit non-empty path.')
        try:
            path = Path(value).expanduser()
            config_path = self.deployment.config_path
            if not path.is_absolute():
                if config_path is None:
                    raise ConfigurationError(
                        f'source.{field}: relative paths require the Hermes '
                        'configuration file path.')
                path = Path(config_path).parent / path
            path = path.resolve()
        except (OSError, RuntimeError, ValueError) as error:
            raise ConfigurationError(
                f'source.{field}: cannot resolve {value!r}: {error}'
            ) from error
        if field == 'deployment_config' and path.suffix != '.json':
            raise ConfigurationError(
                'source.deployment_config must be explicit deployment JSON; '
                'training Python configurations are not supported.')
        return path

    def _validate_precision(self) -> None:
        precision = self.deployment.precision
        if precision != 'fp32':
            raise ConfigurationError(
                'export_package preserves SDK precision_policy and requires '
                'backend precision=fp32; global FP16/INT8 retyping has not '
                f'been validated (requested {precision!r}).')
        conversion = self.deployment.optimization.get('fp16_conversion', False)
        if conversion is not False:
            raise ConfigurationError(
                'export_package forbids optimization.fp16_conversion; '
                'protected SDK precision must remain unchanged.')
        mode = self.deployment.quantization.get('mode', 'none')
        if mode != 'none':
            raise ConfigurationError(
                'export_package requires quantization.mode=none; global '
                'quantization of protected SDK computation is not validated.')

    def prepare(self, device: str) -> PreparedDeployment:
        self._validate_precision()
        paths = {
            field: self._artifact_path(field)
            for field in ('deployment_config', 'checkpoint', 'inputs')
        }
        try:
            import fluxvla_export as sdk
        except ImportError as error:
            raise DependencyError(
                'export_package requires the installed independent '
                f'fluxvla-export SDK and its dependencies: {error}') from error
        try:
            prepared = sdk.prepare_tensorrt(
                recipe=self.source['recipe'],
                config=paths['deployment_config'],
                checkpoint=paths['checkpoint'],
                inputs=paths['inputs'],
                device=device,
            )
        except sdk.ArtifactError as error:
            raise ArtifactError(f'FluxVLA SDK: {error}') from error
        except sdk.FluxVLAExportError as error:
            raise ConfigurationError(f'FluxVLA SDK: {error}') from error
        if not isinstance(prepared, sdk.PreparedDeployment):
            raise ConfigurationError(
                'FluxVLA SDK must return its native PreparedDeployment.')
        _validate_specs(
            prepared.policy_input_specs, self.deployment.inputs,
            owner='Policy inputs')
        _validate_specs(
            prepared.output_specs, self.deployment.outputs,
            owner='Policy outputs')
        for graph in prepared.graphs.values():
            # Ensure every graph boundary is representable without collapsing
            # graphs or substituting the policy's input/output specifications.
            tensor_specs_from_sdk(graph.input_specs)
            tensor_specs_from_sdk(graph.output_specs)
        policy = prepared.metadata.get('precision_policy')
        if not isinstance(policy, Mapping) or not policy:
            raise ConfigurationError(
                'FluxVLA SDK must expose metadata.precision_policy; '
                'refusing to infer or override protected precision.')
        folding = self.deployment.ir.get('do_constant_folding', False)
        if policy.get('autocast_dtype') == 'bfloat16' and folding is not False:
            raise ConfigurationError(
                'BF16 export_package requires '
                'ir_config.do_constant_folding=False: Torch 2.8 legacy ONNX '
                'constant folding fails for native BF16 computation. '
                'No retry or precision conversion is performed.')
        return prepared
