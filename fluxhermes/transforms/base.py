"""Config-built deployment transforms exchanging a shared context dictionary."""
from __future__ import annotations

from typing import Any

from ..core import Registry
from ..errors import ConfigurationError

TRANSFORMS = Registry('deployment transforms')


class BaseTransform:
    name: str
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()

    def validate(self, available: set[str], config) -> set[str]:
        missing = set(self.requires) - available
        if missing:
            raise ConfigurationError(
                f'{type(self).__name__} requires {sorted(missing)} '
                'from earlier transforms.')
        return (available - set(self.invalidates)) | set(self.provides)

    def plan(self) -> dict[str, Any]:
        return dict(type=type(self).__name__, name=self.name, description=self.__doc__)


class Compose:
    """Build and execute transforms in exactly the configured order."""

    def __init__(self, transforms) -> None:
        if not isinstance(transforms, (list, tuple)) or not transforms:
            raise ConfigurationError('pipeline/transforms must be a nonempty list.')
        self.transforms = []
        for definition in transforms:
            if not isinstance(definition, dict) or not isinstance(
                    definition.get('type'), str):
                raise ConfigurationError('Each transform requires a string type.')
            try:
                transform = TRANSFORMS.build(definition)
            except TypeError as error:
                raise ConfigurationError(
                    f'Invalid transform {definition}: {error}') from error
            if not isinstance(transform, BaseTransform):
                raise ConfigurationError(
                    'Deployment transforms must extend BaseTransform.')
            self.transforms.append(transform)

    def validate(self, available: set[str], config) -> set[str]:
        available = set(available)
        for transform in self.transforms:
            available = transform.validate(available, config)
        return available

    def plan(self) -> list[dict[str, Any]]:
        return [transform.plan() for transform in self.transforms]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:

            def execute(context, step=transform):
                available = step.validate(set(context), context['config'])
                result = step(context)
                if not isinstance(result, dict):
                    raise ConfigurationError(
                        f'{type(step).__name__} must return a dictionary.')
                missing = set(step.provides) - result.keys()
                if missing:
                    raise ConfigurationError(
                        f'{type(step).__name__} did not provide {sorted(missing)}.')
                if result is not context:
                    context.update(result)
                for key in set(step.invalidates) - available:
                    context.pop(key, None)
                return context

            data = data['manager'].run(transform.name, execute, data)
        return data
