from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Generic, Iterator, Mapping, Optional, TypeVar

from ..errors import ConfigurationError

T = TypeVar('T')


class Registry(Generic[T]):
    """Small dependency-free registry for codebases, tasks, and backends."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._modules: Dict[str, T] = {}

    @property
    def module_dict(self) -> Mapping[str, T]:
        return dict(self._modules)

    def __contains__(self, name: str) -> bool:
        return name in self._modules

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def get(self, name: str) -> T:
        try:
            return self._modules[name]
        except KeyError as error:
            available = ', '.join(sorted(self._modules)) or '<empty>'
            raise ConfigurationError(
                f'{name!r} is not registered in {self.name}. '
                f'Available: {available}.') from error

    def register(
        self,
        name: Optional[str] = None,
        force: bool = False,
    ) -> Callable[[T], T]:

        def decorator(module: T) -> T:
            key = name or getattr(module, '__name__', None)
            if not key:
                raise ConfigurationError(
                    f'Cannot infer a name for an item in {self.name}.')
            if key in self._modules and not force:
                raise ConfigurationError(
                    f'{key!r} is already registered in {self.name}.')
            self._modules[key] = module
            return module

        return decorator

    def register_module(
        self,
        name: Optional[str] = None,
        force: bool = False,
    ) -> Callable[[T], T]:
        """MMEngine/MMDeploy-style alias."""
        return self.register(name=name, force=force)

    def build(self, config: Mapping[str, Any], **default_kwargs: Any) -> Any:
        values = deepcopy(dict(config))
        type_value = values.pop('type', None)
        if type_value is None:
            raise ConfigurationError(
                f'Build config for {self.name} must contain type.')
        module = self.get(type_value) if isinstance(type_value, str) else type_value
        kwargs = dict(default_kwargs)
        kwargs.update(values)
        return module(**kwargs)
