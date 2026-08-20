from __future__ import annotations

import contextlib
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Tuple

from ..errors import ConfigurationError


@dataclass(frozen=True)
class RewriteContext:
    origin_func: Callable[..., Any]
    config: Any
    codebase: str
    backend: str
    ir: str


@dataclass(frozen=True)
class _RewriteRecord:
    target: str
    function: Callable[..., Any]
    codebase: str
    backend: str
    ir: str

    def matches(self, codebase: str, backend: str, ir: str) -> bool:
        fields = (
            (self.codebase, codebase),
            (self.backend, backend),
            (self.ir, ir),
        )
        return all(
            expected == 'default' or expected == actual
            for expected, actual in fields)

    def score(self, codebase: str, backend: str, ir: str) -> int:
        return sum(expected == actual for expected, actual in (
            (self.codebase, codebase),
            (self.backend, backend),
            (self.ir, ir),
        ))


def _resolve_owner(target: str) -> Tuple[Any, str]:
    parts = target.split('.')
    module = None
    split_index = 0
    for index in range(len(parts), 0, -1):
        try:
            module = importlib.import_module('.'.join(parts[:index]))
            split_index = index
            break
        except ImportError:
            continue
    if module is None:
        raise ConfigurationError(f'Cannot import rewrite target: {target}')
    owner = module
    for name in parts[split_index:-1]:
        owner = getattr(owner, name)
    return owner, parts[-1]


class FunctionRewriter:
    """Backend-aware temporary function rewriting during graph export."""

    def __init__(self) -> None:
        self._records: List[_RewriteRecord] = []

    def register_rewriter(
        self,
        func_name: str,
        codebase: str = 'default',
        backend: str = 'default',
        ir: str = 'default',
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:

        def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
            self._records.append(
                _RewriteRecord(func_name, function, codebase, backend, ir))
            return function

        return decorator

    def _selected(
        self,
        codebase: str,
        backend: str,
        ir: str,
    ) -> Dict[str, _RewriteRecord]:
        selected: Dict[str, _RewriteRecord] = {}
        for record in self._records:
            if not record.matches(codebase, backend, ir):
                continue
            current = selected.get(record.target)
            if current is None or record.score(codebase, backend, ir) >= current.score(
                    codebase, backend, ir):
                selected[record.target] = record
        return selected

    @contextlib.contextmanager
    def context(
        self,
        codebase: str,
        backend: str,
        ir: str,
        config: Any = None,
    ) -> Iterator[None]:
        restored = []
        for target, record in self._selected(codebase, backend, ir).items():
            owner, attribute = _resolve_owner(target)
            origin = getattr(owner, attribute)
            rewrite_context = RewriteContext(
                origin_func=origin,
                config=config,
                codebase=codebase,
                backend=backend,
                ir=ir,
            )

            def replacement(
                *args: Any,
                __record: _RewriteRecord = record,
                __context: RewriteContext = rewrite_context,
                **kwargs: Any,
            ) -> Any:
                return __record.function(__context, *args, **kwargs)

            setattr(owner, attribute, replacement)
            restored.append((owner, attribute, origin))
        try:
            yield
        finally:
            for owner, attribute, origin in reversed(restored):
                setattr(owner, attribute, origin)


FUNCTION_REWRITER = FunctionRewriter()
