from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class StageRecord:
    name: str
    status: str
    started_at: float
    duration_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PipelineManager:
    """Stage runner with hooks and an auditable execution trace."""

    def __init__(self) -> None:
        self._before_hooks: Dict[str, List[Callable[..., None]]] = {}
        self._after_hooks: Dict[str, List[Callable[..., None]]] = {}
        self.records: List[StageRecord] = []

    def add_hook(
        self,
        stage: str,
        hook: Callable[..., None],
        when: str = 'after',
    ) -> None:
        if when not in {'before', 'after'}:
            raise ValueError("Pipeline hook position must be 'before' or 'after'.")
        collection = self._before_hooks if when == 'before' else self._after_hooks
        collection.setdefault(stage, []).append(hook)

    def run(
        self,
        stage: str,
        function: Callable[..., Any],
        *args: Any,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        record = StageRecord(
            name=stage,
            status='running',
            started_at=time.time(),
            metadata=dict(metadata or {}),
        )
        self.records.append(record)
        start = time.perf_counter()
        try:
            for hook in self._before_hooks.get(stage, ()):  # pragma: no branch
                hook(stage=stage, args=args, kwargs=kwargs, record=record)
            result = function(*args, **kwargs)
            record.status = 'completed'
            for hook in self._after_hooks.get(stage, ()):  # pragma: no branch
                hook(stage=stage, result=result, record=record)
            return result
        except Exception as error:
            record.status = 'failed'
            record.error = f'{type(error).__name__}: {error}'
            raise
        finally:
            record.duration_s = time.perf_counter() - start

    def report(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.records]
