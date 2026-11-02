from .contract import GR00TN15Contract

__all__ = [
    'GR00TN15Contract',
    'GR00TN15ModelFamily',
    'GR00TReferenceExecutor',
    'GR00TTensorRTExecutor',
]


def __getattr__(name: str):
    if name == 'GR00TN15ModelFamily':
        from .adapter import GR00TN15ModelFamily
        return GR00TN15ModelFamily
    if name in {'GR00TReferenceExecutor', 'GR00TTensorRTExecutor'}:
        from .executor import GR00TReferenceExecutor, GR00TTensorRTExecutor
        return {
            'GR00TReferenceExecutor': GR00TReferenceExecutor,
            'GR00TTensorRTExecutor': GR00TTensorRTExecutor,
        }[name]
    raise AttributeError(name)
