from .contract import PI05Contract

__all__ = [
    'PI05Contract',
    'PI05ModelFamily',
    'PI05SourceReferenceExecutor',
    'PI05TensorRTExecutor',
]


def __getattr__(name: str):
    if name == 'PI05ModelFamily':
        from .adapter import PI05ModelFamily
        return PI05ModelFamily
    if name in {'PI05SourceReferenceExecutor', 'PI05TensorRTExecutor'}:
        from .executor import PI05SourceReferenceExecutor, PI05TensorRTExecutor
        return {
            'PI05SourceReferenceExecutor': PI05SourceReferenceExecutor,
            'PI05TensorRTExecutor': PI05TensorRTExecutor,
        }[name]
    raise AttributeError(name)
