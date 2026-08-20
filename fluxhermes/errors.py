class FluxHermesError(RuntimeError):
    """Base error for actionable FluxHermes failures."""


class ConfigurationError(FluxHermesError):
    """Raised when a deployment configuration is invalid."""


class DependencyError(FluxHermesError):
    """Raised when an optional execution dependency is unavailable."""


class ArtifactError(FluxHermesError):
    """Raised when a deployment artifact is invalid or has been modified."""


class ValidationError(FluxHermesError):
    """Raised when source and deployed model outputs do not agree."""
