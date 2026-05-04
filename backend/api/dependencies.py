"""FastAPI dependency injection helpers.

All routes obtain the runtime object via Depends(get_runtime) instead of
referencing module-level variables directly, which lets tests swap it via
dependency_overrides.

Composition root strategy:
- `_runtime_factory` decides how to build DemoRuntime; default = create_demo_runtime.
- `set_runtime_factory()` allows replacing the factory in extension scenarios.
- `reset_runtime()` is used for test isolation, forcing the next init to rebuild.
"""

from typing import Callable

from backend.runtime import DemoRuntime, create_demo_runtime


RuntimeFactory = Callable[[], DemoRuntime]

_runtime: DemoRuntime | None = None
_runtime_factory: RuntimeFactory = create_demo_runtime


def set_runtime_factory(factory: RuntimeFactory) -> None:
    """Replace the runtime factory; only takes effect when called before init_runtime."""
    global _runtime_factory
    _runtime_factory = factory


def init_runtime() -> None:
    """Called during the application's lifespan startup; warms up and holds the runtime."""
    global _runtime
    _runtime = _runtime_factory()


def reset_runtime() -> None:
    """Clear the registered runtime object; mainly for test isolation."""
    global _runtime
    _runtime = None


def get_runtime() -> DemoRuntime:
    """FastAPI dependency that returns the initialized runtime object.

    If lifespan startup did not initialize successfully, raises RuntimeError,
    which is mapped to a 503 response by the global exception handler.
    """
    if _runtime is None:
        raise RuntimeError("Runtime is not initialized; check the application startup logs.")
    return _runtime
