from .physics import (
    available_physics_backends,
    list_physics_backends,
    physics_backend_readiness,
    physics_backend_from_string,
    register_physics_backend,
)
from .plugins import load_backend_plugins, normalize_plugin_modules, reset_loaded_backend_plugins
from .render import (
    available_render_backends,
    dispatch_render,
    list_render_backends,
    register_render_backend,
    render_backend_from_string,
)
from .registry import ENV_REGISTRY, available_env_backends, register_env_backend


def collect_physics_backend_readiness(*args, **kwargs):
    from .diagnostics import collect_physics_backend_readiness as _collect_physics_backend_readiness

    return _collect_physics_backend_readiness(*args, **kwargs)


def diagnose_real_backend_switch_test(*args, **kwargs):
    from .diagnostics import diagnose_real_backend_switch_test as _diagnose_real_backend_switch_test

    return _diagnose_real_backend_switch_test(*args, **kwargs)


def collect_backend_diagnostics(*args, **kwargs):
    from .diagnostics import collect_backend_diagnostics as _collect_backend_diagnostics

    return _collect_backend_diagnostics(*args, **kwargs)


__all__ = [
    "ENV_REGISTRY",
    "available_env_backends",
    "available_physics_backends",
    "available_render_backends",
    "dispatch_render",
    "list_physics_backends",
    "physics_backend_readiness",
    "collect_physics_backend_readiness",
    "diagnose_real_backend_switch_test",
    "collect_backend_diagnostics",
    "list_render_backends",
    "load_backend_plugins",
    "reset_loaded_backend_plugins",
    "normalize_plugin_modules",
    "physics_backend_from_string",
    "register_env_backend",
    "register_physics_backend",
    "register_render_backend",
    "render_backend_from_string",
]
