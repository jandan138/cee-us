from importlib import import_module, reload
from threading import Lock

_REGISTERED_PLUGIN_MODULES = set()
_PLUGIN_REGISTRATION_LOCK = Lock()


def normalize_plugin_modules(plugin_modules):
    if plugin_modules is None:
        return []
    if isinstance(plugin_modules, str):
        plugin_modules = [plugin_modules]
    if not isinstance(plugin_modules, (list, tuple)):
        raise TypeError("backend_plugin_modules must be a string or a list/tuple of strings")

    normalized = []
    seen = set()
    for module_name in plugin_modules:
        if not isinstance(module_name, str):
            raise TypeError("backend_plugin_modules values must be strings")
        stripped = module_name.strip()
        if not stripped or stripped in seen:
            continue
        normalized.append(stripped)
        seen.add(stripped)
    return normalized


def load_backend_plugins(plugin_modules, *, force_reload=False):
    loaded_modules = []
    for module_name in normalize_plugin_modules(plugin_modules):
        with _PLUGIN_REGISTRATION_LOCK:
            if not force_reload and module_name in _REGISTERED_PLUGIN_MODULES:
                loaded_modules.append(module_name)
                continue

            try:
                module = import_module(module_name)
            except Exception as error:
                raise ImportError(f"Failed to import backend plugin module '{module_name}'") from error

            if force_reload:
                module = reload(module)

            register_fn = getattr(module, "register_backends", None)
            if callable(register_fn):
                register_fn()

            _REGISTERED_PLUGIN_MODULES.add(module_name)
            loaded_modules.append(module_name)
    return loaded_modules


# Backward-compatible alias kept for older internal call sites.
def load_backend_plugins_from_params(params):
    if params is None:
        return []

    modules = []
    if isinstance(params, dict):
        modules = params.get("backend_plugins", [])
    elif hasattr(params, "get"):
        try:
            modules = params.get("backend_plugins", [])
        except Exception:
            modules = getattr(params, "backend_plugins", [])
    else:
        modules = getattr(params, "backend_plugins", [])

    return load_backend_plugins(modules)


def reset_loaded_backend_plugins():
    with _PLUGIN_REGISTRATION_LOCK:
        _REGISTERED_PLUGIN_MODULES.clear()
