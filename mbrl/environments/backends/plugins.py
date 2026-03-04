from importlib import import_module


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


def load_backend_plugins(plugin_modules):
    loaded_modules = []
    for module_name in normalize_plugin_modules(plugin_modules):
        try:
            import_module(module_name)
        except Exception as error:
            raise ImportError(f"Failed to import backend plugin module '{module_name}'") from error
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
