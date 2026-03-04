import inspect
from abc import ABC


def _filter_supported_kwargs(render_fn, kwargs):
    try:
        signature = inspect.signature(render_fn)
    except (TypeError, ValueError):
        return kwargs

    params = signature.parameters.values()
    supports_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)
    if supports_kwargs:
        return kwargs

    supported = {name for name in signature.parameters.keys() if name != "self"}
    return {key: value for key, value in kwargs.items() if key in supported}


def _invoke_render(render_fn, *, mode=None, width=None, height=None, camera_name=None, **kwargs):
    call_kwargs = dict(kwargs)
    if mode is not None:
        call_kwargs["mode"] = mode
    if width is not None:
        call_kwargs["width"] = width
    if height is not None:
        call_kwargs["height"] = height
    if camera_name is not None:
        call_kwargs["camera_name"] = camera_name

    filtered_kwargs = _filter_supported_kwargs(render_fn, call_kwargs)

    if width is not None and "width" not in filtered_kwargs:
        filtered_kwargs["render_width"] = width
    if height is not None and "height" not in filtered_kwargs:
        filtered_kwargs["render_height"] = height

    filtered_kwargs = _filter_supported_kwargs(render_fn, filtered_kwargs)

    if filtered_kwargs:
        return render_fn(**filtered_kwargs)
    if mode is not None:
        try:
            return render_fn(mode=mode)
        except TypeError:
            pass
    return render_fn()


class RenderBackend(ABC):
    backend_name = ""
    display_name = ""
    implemented = True

    def __init__(self):
        self.options = {}

    def attach(self, env, options=None):
        self.options = options or {}
        env._render_backend = self
        env.render_backend = self.backend_name
        return env

    def render(self, env, *, mode=None, width=None, height=None, camera_name=None, **kwargs):
        raise NotImplementedError


class NativeRenderBackend(RenderBackend):
    backend_name = "native"
    display_name = "Native renderer"
    implemented = True

    def render(self, env, *, mode=None, width=None, height=None, camera_name=None, **kwargs):
        return _invoke_render(
            env.render,
            mode=mode,
            width=width,
            height=height,
            camera_name=camera_name,
            **kwargs,
        )


class HeadlessRenderBackend(NativeRenderBackend):
    backend_name = "headless"
    display_name = "Headless renderer"
    implemented = True

    def render(self, env, *, mode=None, width=None, height=None, camera_name=None, **kwargs):
        mode = "rgb_array" if mode in (None, "human") else mode
        return super().render(
            env,
            mode=mode,
            width=width,
            height=height,
            camera_name=camera_name,
            **kwargs,
        )


class NoRenderBackend(RenderBackend):
    backend_name = "none"
    display_name = "No renderer"
    implemented = True

    def render(self, env, *, mode=None, width=None, height=None, camera_name=None, **kwargs):
        if mode == "rgb_array":
            raise RuntimeError("render_backend='none' cannot produce rgb_array frames")
        return None


class _UnavailableRenderBackend(RenderBackend):
    backend_name = ""
    display_name = ""
    implemented = False

    def attach(self, env, options=None):
        raise NotImplementedError(
            f"Render backend '{self.display_name}' is registered but not implemented yet in this repository. "
            f"Please provide a concrete integration for env '{env.name}' "
            f"(e.g., via mbrl.environments.backends.register_render_backend)."
        )


class IsaacSimRenderBackend(_UnavailableRenderBackend):
    backend_name = "isaacsim"
    display_name = "Isaac Sim renderer"


class GenesisRenderBackend(_UnavailableRenderBackend):
    backend_name = "genesis"
    display_name = "Genesis renderer"


class NewtonRenderBackend(_UnavailableRenderBackend):
    backend_name = "newton"
    display_name = "NVIDIA Newton renderer"


_RENDER_BACKENDS = {
    "native": NativeRenderBackend,
    "headless": HeadlessRenderBackend,
    "none": NoRenderBackend,
    "isaacsim": IsaacSimRenderBackend,
    "genesis": GenesisRenderBackend,
    "newton": NewtonRenderBackend,
}

_RENDER_ALIASES = {
    "default": "native",
    "mujoco": "native",
    "isaac": "isaacsim",
    "isaac_sim": "isaacsim",
    "nvidia_newton": "newton",
}


def render_backend_from_string(backend_name):
    normalized = (backend_name or "native").lower()
    normalized = _RENDER_ALIASES.get(normalized, normalized)
    if normalized not in _RENDER_BACKENDS:
        available = sorted(_RENDER_BACKENDS.keys())
        raise KeyError(f"Unknown render backend '{backend_name}'. Available: {available}")
    return _RENDER_BACKENDS[normalized]()


def list_render_backends():
    info = {}
    for name, backend_cls in _RENDER_BACKENDS.items():
        info[name] = {
            "display_name": backend_cls.display_name or name,
            "implemented": bool(getattr(backend_cls, "implemented", False)),
        }
    return info


def register_render_backend(backend_name, backend_cls, *, aliases=None, override=False):
    normalized = (backend_name or "").strip().lower()
    if not normalized:
        raise ValueError("backend_name must be non-empty")
    if not override and normalized in _RENDER_BACKENDS and _RENDER_BACKENDS[normalized] is not backend_cls:
        raise KeyError(f"Render backend '{normalized}' is already registered")
    _RENDER_BACKENDS[normalized] = backend_cls

    for alias in aliases or []:
        alias_normalized = (alias or "").strip().lower()
        if not alias_normalized:
            raise ValueError("aliases must not contain empty values")
        if not override and alias_normalized in _RENDER_ALIASES and _RENDER_ALIASES[alias_normalized] != normalized:
            raise KeyError(f"Render backend alias '{alias_normalized}' is already registered")
        _RENDER_ALIASES[alias_normalized] = normalized


def available_render_backends():
    return sorted(_RENDER_BACKENDS.keys())


def dispatch_render(env, *, mode=None, width=None, height=None, camera_name=None, **kwargs):
    backend = getattr(env, "_render_backend", None)
    if backend is None:
        return _invoke_render(
            env.render,
            mode=mode,
            width=width,
            height=height,
            camera_name=camera_name,
            **kwargs,
        )
    return backend.render(
        env,
        mode=mode,
        width=width,
        height=height,
        camera_name=camera_name,
        **kwargs,
    )
