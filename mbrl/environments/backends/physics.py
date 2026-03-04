import os
import time
from abc import ABC
from importlib import import_module
from importlib.util import find_spec


class PhysicsBackend(ABC):
    backend_name = ""
    display_name = ""
    implemented = True

    def __init__(self):
        self.options = {}

    def configure(self, options=None):
        self.options = options or {}

    def prepare_backend(self, options=None):
        """Backend-specific setup before constructing environment instances."""

    def create_env(self, *, env_string, env_registry, env_params):
        if env_string not in env_registry:
            raise ImportError(f"add '{env_string}' entry to ENV_REGISTRY")

        backend_entries = env_registry[env_string]
        if self.backend_name not in backend_entries:
            available_backends = sorted(backend_entries.keys())
            raise NotImplementedError(
                f"Env '{env_string}' is not implemented for physics backend '{self.backend_name}'. "
                f"Available backends: {available_backends}"
            )

        self.prepare_backend(options=self.options)

        env_package, env_class = backend_entries[self.backend_name]
        module = import_module(env_package, "mbrl.environments")
        cls = getattr(module, env_class)
        env = cls(**env_params, name=env_string)
        env.physics_backend = self.backend_name
        return env


class MujocoPhysicsBackend(PhysicsBackend):
    backend_name = "mujoco"
    display_name = "MuJoCo"
    implemented = True

    @staticmethod
    def _check_for_mujoco_lock():
        # Avoid stale mujoco-py build locks when many jobs start at once.
        try:
            import cloudpickle

            path = os.path.dirname(cloudpickle.__file__)
            site_packages_path = path.split("cloudpickle")[0]
            lock_file = os.path.join(site_packages_path, "mujoco_py", "generated", "mujocopy-buildlock.lock")
            while os.path.exists(lock_file):
                age_of_lock = time.time() - os.path.getmtime(lock_file)
                if age_of_lock > 300:
                    print(f"Deleting stale mujoco lock in {lock_file}")
                    os.remove(lock_file)
                else:
                    print(
                        f"waiting for mujoco lock to be released (I kill it in {round(300 - age_of_lock)}s) "
                        f"{lock_file}"
                    )
                    time.sleep(5)
        except BaseException:
            # Keep backward-compatible behavior: never fail hard on lock probing.
            pass

    def prepare_backend(self, options=None):
        self._check_for_mujoco_lock()


class _UnavailablePhysicsBackend(PhysicsBackend):
    backend_name = ""
    display_name = ""
    implemented = False

    def create_env(self, *, env_string, env_registry, env_params):
        raise NotImplementedError(
            f"Physics backend '{self.display_name}' is registered but not implemented yet in this repository. "
            f"Please add concrete env implementations for '{env_string}' under the '{self.backend_name}' backend "
            f"(e.g., via mbrl.environments.backends.register_env_backend)."
        )


class IsaacSimPhysicsBackend(_UnavailablePhysicsBackend):
    backend_name = "isaacsim"
    display_name = "Isaac Sim"


class GenesisPhysicsBackend(PhysicsBackend):
    backend_name = "genesis"
    display_name = "Genesis"
    implemented = True

    @staticmethod
    def _is_genesis_available(module_name):
        return find_spec(module_name) is not None

    def prepare_backend(self, options=None):
        options = options or {}
        if options.get("skip_dependency_check", False):
            return

        module_name = options.get("module_name", "genesis")
        if not self._is_genesis_available(module_name):
            raise ImportError(
                "Physics backend 'Genesis' selected, but the 'genesis' package is not available. "
                "Install the Genesis runtime or pass "
                "physics_backend_options={'skip_dependency_check': true} for synthetic tests."
            )


class NewtonPhysicsBackend(_UnavailablePhysicsBackend):
    backend_name = "newton"
    display_name = "NVIDIA Newton"


_PHYSICS_BACKENDS = {
    "mujoco": MujocoPhysicsBackend,
    "isaacsim": IsaacSimPhysicsBackend,
    "genesis": GenesisPhysicsBackend,
    "newton": NewtonPhysicsBackend,
}

_PHYSICS_ALIASES = {
    "default": "mujoco",
    "native": "mujoco",
    "isaac": "isaacsim",
    "isaac_sim": "isaacsim",
    "nvidia_newton": "newton",
}


def physics_backend_from_string(backend_name):
    normalized = (backend_name or "mujoco").lower()
    normalized = _PHYSICS_ALIASES.get(normalized, normalized)
    if normalized not in _PHYSICS_BACKENDS:
        available = sorted(_PHYSICS_BACKENDS.keys())
        raise KeyError(f"Unknown physics backend '{backend_name}'. Available: {available}")
    return _PHYSICS_BACKENDS[normalized]()


def list_physics_backends():
    info = {}
    for name, backend_cls in _PHYSICS_BACKENDS.items():
        info[name] = {
            "display_name": backend_cls.display_name or name,
            "implemented": bool(getattr(backend_cls, "implemented", False)),
        }
    return info


def register_physics_backend(backend_name, backend_cls, *, aliases=None, override=False):
    normalized = (backend_name or "").strip().lower()
    if not normalized:
        raise ValueError("backend_name must be non-empty")
    if not override and normalized in _PHYSICS_BACKENDS and _PHYSICS_BACKENDS[normalized] is not backend_cls:
        raise KeyError(f"Physics backend '{normalized}' is already registered")
    _PHYSICS_BACKENDS[normalized] = backend_cls

    for alias in aliases or []:
        alias_normalized = (alias or "").strip().lower()
        if not alias_normalized:
            raise ValueError("aliases must not contain empty values")
        if not override and alias_normalized in _PHYSICS_ALIASES and _PHYSICS_ALIASES[alias_normalized] != normalized:
            raise KeyError(f"Physics backend alias '{alias_normalized}' is already registered")
        _PHYSICS_ALIASES[alias_normalized] = normalized


def available_physics_backends():
    return sorted(_PHYSICS_BACKENDS.keys())
