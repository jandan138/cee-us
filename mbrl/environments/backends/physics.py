import os
import shlex
import subprocess
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
        # Backend readiness metadata used by diagnostics.
        self.dependency_source = "local"

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

    @staticmethod
    def _normalize_external_python_command(external_python):
        if isinstance(external_python, str):
            parts = shlex.split(external_python.strip())
        elif isinstance(external_python, (list, tuple)):
            parts = []
            for part in external_python:
                if part is None:
                    continue
                normalized_part = str(part).strip()
                if normalized_part:
                    parts.append(normalized_part)
        else:
            raise TypeError("external_python must be a command string or a list/tuple of argv parts")

        if not parts:
            raise ValueError("external_python must not be empty")
        return parts

    @classmethod
    def _is_genesis_available_via_external_runtime(
        cls,
        module_name,
        external_python,
        *,
        pyopengl_platform=None,
        timeout_seconds=10.0,
        extra_env=None,
    ):
        command = cls._normalize_external_python_command(external_python)
        probe_code = (
            "import importlib.util, sys; "
            "module_name = sys.argv[1]; "
            "sys.exit(0 if importlib.util.find_spec(module_name) is not None else 1)"
        )
        probe_command = command + ["-c", probe_code, module_name]

        probe_env = dict(os.environ)
        if extra_env is not None:
            if not isinstance(extra_env, dict):
                raise TypeError("external_probe_env/external_env must be a dict when provided")
            for env_key, env_value in extra_env.items():
                if env_value is None:
                    continue
                probe_env[str(env_key)] = str(env_value)
        if pyopengl_platform not in (None, ""):
            probe_env["PYOPENGL_PLATFORM"] = str(pyopengl_platform)

        try:
            result = subprocess.run(
                probe_command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=probe_env,
                timeout=float(timeout_seconds),
            )
        except subprocess.TimeoutExpired:
            return False, f"external runtime probe timed out after {timeout_seconds}s"
        except Exception as error:
            return False, f"external runtime probe failed to execute: {type(error).__name__}: {error}"

        if result.returncode == 0:
            return True, "module import probe succeeded in external runtime"

        probe_output = (result.stderr or "").strip() or (result.stdout or "").strip()
        if probe_output:
            return False, f"external runtime probe returned {result.returncode}: {probe_output}"
        return False, f"external runtime probe returned {result.returncode}"

    def prepare_backend(self, options=None):
        options = options or {}
        if options.get("skip_dependency_check", False):
            self.dependency_source = "synthetic"
            return

        module_name = options.get("module_name", "genesis")
        if self._is_genesis_available(module_name):
            self.dependency_source = "local"
            return

        external_python = options.get("external_python")
        external_probe_reason = ""
        if external_python:
            self.dependency_source = "external"
            pyopengl_platform = options.get("PYOPENGL_PLATFORM", options.get("pyopengl_platform"))
            external_probe_env = options.get("external_probe_env", options.get("external_env"))
            timeout_seconds = options.get("external_probe_timeout_sec", 10.0)
            available_in_external_runtime, external_probe_reason = self._is_genesis_available_via_external_runtime(
                module_name,
                external_python,
                pyopengl_platform=pyopengl_platform,
                timeout_seconds=timeout_seconds,
                extra_env=external_probe_env,
            )
            if available_in_external_runtime:
                return
        else:
            self.dependency_source = "local"

        error_message = (
            "Physics backend 'Genesis' selected, but the 'genesis' package is not available. "
            "Install the Genesis runtime or pass "
            "physics_backend_options={'skip_dependency_check': true} for synthetic tests."
        )
        if external_python:
            error_message += f" External runtime probe attempted via external_python={external_python!r}"
            if external_probe_reason:
                error_message += f" and failed ({external_probe_reason})."
            else:
                error_message += " and failed."
        raise ImportError(error_message)


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


def _runtime_mode_from_dependency_source(dependency_source):
    normalized = str(dependency_source or "").strip().lower()
    if normalized == "synthetic":
        return "synthetic-runtime"
    if normalized == "external":
        return "external-runtime"
    if normalized in {"local", ""}:
        return "true-runtime"
    return "unknown-runtime"


def physics_backend_readiness(backend_name, options=None):
    readiness = {
        "backend": backend_name,
        "ready": False,
        "error_type": None,
        "reason": "",
        "dependency_source": "unknown",
        "runtime_mode": "unknown-runtime",
    }
    try:
        backend = physics_backend_from_string(backend_name)
    except Exception as error:
        readiness["error_type"] = type(error).__name__
        readiness["reason"] = str(error)
        return readiness

    readiness["backend"] = backend.backend_name
    readiness["dependency_source"] = getattr(backend, "dependency_source", "local")
    readiness["runtime_mode"] = _runtime_mode_from_dependency_source(readiness["dependency_source"])
    if not bool(getattr(backend, "implemented", False)):
        readiness["error_type"] = "NotImplementedError"
        readiness["reason"] = (
            f"Physics backend '{backend.display_name or backend.backend_name}' is registered but not implemented."
        )
        return readiness

    backend.configure(options=options or {})
    try:
        backend.prepare_backend(options=backend.options)
    except Exception as error:
        readiness["dependency_source"] = getattr(backend, "dependency_source", readiness["dependency_source"])
        readiness["runtime_mode"] = _runtime_mode_from_dependency_source(readiness["dependency_source"])
        readiness["error_type"] = type(error).__name__
        readiness["reason"] = str(error)
        return readiness

    readiness["dependency_source"] = getattr(backend, "dependency_source", readiness["dependency_source"])
    readiness["runtime_mode"] = _runtime_mode_from_dependency_source(readiness["dependency_source"])
    readiness["ready"] = True
    return readiness


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
