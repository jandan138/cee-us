import argparse
import json
import os

from .physics import list_physics_backends, physics_backend_readiness
from .plugins import load_backend_plugins, normalize_plugin_modules
from .registry import ENV_REGISTRY
from .render import render_backend_from_string

REAL_SWITCH_TEST_ENV_VAR = "ENABLE_REAL_BACKEND_TESTS"
REAL_SWITCH_TEST_PLUGIN_MODULES_ENV_VAR = "REAL_BACKEND_SWITCH_PLUGIN_MODULES"
REAL_SWITCH_TEST_PHYSICS_OPTIONS_ENV_VAR = "REAL_BACKEND_SWITCH_PHYSICS_OPTIONS_JSON"
REAL_SWITCH_TEST_REQUIRE_TRUE_RUNTIME_ENV_VAR = "REAL_BACKEND_SWITCH_REQUIRE_TRUE_RUNTIME"
REAL_SWITCH_TEST_GENESIS_EXTERNAL_PYTHON_ENV_VAR = "REAL_BACKEND_SWITCH_GENESIS_EXTERNAL_PYTHON"
REAL_SWITCH_TEST_GENESIS_PROBE_TIMEOUT_SEC_ENV_VAR = "REAL_BACKEND_SWITCH_GENESIS_PROBE_TIMEOUT_SEC"
REAL_SWITCH_TEST_GENESIS_PYOPENGL_PLATFORM_ENV_VAR = "REAL_BACKEND_SWITCH_GENESIS_PYOPENGL_PLATFORM"


def _normalize_backend_names(backend_names):
    if backend_names is None:
        return list(list_physics_backends().keys())
    if isinstance(backend_names, str):
        backend_names = [backend_names]

    normalized = []
    seen = set()
    for backend_name in backend_names:
        name = str(backend_name).strip().lower()
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return normalized


def _normalize_real_switch_plugin_modules(plugin_modules):
    if plugin_modules is None:
        return []
    if isinstance(plugin_modules, str):
        plugin_modules = plugin_modules.split(",")
    return normalize_plugin_modules(plugin_modules)


def _normalize_options_by_backend(options_by_backend):
    if options_by_backend is None:
        return {}
    if isinstance(options_by_backend, str):
        options_raw = options_by_backend.strip()
        if not options_raw:
            return {}
        try:
            options_by_backend = json.loads(options_raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                "REAL_BACKEND_SWITCH_PHYSICS_OPTIONS_JSON must be valid JSON object text."
            ) from error

    if not isinstance(options_by_backend, dict):
        raise TypeError("options_by_backend must be a dict or a JSON object string")

    normalized = {}
    for backend_name, backend_options in options_by_backend.items():
        normalized_name = str(backend_name).strip().lower()
        if not normalized_name:
            continue
        if backend_options is None:
            normalized[normalized_name] = {}
            continue
        if not isinstance(backend_options, dict):
            raise TypeError(f"options_by_backend['{normalized_name}'] must be a dict")
        normalized[normalized_name] = dict(backend_options)
    return normalized


def _normalize_bool_flag(flag_value, *, option_name):
    if isinstance(flag_value, bool):
        return flag_value
    normalized = str(flag_value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{option_name} must be one of: 1/0, true/false, yes/no, on/off.")


def _normalize_optional_timeout_seconds(timeout_value, *, option_name):
    if timeout_value is None:
        return None
    if isinstance(timeout_value, (int, float)):
        normalized = float(timeout_value)
    else:
        raw = str(timeout_value).strip()
        if not raw:
            return None
        try:
            normalized = float(raw)
        except ValueError as error:
            raise ValueError(f"{option_name} must be a positive number.") from error

    if normalized <= 0:
        raise ValueError(f"{option_name} must be > 0.")
    return normalized


def _merge_genesis_external_probe_options(
    options_by_backend,
    *,
    external_python=None,
    probe_timeout_seconds=None,
    pyopengl_platform=None,
):
    resolved_external_python = None if external_python is None else str(external_python).strip()
    if resolved_external_python == "":
        resolved_external_python = None
    resolved_pyopengl_platform = None if pyopengl_platform is None else str(pyopengl_platform).strip()
    if resolved_pyopengl_platform == "":
        resolved_pyopengl_platform = None
    resolved_timeout = _normalize_optional_timeout_seconds(
        probe_timeout_seconds,
        option_name=REAL_SWITCH_TEST_GENESIS_PROBE_TIMEOUT_SEC_ENV_VAR,
    )

    if resolved_external_python is None and resolved_timeout is None and resolved_pyopengl_platform is None:
        return options_by_backend

    merged = dict(options_by_backend or {})
    genesis_options = dict(merged.get("genesis", {}))

    if resolved_external_python is not None and "external_python" not in genesis_options:
        genesis_options["external_python"] = resolved_external_python
    if resolved_timeout is not None and "external_probe_timeout_sec" not in genesis_options:
        genesis_options["external_probe_timeout_sec"] = resolved_timeout
    if (
        resolved_pyopengl_platform is not None
        and "PYOPENGL_PLATFORM" not in genesis_options
        and "pyopengl_platform" not in genesis_options
    ):
        genesis_options["PYOPENGL_PLATFORM"] = resolved_pyopengl_platform

    if genesis_options:
        merged["genesis"] = genesis_options
    return merged


def _classify_runtime_mode(candidate_options):
    options = candidate_options or {}
    if bool(options.get("skip_dependency_check", False)):
        return "synthetic-runtime", "skip_dependency_check=true"
    return "true-runtime", "no synthetic-only options enabled"


def resolve_real_backend_switch_configuration(
    *,
    backend_plugin_modules=None,
    options_by_backend=None,
    require_true_runtime=None,
    genesis_external_python=None,
    genesis_probe_timeout_sec=None,
    genesis_pyopengl_platform=None,
):
    configured_plugin_modules = (
        backend_plugin_modules
        if backend_plugin_modules is not None
        else os.environ.get(REAL_SWITCH_TEST_PLUGIN_MODULES_ENV_VAR, "")
    )
    configured_options = (
        options_by_backend
        if options_by_backend is not None
        else os.environ.get(REAL_SWITCH_TEST_PHYSICS_OPTIONS_ENV_VAR, "")
    )
    configured_require_true_runtime = (
        require_true_runtime
        if require_true_runtime is not None
        else os.environ.get(REAL_SWITCH_TEST_REQUIRE_TRUE_RUNTIME_ENV_VAR, "0")
    )
    configured_genesis_external_python = (
        genesis_external_python
        if genesis_external_python is not None
        else os.environ.get(REAL_SWITCH_TEST_GENESIS_EXTERNAL_PYTHON_ENV_VAR, "")
    )
    configured_genesis_probe_timeout_sec = (
        genesis_probe_timeout_sec
        if genesis_probe_timeout_sec is not None
        else os.environ.get(REAL_SWITCH_TEST_GENESIS_PROBE_TIMEOUT_SEC_ENV_VAR, "")
    )
    configured_genesis_pyopengl_platform = (
        genesis_pyopengl_platform
        if genesis_pyopengl_platform is not None
        else os.environ.get(REAL_SWITCH_TEST_GENESIS_PYOPENGL_PLATFORM_ENV_VAR, "")
    )
    resolved_options_by_backend = _normalize_options_by_backend(configured_options)
    resolved_options_by_backend = _merge_genesis_external_probe_options(
        resolved_options_by_backend,
        external_python=configured_genesis_external_python,
        probe_timeout_seconds=configured_genesis_probe_timeout_sec,
        pyopengl_platform=configured_genesis_pyopengl_platform,
    )
    return {
        "backend_plugin_modules": _normalize_real_switch_plugin_modules(configured_plugin_modules),
        "options_by_backend": resolved_options_by_backend,
        "require_true_runtime": _normalize_bool_flag(
            configured_require_true_runtime,
            option_name=REAL_SWITCH_TEST_REQUIRE_TRUE_RUNTIME_ENV_VAR,
        ),
    }


def collect_physics_backend_readiness(backend_names=None, *, options_by_backend=None):
    options_by_backend = _normalize_options_by_backend(options_by_backend)
    readiness = {}
    for backend_name in _normalize_backend_names(backend_names):
        readiness[backend_name] = physics_backend_readiness(
            backend_name,
            options=options_by_backend.get(backend_name),
        )
    return readiness


def _format_readiness_failure(readiness):
    backend_name = readiness.get("backend", "<unknown>")
    error_type = readiness.get("error_type") or "UnknownError"
    reason = readiness.get("reason") or "unknown reason"
    return f"{backend_name}: {error_type}: {reason}"


def _select_first_ready_candidate(implemented_non_mujoco, implemented_readiness):
    candidate = None
    ready_backends_without_mapping = []

    for backend_name in implemented_non_mujoco:
        readiness = implemented_readiness.get(backend_name, {})
        if not readiness.get("ready", False):
            continue
        mapped_env_name = None
        for env_name, env_backends in ENV_REGISTRY.items():
            if backend_name in env_backends:
                mapped_env_name = env_name
                break
        if mapped_env_name is None:
            ready_backends_without_mapping.append(backend_name)
            continue

        candidate = {
            "env_name": mapped_env_name,
            "backend_name": backend_name,
        }
        break

    return candidate, ready_backends_without_mapping


def render_backend_readiness(backend_name):
    readiness = {
        "backend": backend_name,
        "ready": False,
        "implemented": False,
        "error_type": None,
        "reason": "",
    }
    try:
        backend = render_backend_from_string(backend_name)
    except Exception as error:
        readiness["error_type"] = type(error).__name__
        readiness["reason"] = str(error)
        return readiness

    readiness["backend"] = backend.backend_name
    readiness["implemented"] = bool(getattr(backend, "implemented", False))
    if not readiness["implemented"]:
        readiness["error_type"] = "NotImplementedError"
        readiness["reason"] = (
            f"Render backend '{backend.display_name or backend.backend_name}' is registered but not implemented."
        )
        return readiness

    readiness["ready"] = True
    return readiness


def diagnose_unified_switch_readiness(
    env_name,
    *,
    physics_backend_name="mujoco",
    render_backend_name="native",
    physics_backend_options=None,
):
    physics_backend_name = physics_backend_name or "mujoco"
    render_backend_name = render_backend_name or "native"

    physics_readiness = physics_backend_readiness(
        physics_backend_name,
        options=physics_backend_options or {},
    )
    resolved_physics_backend = physics_readiness.get("backend") or physics_backend_name
    render_readiness = render_backend_readiness(render_backend_name)

    env_mapping = {
        "env_name": env_name,
        "physics_backend": resolved_physics_backend,
        "exists": False,
        "error_type": None,
        "reason": "",
        "available_backends": [],
        "target": None,
    }
    if not env_name:
        env_mapping["error_type"] = "ValueError"
        env_mapping["reason"] = "env_name must be non-empty for unified switch readiness diagnostics."
    elif env_name not in ENV_REGISTRY:
        env_mapping["error_type"] = "KeyError"
        env_mapping["reason"] = f"Env '{env_name}' is not registered in ENV_REGISTRY."
    else:
        backend_entries = ENV_REGISTRY[env_name]
        env_mapping["available_backends"] = sorted(backend_entries.keys())
        if resolved_physics_backend not in backend_entries:
            env_mapping["error_type"] = "NotImplementedError"
            env_mapping["reason"] = (
                f"Env '{env_name}' has no mapping for physics backend '{resolved_physics_backend}'."
            )
        else:
            module_path, class_name = backend_entries[resolved_physics_backend]
            env_mapping["exists"] = True
            env_mapping["target"] = {
                "module_path": module_path,
                "class_name": class_name,
            }

    overall_ready = (
        bool(physics_readiness.get("ready", False))
        and bool(render_readiness.get("ready", False))
        and bool(env_mapping.get("exists", False))
    )
    next_actions = []
    if not physics_readiness.get("ready", False):
        next_actions.append(
            "Fix physics backend readiness first (dependency or implementation), then retry tuple diagnostics."
        )
    if not render_readiness.get("ready", False):
        next_actions.append(
            "Use an implemented render backend (e.g. native/headless/none) or register a concrete render plugin."
        )
    if not env_mapping.get("exists", False):
        next_actions.append("Add or register ENV_REGISTRY mapping for the selected env and physics backend.")

    return {
        "requested": {
            "env_name": env_name,
            "physics_backend": physics_backend_name,
            "render_backend": render_backend_name,
        },
        "resolved": {
            "physics_backend": resolved_physics_backend,
            "render_backend": render_readiness.get("backend"),
        },
        "physics_backend_readiness": physics_readiness,
        "render_backend_readiness": render_readiness,
        "env_mapping": env_mapping,
        "ready": overall_ready,
        "next_actions": next_actions,
    }


def diagnose_real_backend_switch_test(
    *,
    enable_real_backend_tests=None,
    options_by_backend=None,
    backend_plugin_modules=None,
    require_true_runtime=None,
    genesis_external_python=None,
    genesis_probe_timeout_sec=None,
    genesis_pyopengl_platform=None,
):
    config = resolve_real_backend_switch_configuration(
        backend_plugin_modules=backend_plugin_modules,
        options_by_backend=options_by_backend,
        require_true_runtime=require_true_runtime,
        genesis_external_python=genesis_external_python,
        genesis_probe_timeout_sec=genesis_probe_timeout_sec,
        genesis_pyopengl_platform=genesis_pyopengl_platform,
    )
    configured_plugin_modules = config["backend_plugin_modules"]
    resolved_options_by_backend = config["options_by_backend"]
    require_true_runtime_enabled = config["require_true_runtime"]
    require_true_runtime_raw = (
        os.environ.get(REAL_SWITCH_TEST_REQUIRE_TRUE_RUNTIME_ENV_VAR, "0")
        if require_true_runtime is None
        else str(require_true_runtime)
    )
    require_true_runtime_source = "env" if require_true_runtime is None else "api"
    loaded_plugin_modules = load_backend_plugins(configured_plugin_modules)

    env_flag_raw = (
        os.environ.get(REAL_SWITCH_TEST_ENV_VAR, "0")
        if enable_real_backend_tests is None
        else str(enable_real_backend_tests)
    )
    env_flag_enabled = env_flag_raw == "1"

    implemented_non_mujoco = [
        name for name, info in list_physics_backends().items() if info.get("implemented", False) and name != "mujoco"
    ]

    implemented_readiness = collect_physics_backend_readiness(
        implemented_non_mujoco,
        options_by_backend=resolved_options_by_backend,
    )

    candidate, ready_backends_without_mapping = _select_first_ready_candidate(
        implemented_non_mujoco,
        implemented_readiness,
    )

    readiness_failures = []
    for backend_name in implemented_non_mujoco:
        readiness = implemented_readiness.get(backend_name, {})
        if not readiness.get("ready", False):
            readiness_failures.append(_format_readiness_failure(readiness))

    candidate_readiness = None
    candidate_physics_backend_options = {}
    candidate_runtime_mode = None
    candidate_runtime_mode_reason = ""
    if candidate is not None:
        candidate_readiness = implemented_readiness.get(candidate["backend_name"])
        candidate_physics_backend_options = resolved_options_by_backend.get(candidate["backend_name"], {})
        candidate_runtime_mode, candidate_runtime_mode_reason = _classify_runtime_mode(
            candidate_physics_backend_options
        )

    strict_runtime_violated = bool(
        require_true_runtime_enabled and candidate is not None and candidate_runtime_mode == "synthetic-runtime"
    )

    if not env_flag_enabled:
        first_skip_reason = "Set ENABLE_REAL_BACKEND_TESTS=1 to run real non-MuJoCo backend switch tests."
    elif not implemented_non_mujoco:
        first_skip_reason = "No implemented non-MuJoCo physics backend is registered yet."
    elif candidate is None:
        detail_parts = []
        if readiness_failures:
            detail_parts.append(f"readiness failures: {'; '.join(readiness_failures)}")
        if ready_backends_without_mapping:
            detail_parts.append(
                "ready backends missing ENV_REGISTRY mapping: " + ", ".join(sorted(ready_backends_without_mapping))
            )
        detail_suffix = f" Details: {' | '.join(detail_parts)}" if detail_parts else ""
        first_skip_reason = (
            "No ready non-MuJoCo backend candidate could be selected for real backend switch test." + detail_suffix
        )
    elif strict_runtime_violated:
        first_skip_reason = (
            "Strict true-runtime policy is enabled, but candidate "
            f"{candidate['env_name']} / {candidate['backend_name']} is synthetic-runtime "
            f"({candidate_runtime_mode_reason})."
        )
    else:
        first_skip_reason = ""

    next_actions = []
    if not env_flag_enabled:
        next_actions.append("Export ENABLE_REAL_BACKEND_TESTS=1 when you want to run real backend switch tests.")
    if not require_true_runtime_enabled:
        next_actions.append(
            "Set REAL_BACKEND_SWITCH_REQUIRE_TRUE_RUNTIME=1 (or pass --real-switch-require-true-runtime) "
            "to block synthetic-runtime candidates."
        )
    if configured_plugin_modules:
        next_actions.append(
            "Keep real-switch plugin module registration deterministic; ensure modules are importable in test env."
        )
    if not implemented_non_mujoco:
        next_actions.append("Register at least one implemented non-MuJoCo physics backend.")
    if candidate is None and ready_backends_without_mapping:
        next_actions.append("Add an ENV_REGISTRY mapping for at least one implemented non-MuJoCo backend.")
    if readiness_failures:
        next_actions.append(
            "Fix the candidate backend dependency/readiness issue before expecting a successful real switch run."
        )
    if strict_runtime_violated:
        next_actions.append(
            "Use true runtime dependencies/options (e.g., disable skip_dependency_check) for strict mode runs."
        )

    return {
        "env_flag": {
            "name": REAL_SWITCH_TEST_ENV_VAR,
            "value": env_flag_raw,
            "enabled": env_flag_enabled,
            "expected": "1",
        },
        "plugin_modules_env_var": {
            "name": REAL_SWITCH_TEST_PLUGIN_MODULES_ENV_VAR,
            "value": os.environ.get(REAL_SWITCH_TEST_PLUGIN_MODULES_ENV_VAR, ""),
        },
        "physics_options_env_var": {
            "name": REAL_SWITCH_TEST_PHYSICS_OPTIONS_ENV_VAR,
            "value": os.environ.get(REAL_SWITCH_TEST_PHYSICS_OPTIONS_ENV_VAR, ""),
        },
        "genesis_external_python_env_var": {
            "name": REAL_SWITCH_TEST_GENESIS_EXTERNAL_PYTHON_ENV_VAR,
            "value": os.environ.get(REAL_SWITCH_TEST_GENESIS_EXTERNAL_PYTHON_ENV_VAR, ""),
        },
        "genesis_probe_timeout_sec_env_var": {
            "name": REAL_SWITCH_TEST_GENESIS_PROBE_TIMEOUT_SEC_ENV_VAR,
            "value": os.environ.get(REAL_SWITCH_TEST_GENESIS_PROBE_TIMEOUT_SEC_ENV_VAR, ""),
        },
        "genesis_pyopengl_platform_env_var": {
            "name": REAL_SWITCH_TEST_GENESIS_PYOPENGL_PLATFORM_ENV_VAR,
            "value": os.environ.get(REAL_SWITCH_TEST_GENESIS_PYOPENGL_PLATFORM_ENV_VAR, ""),
        },
        "require_true_runtime_flag": {
            "name": REAL_SWITCH_TEST_REQUIRE_TRUE_RUNTIME_ENV_VAR,
            "value": require_true_runtime_raw,
            "enabled": require_true_runtime_enabled,
            "expected": "1",
            "source": require_true_runtime_source,
            "violated": strict_runtime_violated,
        },
        "configured_backend_plugin_modules": configured_plugin_modules,
        "loaded_backend_plugin_modules": loaded_plugin_modules,
        "configured_physics_options_by_backend": resolved_options_by_backend,
        "implemented_non_mujoco_backends": implemented_non_mujoco,
        "candidate": candidate,
        "candidate_physics_backend_options": candidate_physics_backend_options,
        "candidate_runtime_mode": candidate_runtime_mode,
        "candidate_runtime_mode_reason": candidate_runtime_mode_reason,
        "candidate_readiness": candidate_readiness,
        "ready_backends_without_mapping": ready_backends_without_mapping,
        "readiness_failures": readiness_failures,
        "implemented_backend_readiness": implemented_readiness,
        "would_skip": bool(first_skip_reason),
        "first_skip_reason": first_skip_reason,
        "next_actions": next_actions,
    }


def collect_backend_diagnostics(
    backend_names=None,
    *,
    enable_real_backend_tests=None,
    options_by_backend=None,
    real_switch_backend_plugin_modules=None,
    real_switch_options_by_backend=None,
    real_switch_require_true_runtime=None,
    real_switch_genesis_external_python=None,
    real_switch_genesis_probe_timeout_sec=None,
    real_switch_genesis_pyopengl_platform=None,
    env_name=None,
    physics_backend_name=None,
    render_backend_name=None,
):
    options_by_backend = _normalize_options_by_backend(options_by_backend)
    resolved_real_switch_options = real_switch_options_by_backend
    if resolved_real_switch_options is None and options_by_backend:
        resolved_real_switch_options = options_by_backend
    report = {
        "physics_backend_readiness": collect_physics_backend_readiness(
            backend_names=backend_names,
            options_by_backend=options_by_backend,
        ),
        "real_backend_switch_test": diagnose_real_backend_switch_test(
            enable_real_backend_tests=enable_real_backend_tests,
            backend_plugin_modules=real_switch_backend_plugin_modules,
            options_by_backend=resolved_real_switch_options,
            require_true_runtime=real_switch_require_true_runtime,
            genesis_external_python=real_switch_genesis_external_python,
            genesis_probe_timeout_sec=real_switch_genesis_probe_timeout_sec,
            genesis_pyopengl_platform=real_switch_genesis_pyopengl_platform,
        ),
    }
    if env_name is not None:
        tuple_backend_name = physics_backend_name or "mujoco"
        report["unified_switch_readiness"] = diagnose_unified_switch_readiness(
            env_name,
            physics_backend_name=tuple_backend_name,
            render_backend_name=render_backend_name or "native",
            physics_backend_options=options_by_backend.get(tuple_backend_name),
        )
    return report


def _format_text_report(report):
    lines = []
    lines.append("Physics backend readiness:")
    readiness = report["physics_backend_readiness"]
    for requested_name, backend_result in readiness.items():
        resolved_backend = backend_result.get("backend")
        status = "READY" if backend_result.get("ready") else "NOT READY"
        line = f"- {requested_name} -> {resolved_backend}: {status}"
        if not backend_result.get("ready"):
            error_type = backend_result.get("error_type") or "Error"
            reason = backend_result.get("reason", "")
            line = f"{line} ({error_type}: {reason})"
        lines.append(line)

    real_switch = report["real_backend_switch_test"]
    lines.append("")
    lines.append("Real backend switch test precheck:")
    env_flag = real_switch["env_flag"]
    flag_state = "PASS" if env_flag["enabled"] else "FAIL"
    lines.append(
        f"- Env flag {env_flag['name']}={env_flag['value']} (expected {env_flag['expected']}): {flag_state}"
    )
    strict_flag = real_switch["require_true_runtime_flag"]
    strict_state = "ENFORCED" if strict_flag["enabled"] else "disabled"
    lines.append(
        f"- Strict true-runtime policy {strict_flag['name']}={strict_flag['value']} "
        f"(expected {strict_flag['expected']}): {strict_state}"
    )
    lines.append(
        f"- Configured plugin modules: {real_switch['configured_backend_plugin_modules'] or 'none'}"
    )
    lines.append(
        "- Genesis external runtime bridge env: "
        + f"{real_switch['genesis_external_python_env_var']['name']}="
        + f"{real_switch['genesis_external_python_env_var']['value'] or '<unset>'}, "
        + f"{real_switch['genesis_probe_timeout_sec_env_var']['name']}="
        + f"{real_switch['genesis_probe_timeout_sec_env_var']['value'] or '<unset>'}, "
        + f"{real_switch['genesis_pyopengl_platform_env_var']['name']}="
        + f"{real_switch['genesis_pyopengl_platform_env_var']['value'] or '<unset>'}"
    )
    lines.append(
        f"- Loaded plugin modules: {real_switch['loaded_backend_plugin_modules'] or 'none'}"
    )
    lines.append(
        f"- Implemented non-MuJoCo backends: {real_switch['implemented_non_mujoco_backends'] or 'none'}"
    )

    candidate = real_switch["candidate"]
    if candidate is None:
        lines.append("- Candidate env/backend mapping: none")
    else:
        lines.append(
            f"- Candidate env/backend mapping: {candidate['env_name']} / {candidate['backend_name']}"
        )
        lines.append(
            f"- Candidate runtime mode: {real_switch.get('candidate_runtime_mode')} "
            f"({real_switch.get('candidate_runtime_mode_reason')})"
        )
        candidate_readiness = real_switch.get("candidate_readiness")
        if candidate_readiness is not None:
            candidate_status = "READY" if candidate_readiness.get("ready") else "NOT READY"
            lines.append(f"- Candidate backend readiness: {candidate_status}")
            if not candidate_readiness.get("ready"):
                lines.append(
                    "  "
                    + f"Reason: {candidate_readiness.get('error_type')}: {candidate_readiness.get('reason')}"
                )
        candidate_options = real_switch.get("candidate_physics_backend_options", {})
        lines.append(f"- Candidate physics backend options: {candidate_options or '{}'}")

    if real_switch["would_skip"]:
        lines.append(f"- Result: test would be skipped ({real_switch['first_skip_reason']})")
    else:
        lines.append("- Result: skip preconditions satisfied (test should execute).")

    if real_switch["next_actions"]:
        lines.append("- Suggested next actions:")
        for action in real_switch["next_actions"]:
            lines.append(f"  - {action}")

    unified = report.get("unified_switch_readiness")
    if unified is not None:
        lines.append("")
        lines.append("Unified switch readiness (env/backend/render):")
        requested = unified["requested"]
        resolved = unified["resolved"]
        lines.append(
            "- Requested tuple: "
            + f"{requested['env_name']} / {requested['physics_backend']} / {requested['render_backend']}"
        )
        lines.append(
            "- Resolved tuple: "
            + f"{requested['env_name']} / {resolved['physics_backend']} / {resolved['render_backend']}"
        )

        tuple_physics = unified["physics_backend_readiness"]
        tuple_physics_status = "READY" if tuple_physics.get("ready") else "NOT READY"
        physics_line = f"- Physics backend readiness: {tuple_physics_status}"
        if not tuple_physics.get("ready"):
            physics_line += f" ({tuple_physics.get('error_type')}: {tuple_physics.get('reason')})"
        lines.append(physics_line)

        tuple_render = unified["render_backend_readiness"]
        tuple_render_status = "READY" if tuple_render.get("ready") else "NOT READY"
        render_line = (
            "- Render backend readiness: "
            + f"{tuple_render_status} (implemented={tuple_render.get('implemented', False)})"
        )
        if not tuple_render.get("ready"):
            render_line += f" ({tuple_render.get('error_type')}: {tuple_render.get('reason')})"
        lines.append(render_line)

        env_mapping = unified["env_mapping"]
        mapping_state = "FOUND" if env_mapping.get("exists") else "MISSING"
        mapping_line = f"- Env mapping status: {mapping_state}"
        if env_mapping.get("exists"):
            target = env_mapping.get("target") or {}
            mapping_line += f" ({target.get('module_path')}, {target.get('class_name')})"
        else:
            mapping_line += f" ({env_mapping.get('error_type')}: {env_mapping.get('reason')})"
        lines.append(mapping_line)
        lines.append(f"- Overall tuple readiness: {'READY' if unified.get('ready') else 'NOT READY'}")

        if unified["next_actions"]:
            lines.append("- Tuple suggested next actions:")
            for action in unified["next_actions"]:
                lines.append(f"  - {action}")
    return "\n".join(lines)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Backend diagnostics for readiness and real switch test skip reasons."
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        default=None,
        help="Backend name to probe (repeatable). Default: all registered physics backends.",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Env name for unified tuple readiness diagnostics.",
    )
    parser.add_argument(
        "--physics-backend",
        default=None,
        help="Physics backend for unified tuple readiness diagnostics. Default: mujoco.",
    )
    parser.add_argument(
        "--render-backend",
        default=None,
        help="Render backend for unified tuple readiness diagnostics. Default: native.",
    )
    parser.add_argument(
        "--skip-genesis-dependency-check",
        action="store_true",
        help="Probe Genesis readiness with skip_dependency_check=true (synthetic readiness).",
    )
    parser.add_argument(
        "--real-switch-plugin-module",
        action="append",
        dest="real_switch_plugin_modules",
        default=None,
        help=(
            "Plugin module to load before real-switch candidate discovery "
            "(repeatable, same behavior as REAL_BACKEND_SWITCH_PLUGIN_MODULES)."
        ),
    )
    parser.add_argument(
        "--real-switch-physics-options-json",
        default=None,
        help=(
            "JSON object for real-switch readiness/env options by backend, "
            "e.g. '{\"genesis\": {\"skip_dependency_check\": true}}'."
        ),
    )
    parser.add_argument(
        "--real-switch-require-true-runtime",
        action="store_true",
        default=None,
        help=(
            "Require candidate runtime mode to be true-runtime. "
            "Equivalent to REAL_BACKEND_SWITCH_REQUIRE_TRUE_RUNTIME=1."
        ),
    )
    parser.add_argument(
        "--real-switch-genesis-external-python",
        default=None,
        help=(
            "External Python executable for Genesis readiness probe when local runtime cannot import genesis. "
            f"Equivalent to {REAL_SWITCH_TEST_GENESIS_EXTERNAL_PYTHON_ENV_VAR}."
        ),
    )
    parser.add_argument(
        "--real-switch-genesis-probe-timeout-sec",
        type=float,
        default=None,
        help=(
            "Timeout seconds for Genesis external readiness probe. "
            f"Equivalent to {REAL_SWITCH_TEST_GENESIS_PROBE_TIMEOUT_SEC_ENV_VAR}."
        ),
    )
    parser.add_argument(
        "--real-switch-genesis-pyopengl-platform",
        default=None,
        help=(
            "Value injected as PYOPENGL_PLATFORM for Genesis external readiness probe. "
            f"Equivalent to {REAL_SWITCH_TEST_GENESIS_PYOPENGL_PLATFORM_ENV_VAR}."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print diagnostics as JSON.",
    )
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = _parse_args(argv)

    options_by_backend = {}
    if args.skip_genesis_dependency_check:
        options_by_backend["genesis"] = {"skip_dependency_check": True}

    real_switch_options_by_backend = None
    if args.real_switch_physics_options_json is not None:
        real_switch_options_by_backend = _normalize_options_by_backend(args.real_switch_physics_options_json)
    elif options_by_backend:
        real_switch_options_by_backend = options_by_backend

    tuple_backend_name = args.physics_backend
    if args.env is not None:
        normalized_backends = _normalize_backend_names(args.backends)
        if tuple_backend_name is None and len(normalized_backends) == 1:
            tuple_backend_name = normalized_backends[0]
        if tuple_backend_name is None and len(normalized_backends) > 1:
            parser.error("When --env is set with multiple --backend values, please set --physics-backend explicitly.")
        if tuple_backend_name is None:
            tuple_backend_name = "mujoco"

    report = collect_backend_diagnostics(
        backend_names=args.backends,
        options_by_backend=options_by_backend,
        real_switch_backend_plugin_modules=args.real_switch_plugin_modules,
        real_switch_options_by_backend=real_switch_options_by_backend,
        real_switch_require_true_runtime=args.real_switch_require_true_runtime,
        real_switch_genesis_external_python=args.real_switch_genesis_external_python,
        real_switch_genesis_probe_timeout_sec=args.real_switch_genesis_probe_timeout_sec,
        real_switch_genesis_pyopengl_platform=args.real_switch_genesis_pyopengl_platform,
        env_name=args.env,
        physics_backend_name=tuple_backend_name,
        render_backend_name=args.render_backend,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_text_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
