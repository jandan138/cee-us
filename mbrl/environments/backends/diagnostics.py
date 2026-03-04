import argparse
import json
import os

from .physics import list_physics_backends, physics_backend_readiness
from .registry import ENV_REGISTRY

REAL_SWITCH_TEST_ENV_VAR = "ENABLE_REAL_BACKEND_TESTS"


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


def collect_physics_backend_readiness(backend_names=None, *, options_by_backend=None):
    options_by_backend = options_by_backend or {}
    readiness = {}
    for backend_name in _normalize_backend_names(backend_names):
        readiness[backend_name] = physics_backend_readiness(
            backend_name,
            options=options_by_backend.get(backend_name),
        )
    return readiness


def diagnose_real_backend_switch_test(*, enable_real_backend_tests=None, options_by_backend=None):
    options_by_backend = options_by_backend or {}
    env_flag_raw = (
        os.environ.get(REAL_SWITCH_TEST_ENV_VAR, "0")
        if enable_real_backend_tests is None
        else str(enable_real_backend_tests)
    )
    env_flag_enabled = env_flag_raw == "1"

    implemented_non_mujoco = [
        name for name, info in list_physics_backends().items() if info.get("implemented", False) and name != "mujoco"
    ]

    candidate = None
    for env_name, env_backends in ENV_REGISTRY.items():
        for backend_name in implemented_non_mujoco:
            if backend_name in env_backends:
                candidate = {
                    "env_name": env_name,
                    "backend_name": backend_name,
                }
                break
        if candidate is not None:
            break

    if not env_flag_enabled:
        first_skip_reason = "Set ENABLE_REAL_BACKEND_TESTS=1 to run real non-MuJoCo backend switch tests."
    elif not implemented_non_mujoco:
        first_skip_reason = "No implemented non-MuJoCo physics backend is registered yet."
    elif candidate is None:
        first_skip_reason = "No ENV_REGISTRY entry exists for implemented non-MuJoCo backends."
    else:
        first_skip_reason = ""

    implemented_readiness = collect_physics_backend_readiness(
        implemented_non_mujoco,
        options_by_backend=options_by_backend,
    )
    candidate_readiness = None
    if candidate is not None:
        candidate_readiness = implemented_readiness.get(candidate["backend_name"])

    next_actions = []
    if not env_flag_enabled:
        next_actions.append("Export ENABLE_REAL_BACKEND_TESTS=1 when you want to run real backend switch tests.")
    if not implemented_non_mujoco:
        next_actions.append("Register at least one implemented non-MuJoCo physics backend.")
    if implemented_non_mujoco and candidate is None:
        next_actions.append("Add an ENV_REGISTRY mapping for at least one implemented non-MuJoCo backend.")
    if candidate_readiness is not None and not candidate_readiness.get("ready", False):
        next_actions.append(
            "Fix the candidate backend dependency/readiness issue before expecting a successful real switch run."
        )

    return {
        "env_flag": {
            "name": REAL_SWITCH_TEST_ENV_VAR,
            "value": env_flag_raw,
            "enabled": env_flag_enabled,
            "expected": "1",
        },
        "implemented_non_mujoco_backends": implemented_non_mujoco,
        "candidate": candidate,
        "candidate_readiness": candidate_readiness,
        "implemented_backend_readiness": implemented_readiness,
        "would_skip": bool(first_skip_reason),
        "first_skip_reason": first_skip_reason,
        "next_actions": next_actions,
    }


def collect_backend_diagnostics(backend_names=None, *, enable_real_backend_tests=None, options_by_backend=None):
    options_by_backend = options_by_backend or {}
    return {
        "physics_backend_readiness": collect_physics_backend_readiness(
            backend_names=backend_names,
            options_by_backend=options_by_backend,
        ),
        "real_backend_switch_test": diagnose_real_backend_switch_test(
            enable_real_backend_tests=enable_real_backend_tests,
            options_by_backend=options_by_backend,
        ),
    }


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
        candidate_readiness = real_switch.get("candidate_readiness")
        if candidate_readiness is not None:
            candidate_status = "READY" if candidate_readiness.get("ready") else "NOT READY"
            lines.append(f"- Candidate backend readiness: {candidate_status}")
            if not candidate_readiness.get("ready"):
                lines.append(
                    "  "
                    + f"Reason: {candidate_readiness.get('error_type')}: {candidate_readiness.get('reason')}"
                )

    if real_switch["would_skip"]:
        lines.append(f"- Result: test would be skipped ({real_switch['first_skip_reason']})")
    else:
        lines.append("- Result: skip preconditions satisfied (test should execute).")

    if real_switch["next_actions"]:
        lines.append("- Suggested next actions:")
        for action in real_switch["next_actions"]:
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
        "--skip-genesis-dependency-check",
        action="store_true",
        help="Probe Genesis readiness with skip_dependency_check=true (synthetic readiness).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print diagnostics as JSON.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    options_by_backend = {}
    if args.skip_genesis_dependency_check:
        options_by_backend["genesis"] = {"skip_dependency_check": True}

    report = collect_backend_diagnostics(
        backend_names=args.backends,
        options_by_backend=options_by_backend,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_text_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
