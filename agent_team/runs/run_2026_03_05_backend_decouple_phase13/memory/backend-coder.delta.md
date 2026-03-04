# Memory Delta: backend-coder

- Run ID: run_2026_03_05_backend_decouple_phase13
- Source Agent: backend-coder
- Recorder: backend-coder

## 1) Delta Summary
- What changed in this run:
  - Added `dependency_source` observability to `physics_backend_readiness`.
  - Extended candidate runtime classification with `external-runtime`.
  - Added reusable real-switch execution entry helper `resolve_real_backend_switch_execution_target(...)`.
  - Updated backend test helpers and assertions to validate synthetic/local/external dependency semantics.

## 2) New Stable Preferences
- Keep runtime policy semantics orthogonal: strict true-runtime should continue to block only synthetic mode unless explicitly redesigned.
- Prefer additive diagnostic metadata fields over changing existing keys/contract.

## 3) Pitfalls Learned
- External runtime readiness and runtime mode are related but not equivalent to full backend integration success.
- Test helper logic should prefer reusable diagnostics API instead of duplicating candidate/skip policy in each test file.

## 4) Reusable Patterns
- Attach backend readiness provenance metadata on backend instance, then surface it from generic readiness wrappers.
- Provide execution-target resolver APIs that wrap diagnostics report and keep backward-compatible report payload.

## 5) Evidence Links
- `mbrl/environments/backends/physics.py`
- `mbrl/environments/backends/diagnostics.py`
- `mbrl/environments/backends/__init__.py`
- `tests/test_backend_diagnostics.py`
- `tests/test_environment_backends.py`
- `tests/test_genesis_external_runtime_bridge.py`
