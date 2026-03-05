# Agent Memory: backend-coder

## Stable Preferences
- 

## Known Pitfalls
- 

## Proven Patterns
- 

## Anti-Patterns
- 

## Tooling Notes
- 

## Recent Deltas
- Run: none
  - Summary: 
  - Source: 



- Run: run_2026_03_04_backend_decouple_phase2
  - Summary: merged delta at 2026-03-04T15:09:56Z
  - Source: runs/run_2026_03_04_backend_decouple_phase2/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase2
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - What changed in this run:
    
    ## 2) New Stable Preferences
    - 
    
    ## 3) Pitfalls Learned
    - 
    
    ## 4) Reusable Patterns
    - 
    
    ## 5) Evidence Links
    - 

- Run: run_2026_03_04_backend_decouple_phase2
  - Summary: merged delta at 2026-03-04T15:11:00Z
  - Source: runs/run_2026_03_04_backend_decouple_phase2/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase2
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - What changed in this run:
    
    ## 2) New Stable Preferences
    - 
    
    ## 3) Pitfalls Learned
    - 
    
    ## 4) Reusable Patterns
    - 
    
    ## 5) Evidence Links
    - 

- Run: run_2026_03_04_backend_decouple_phase3
  - Summary: merged delta at 2026-03-04T15:27:19Z
  - Source: runs/run_2026_03_04_backend_decouple_phase3/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase3
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - What changed in this run:
    
    ## 2) New Stable Preferences
    - 
    
    ## 3) Pitfalls Learned
    - 
    
    ## 4) Reusable Patterns
    - 
    
    ## 5) Evidence Links
    - 

- Run: run_2026_03_04_backend_decouple_phase4
  - Summary: merged delta at 2026-03-04T15:33:07Z
  - Source: runs/run_2026_03_04_backend_decouple_phase4/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase4
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - What changed in this run:
    
    ## 2) New Stable Preferences
    - 
    
    ## 3) Pitfalls Learned
    - 
    
    ## 4) Reusable Patterns
    - 
    
    ## 5) Evidence Links
    - 

- Run: run_2026_03_04_backend_decouple_phase5
  - Summary: merged delta at 2026-03-04T15:40:48Z
  - Source: runs/run_2026_03_04_backend_decouple_phase5/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase5
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - What changed in this run:
    
    ## 2) New Stable Preferences
    - 
    
    ## 3) Pitfalls Learned
    - 
    
    ## 4) Reusable Patterns
    - 
    
    ## 5) Evidence Links
    - 

- Run: run_2026_03_04_backend_decouple_phase5
  - Summary: merged delta at 2026-03-04T15:44:36Z
  - Source: runs/run_2026_03_04_backend_decouple_phase5/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase5
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - What changed in this run:
    
    ## 2) New Stable Preferences
    - 
    
    ## 3) Pitfalls Learned
    - 
    
    ## 4) Reusable Patterns
    - 
    
    ## 5) Evidence Links
    - 

- Run: run_2026_03_04_backend_decouple_phase6
  - Summary: merged delta at 2026-03-04T15:48:33Z
  - Source: runs/run_2026_03_04_backend_decouple_phase6/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase6
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - What changed in this run:
    
    ## 2) New Stable Preferences
    - 
    
    ## 3) Pitfalls Learned
    - 
    
    ## 4) Reusable Patterns
    - 
    
    ## 5) Evidence Links
    - 

- Run: run_2026_03_04_backend_decouple_phase7
  - Summary: merged delta at 2026-03-04T15:58:52Z
  - Source: runs/run_2026_03_04_backend_decouple_phase7/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_04_backend_decouple_phase7
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - Added a backend diagnostics API/CLI that explicitly reports physics readiness and real backend switch test skip reasons.
    
    ## 2) New Stable Preferences
    - For optional backends (e.g., Genesis), provide machine-readable diagnostics before attempting runtime switching.
    
    ## 3) Pitfalls Learned
    - Sub-agent logs must follow repository-required section headers exactly; otherwise `check_run_logs.sh` fails.
    
    ## 4) Reusable Patterns
    - Mirror test gate order in diagnostics output so operator feedback and test behavior stay consistent.
    - Use lazy wrappers in package `__init__.py` for module entrypoints executed via `python -m`.
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase7/logs/backend-coder.md`
    - Subagent commit `3469dc34541b1ccdf070a1403eb2adaa354a545f`

- Run: run_2026_03_05_backend_decouple_phase8
  - Summary: merged delta at 2026-03-04T16:10:02Z
  - Source: runs/run_2026_03_05_backend_decouple_phase8/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_05_backend_decouple_phase8
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - Added unified tuple-level readiness diagnostics (`env + physics + render`) and CLI parameters to query this tuple directly.
    
    ## 2) New Stable Preferences
    - Extend diagnostics APIs additively to keep prior phase JSON keys backward compatible.
    
    ## 3) Pitfalls Learned
    - In this workspace, `pytest` may be absent; keep fallback validation path using `python3 -m unittest` and CLI smoke checks.
    
    ## 4) Reusable Patterns
    - Tuple readiness schema pattern:
      - requested tuple
      - resolved tuple
      - per-dimension readiness (physics/render/mapping)
      - overall `ready`
      - actionable `next_actions`
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_05_backend_decouple_phase8/logs/backend-coder.md`
    - subagent commit `95e4a4a14023c91a75889973920e4872aeff5ee0`

- Run: run_2026_03_05_backend_decouple_phase9
  - Summary: merged delta at 2026-03-04T16:31:37Z
  - Source: runs/run_2026_03_05_backend_decouple_phase9/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_05_backend_decouple_phase9
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - Added a shared phase9 configuration bridge so real backend switch tests can discover plugin-provided candidates and reuse backend options for both readiness probing and actual env construction.
    
    ## 2) New Stable Preferences
    - For gated test configurability, prefer one canonical resolver (`resolve_real_backend_switch_configuration`) consumed by both diagnostics and tests to avoid drift.
    
    ## 3) Pitfalls Learned
    - Keeping candidate discovery and candidate readiness separate can produce false-positive "would run" signals; candidate selection should require readiness when used by execution-gated tests.
    
    ## 4) Reusable Patterns
    - Environment-variable + explicit-input bridge pattern:
      - parse/normalize values centrally,
      - keep unset behavior no-op,
      - expose resolved config in diagnostics report,
      - feed same resolved options into runtime construction calls.
    
    ## 5) Evidence Links
    - `/tmp/agent_team_worktrees/cee-us/run_2026_03_05_backend_decouple_phase9/backend-coder/mbrl/environments/backends/diagnostics.py`
    - `/tmp/agent_team_worktrees/cee-us/run_2026_03_05_backend_decouple_phase9/backend-coder/tests/test_environment_backends.py`
    - commit `404a8114c1223ea2cfa8a8cbf4f49ad03a08e264`

- Run: run_2026_03_05_backend_decouple_phase10
  - Summary: merged delta at 2026-03-04T16:45:32Z
  - Source: runs/run_2026_03_05_backend_decouple_phase10/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_05_backend_decouple_phase10
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - Extended real-switch diagnostics with runtime-mode labeling and opt-in strict true-runtime policy (env/API/CLI).
    - Reused existing `would_skip/first_skip_reason` contract to gate synthetic candidates with explicit reasons.
    - Updated gated helper tests and added phase10 change note.
    
    ## 2) New Stable Preferences
    - Prefer adding strict-mode controls as additive flags with default-off behavior to preserve existing pipelines.
    - For diagnostics feature flags, expose both effective boolean and raw/source metadata in report payloads.
    
    ## 3) Pitfalls Learned
    - Synthetic readiness pathways can silently satisfy candidate selection; strict policy must be checked after candidate resolution, not before.
    - Tests that rely on synthetic options should explicitly set strict flag to `0` in env patches to avoid external env leakage.
    
    ## 4) Reusable Patterns
    - Centralize env/API normalization in a single resolver (`resolve_real_backend_switch_configuration`) and fan out to CLI/report/test helper.
    - Keep gating behavior inside shared diagnostics API, then make helper methods thin passthrough wrappers.
    
    ## 5) Evidence Links
    - Commit: `6f31c0a`
    - Validation: `python3 -m unittest tests.test_backend_diagnostics tests.test_environment_backends -v`
    - Docs: `zh_docs/changes/2026-03-05_real-backend-switch-runtime-policy-phase10.md`

- Run: run_2026_03_05_backend_decouple_phase12
  - Summary: merged delta at 2026-03-04T17:11:19Z
  - Source: runs/run_2026_03_05_backend_decouple_phase12/memory/backend-coder.delta.md
  - Notes:
    # Memory Delta: backend-coder
    
    - Run ID: run_2026_03_05_backend_decouple_phase12
    - Source Agent: backend-coder
    - Recorder: backend-coder
    
    ## 1) Delta Summary
    - What changed in this run:
      - Implemented optional Genesis external runtime readiness probe in physics backend.
      - Bridged Genesis external probe env-vars into diagnostics real-switch config resolution.
      - Added tests for probe behavior/config parsing/CLI pass-through and strict-mode compatibility.
      - Added phase12 documentation with `.venv-genesis` command examples.
    
    ## 2) New Stable Preferences
    - Prefer keeping strict runtime policy logic isolated (`skip_dependency_check` remains sole synthetic marker) when adding new readiness paths.
    - For env-var bridges, merge into backend options with "explicit option wins over env default" semantics.
    
    ## 3) Pitfalls Learned
    - `python` alias is not guaranteed in this environment; use `python3` for unit tests.
    - `init_run.sh` creates template run records, but `check_run_logs.sh` still requires populated worktree registry from `setup_run_worktrees.sh`.
    
    ## 4) Reusable Patterns
    - Add external dependency probes as optional fallback in backend `prepare_backend`, not as unconditional behavior change.
    - Reuse `resolve_real_backend_switch_configuration(...)` as the single bridge point for env/API/CLI config normalization.
    
    ## 5) Evidence Links
    - mbrl/environments/backends/physics.py
    - mbrl/environments/backends/diagnostics.py
    - tests/test_environment_backends.py
    - tests/test_backend_diagnostics.py
    - zh_docs/changes/2026-03-05_genesis-external-runtime-bridge-phase12.md

- Run: run_2026_03_05_backend_decouple_phase13
  - Summary: merged delta at 2026-03-04T17:30:47Z
  - Source: runs/run_2026_03_05_backend_decouple_phase13/memory/backend-coder.delta.md
  - Notes:
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
