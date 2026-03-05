# Agent Memory: researcher

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
  - Source: runs/run_2026_03_04_backend_decouple_phase2/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase2
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - phase2 建议聚焦 backend capability metadata 与无 MuJoCo 单测。
    
    ## 2) New Stable Preferences
    - 先提高可观测性和测试覆盖，再推进真实外部引擎适配。
    
    ## 3) Pitfalls Learned
    - 只有工厂路由不足以保障演进，需要能力矩阵与自动化验证配套。
    
    ## 4) Reusable Patterns
    - 每个 backend class 显式提供 `implemented` 标记，并导出列表 API。
    
    ## 5) Evidence Links
    - agent_team/runs/run_2026_03_04_backend_decouple_phase2/logs/researcher.md

- Run: run_2026_03_04_backend_decouple_phase2
  - Summary: merged delta at 2026-03-04T15:11:00Z
  - Source: runs/run_2026_03_04_backend_decouple_phase2/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase2
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - phase2 建议聚焦 backend capability metadata 与无 MuJoCo 单测。
    
    ## 2) New Stable Preferences
    - 先提高可观测性和测试覆盖，再推进真实外部引擎适配。
    
    ## 3) Pitfalls Learned
    - 只有工厂路由不足以保障演进，需要能力矩阵与自动化验证配套。
    
    ## 4) Reusable Patterns
    - 每个 backend class 显式提供 `implemented` 标记，并导出列表 API。
    
    ## 5) Evidence Links
    - agent_team/runs/run_2026_03_04_backend_decouple_phase2/logs/researcher.md

- Run: run_2026_03_04_backend_decouple_phase3
  - Summary: merged delta at 2026-03-04T15:27:19Z
  - Source: runs/run_2026_03_04_backend_decouple_phase3/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase3
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - 已确认 phase3 最优先下一步是“运行时插件加载”而非继续堆内置分支。
    - 给出验收方向：配置化加载、默认兼容、错误可诊断、无 MuJoCo 单测可跑。
    
    ## 2) New Stable Preferences
    - 优先做“扩展路径能力”而不是一次性接某个具体引擎。
    
    ## 3) Pitfalls Learned
    - 只有 `register_*` API 不够，缺入口会导致第三方集成仍需改核心代码。
    
    ## 4) Reusable Patterns
    - 插件模块导入即注册（import-time registration）是低复杂度高收益方案。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase3/logs/researcher.md`

- Run: run_2026_03_04_backend_decouple_phase4
  - Summary: merged delta at 2026-03-04T15:33:07Z
  - Source: runs/run_2026_03_04_backend_decouple_phase4/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase4
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - 推荐下一步聚焦“插件加载线程安全与并发重复调用可靠性”，保持接口不变，仅增强内部保证。
    
    ## 2) New Stable Preferences
    - 优先做小步、可测、接口不变的稳定性改进（先 reliability，后 feature）。
    
    ## 3) Pitfalls Learned
    - 仅有“去重集合”不等于并发安全；多线程重复加载时仍可能出现竞态。
    
    ## 4) Reusable Patterns
    - 用模块级锁 + 原子状态更新实现“exactly-once registration”语义。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase4/logs/researcher.md`
    

- Run: run_2026_03_04_backend_decouple_phase5
  - Summary: merged delta at 2026-03-04T15:40:48Z
  - Source: runs/run_2026_03_04_backend_decouple_phase5/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase5
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - 明确了“真实后端切换测试”的最早引入时点：首个非 MuJoCo 后端可实例化并完成 env 映射后的同一天。
    - 给出了 5 条仓库内可执行 gating conditions（实现、映射、渲染、依赖、smoke 配置）。
    
    ## 2) New Stable Preferences
    - 采用“两层测试策略”：先框架级常驻测试，再真实后端 smoke 自动激活。
    
    ## 3) Pitfalls Learned
    - 只实现 backend class 不足以开展真实切换测试，必须同时具备 env registry 映射与可运行依赖环境。
    
    ## 4) Reusable Patterns
    - 用“无后端则 skip、有后端则强制执行”的测试骨架降低引入成本并避免误报。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase5/logs/researcher.md`

- Run: run_2026_03_04_backend_decouple_phase5
  - Summary: merged delta at 2026-03-04T15:44:36Z
  - Source: runs/run_2026_03_04_backend_decouple_phase5/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase5
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - 明确了“真实后端切换测试”的最早引入时点：首个非 MuJoCo 后端可实例化并完成 env 映射后的同一天。
    - 给出了 5 条仓库内可执行 gating conditions（实现、映射、渲染、依赖、smoke 配置）。
    
    ## 2) New Stable Preferences
    - 采用“两层测试策略”：先框架级常驻测试，再真实后端 smoke 自动激活。
    
    ## 3) Pitfalls Learned
    - 只实现 backend class 不足以开展真实切换测试，必须同时具备 env registry 映射与可运行依赖环境。
    
    ## 4) Reusable Patterns
    - 用“无后端则 skip、有后端则强制执行”的测试骨架降低引入成本并避免误报。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase5/logs/researcher.md`

- Run: run_2026_03_04_backend_decouple_phase6
  - Summary: merged delta at 2026-03-04T15:48:33Z
  - Source: runs/run_2026_03_04_backend_decouple_phase6/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase6
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - 建议新增统一的 backend readiness API，避免依赖异常文本来驱动测试门控。
    - 建议把真实 Genesis 切换测试改成“readiness 通过才执行，否则结构化 skip”。
    
    ## 2) New Stable Preferences
    - 对可选依赖后端（Genesis/Isaac/Newton）优先采用“状态探测 + 门控执行”，不要把缺依赖当失败。
    
    ## 3) Pitfalls Learned
    - 仅靠 `ImportError` 字符串判断会导致 CI 不稳定（信息变化即误报）。
    
    ## 4) Reusable Patterns
    - 三层测试矩阵：
      - L1: synthetic always-on
      - L2: real-backend gated by readiness + env var
      - L3: explicit skip with machine-readable reason
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_04_backend_decouple_phase6/logs/researcher.md`

- Run: run_2026_03_04_backend_decouple_phase7
  - Summary: merged delta at 2026-03-04T15:58:52Z
  - Source: runs/run_2026_03_04_backend_decouple_phase7/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_04_backend_decouple_phase7
    - Source Agent: researcher
    - Recorder: researcher
    
    ## 1) Delta Summary
    - Repo already has backend decoupling, plugin loading, Genesis physics dependency check, and gated real-switch test entrypoint.
    - Core gap identified: no unified readiness diagnostic for actual backend switch (dependency + env mapping + render readiness).
    - Recommended next incremental step: add `backend_switch_readiness(...)` API and use it in real backend switch tests for deterministic skip/fail reasons without requiring Genesis installation.
    
    ## 2) New Stable Preferences
    - For backend bring-up, prioritize deterministic diagnostics that work in environments without optional runtimes installed.
    
    ## 3) Pitfalls Learned
    - `physics_backend_readiness("genesis")` alone can be misleading; it does not verify `ENV_REGISTRY` mapping or render backend executability.
    - Real-switch test gating currently depends on dynamic conditions and can skip without rich reason granularity.
    
    ## 4) Reusable Patterns
    - Compose readiness in layers:
      1) backend implementation/dependency,
      2) env mapping existence,
      3) render backend readiness,
      4) optional runtime gate env var.
    - Use test-only dummy env mappings to validate framework behavior independent of external engine installs.
    
    ## 5) Evidence Links
    - `mbrl/environments/backends/physics.py`
    - `mbrl/environments/backends/render.py`
    - `mbrl/environments/backends/registry.py`
    - `tests/test_environment_backends.py`
    - `requirements.txt`
    - `requirements.no_mujoco.txt`

- Run: run_2026_03_05_backend_decouple_phase8
  - Summary: merged delta at 2026-03-04T16:10:02Z
  - Source: runs/run_2026_03_05_backend_decouple_phase8/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_05_backend_decouple_phase8
    - Source Agent: researcher
    - Recorder: researcher
    
    ## 1) Delta Summary
    - Confirmed phase7 already has structured physics readiness and real-switch precheck, but still lacks unified switch readiness that composes physics + render + env mapping.
    - Proposed phase8 additive contract:
      - add `render_backend_readiness(...)`
      - add `backend_switch_readiness(...)`
      - extend diagnostics with `candidate_switch_readiness` while preserving existing phase7 keys/semantics.
    - Verified current workspace behavior: genesis physics not ready (`ImportError`), no non-MuJoCo mapping candidate, backend tests pass with expected gated skip.
    
    ## 2) New Stable Preferences
    - For backend-switch diagnostics, prefer additive schema evolution over in-place key semantics changes.
    - Keep real-switch default viability checks aligned with the actual test constructor path (`render_backend="none"`).
    
    ## 3) Pitfalls Learned
    - `physics_backend_readiness` alone can produce false confidence for switchability because mapping and render constraints are external to that API.
    - Replacing `candidate_readiness` payload shape would risk phase7 test/API breakage; introduce `candidate_switch_readiness` instead.
    
    ## 4) Reusable Patterns
    - Readiness composition pattern for backend bring-up:
      - `mapping.exists` check
      - `physics.ready` check
      - `render.ready` check
      - aggregate to `ready` + `blocking_checks[]`
    - Backward-compatible diagnostics extension pattern:
      - preserve existing keys and meanings
      - add new keys for richer phase8 detail.
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_05_backend_decouple_phase8/logs/researcher.md`
    - `mbrl/environments/backends/diagnostics.py`
    - `mbrl/environments/backends/physics.py`
    - `mbrl/environments/backends/render.py`
    - `mbrl/environments/backends/registry.py`
    - `tests/test_environment_backends.py`

- Run: run_2026_03_05_backend_decouple_phase9
  - Summary: merged delta at 2026-03-04T16:31:37Z
  - Source: runs/run_2026_03_05_backend_decouple_phase9/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_05_backend_decouple_phase9
    - Source Agent: researcher
    - Recorder: researcher
    
    ## 1) Delta Summary
    - Validated with repo evidence that backend plugin modules can create a real-switch precheck candidate at runtime without editing static core registry source.
    - Confirmed current no-Genesis host behavior:
      - true-runtime Genesis readiness fails with explicit `ImportError`;
      - synthetic readiness remains available via `physics_backend_options.skip_dependency_check=true`.
    - Produced a low-risk phase9 plan: additive plugin-aware diagnostics + explicit runtime mode semantics (`true-runtime` vs `synthetic-runtime`) + deterministic candidate/test coverage.
    
    ## 2) New Stable Preferences
    - Prefer plugin-driven registration for backend switch candidate bring-up over hardcoding mappings in `mbrl/environments/backends/registry.py`.
    - Treat true-runtime and synthetic-runtime as separate contracts in tests/reports; never merge them into one implicit "ready" signal.
    - Preserve existing diagnostics/report keys and test gates; add fields instead of mutating semantics.
    
    ## 3) Pitfalls Learned
    - `candidate_readiness.ready=true` can be misleading unless runtime mode is explicit; synthetic-ready should not be interpreted as true-runtime-ready.
    - First-match candidate selection is sensitive to registry/plugin load order; deterministic candidate selection is safer for CI stability.
    - Plugin loading mutates global registries; snapshot/restore discipline in tests is required to prevent cross-test pollution.
    
    ## 4) Reusable Patterns
    - Candidate bring-up pattern (no static registry edits):
      - provide plugin module with `register_backends()`
      - load via `backend_plugin_modules`
      - run diagnostics/test gate afterward.
    - Synthetic-vs-true split pattern:
      - true-runtime: dependency checks ON, no bypass options
      - synthetic-runtime: explicit bypass options ON + explicit mode label in diagnostics output.
    - Low-risk evolution pattern:
      - maintain backward-compatible defaults and keys;
      - add optional plugin/options inputs and new report fields.
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_05_backend_decouple_phase9/logs/researcher.md`
    - `mbrl/environments/backends/plugins.py`
    - `mbrl/environments/backends/physics.py`
    - `mbrl/environments/backends/diagnostics.py`
    - `mbrl/environments/__init__.py`
    - `mbrl/main.py`
    - `tests/test_environment_backends.py`
    - `tests/test_backend_diagnostics.py`
    - `tests/backend_plugin_example.py`

- Run: run_2026_03_05_backend_decouple_phase10
  - Summary: merged delta at 2026-03-04T16:45:32Z
  - Source: runs/run_2026_03_05_backend_decouple_phase10/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_05_backend_decouple_phase10
    - Source Agent: researcher
    - Recorder: researcher
    
    ## 1) Delta Summary
    - What changed in this run:
      - Produced a phase10 design contract to explicitly distinguish `true-runtime` vs `synthetic-runtime` in real backend switch diagnostics and gated test path.
      - Verified from repo and runtime probes that current phase9 behavior can report `would_skip=false` for synthetic Genesis (`skip_dependency_check=true`) without explicit mode labeling.
      - Defined additive API/env-var/CLI/test extensions that preserve default behavior.
    
    ## 2) New Stable Preferences
    - Prefer additive diagnostics fields over key/semantic rewrites (`candidate_runtime_mode`, policy metadata, per-backend mode map).
    - Keep strictness opt-in via env var/API arg (`REAL_BACKEND_SWITCH_RUNTIME_POLICY=require_true_runtime`) and default to legacy-compatible `allow_synthetic`.
    - Require skip reasons in strict mode to mention detected runtime mode and concrete remediation.
    
    ## 3) Pitfalls Learned
    - `candidate_readiness.ready=true` alone is insufficient for release confidence; it can represent synthetic bypass rather than true runtime.
    - Reusing `candidate_physics_backend_options` directly in env construction couples diagnostic ambiguity into gated tests unless runtime policy is enforced.
    - Plugin and options bridges are powerful, but without explicit mode labels reviewers can misread execution status.
    
    ## 4) Reusable Patterns
    - Minimal-risk evolution pattern:
      - keep existing outputs and defaults untouched;
      - add explicit runtime-mode metadata;
      - add opt-in strict policy.
    - Gating pattern:
      - diagnostics produces a candidate + runtime mode;
      - test helper enforces policy and emits mode-aware skip reasons.
    - Evidence pattern:
      - validate with paired probes (plugin true-like path vs genesis synthetic path) to catch semantic conflation.
    
    ## 5) Evidence Links
    - `mbrl/environments/backends/diagnostics.py`
    - `mbrl/environments/backends/physics.py`
    - `tests/test_environment_backends.py`
    - `tests/test_backend_diagnostics.py`
    - `zh_docs/changes/2026-03-05_real-backend-switch-config-bridge-phase9.md`
    - `agent_team/runs/run_2026_03_05_backend_decouple_phase10/logs/researcher.md`

- Run: run_2026_03_05_backend_decouple_phase12
  - Summary: merged delta at 2026-03-04T17:11:19Z
  - Source: runs/run_2026_03_05_backend_decouple_phase12/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_05_backend_decouple_phase12
    - Source Agent: researcher
    - Recorder: researcher
    
    ## 1) Delta Summary
    - 完成 phase12 研究结论：当前仓库 Genesis readiness 仅检查当前 Python 进程（py3.8），无法识别外部 `.venv-genesis`（py3.10）已安装场景。
    - 提出最小可行桥接方案（不改默认行为）：在 `physics_backend_options` 中显式提供外部 Python 探测配置，将结果并入 readiness 与 real-switch diagnostics。
    - 给出与 strict true-runtime 兼容的规则：`synthetic-runtime` 继续被 strict 拦截，`external-runtime` 视为非 synthetic 真实依赖来源。
    
    ## 2) New Stable Preferences
    - 对“多 Python 版本依赖”问题优先采用 diagnostics-first 方案：先解决 readiness/可观测性，再评估跨进程执行桥。
    - 任何新增后端桥接能力都应保持默认零侵入（未配置不生效、旧参数不变）。
    - 对配置命名采用双轨：代码内 `physics_backend_options.*`，CI 侧 `REAL_BACKEND_SWITCH_*` 环境变量。
    
    ## 3) Pitfalls Learned
    - `importlib.util.find_spec` 仅反映当前解释器状态，不能代表系统上其他 venv 的可用性。
    - 把 external-runtime 与 synthetic-runtime 混为一类会误伤 strict true-runtime 门禁。
    - “外部 import 成功”不等于“主进程可直接仿真执行”，必须在文档中清晰标注边界。
    
    ## 4) Reusable Patterns
    - 后端 readiness 三态模式：`true-runtime` / `external-runtime` / `synthetic-runtime`。
    - 外部探测实现要点：`subprocess.run(..., shell=False)` + timeout + 结构化错误摘要。
    - 诊断返回新增 `dependency_source` 与 `runtime_probe` 元数据，便于 CI 排障与治理审计。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_05_backend_decouple_phase12/logs/researcher.md`
    - `mbrl/environments/backends/physics.py`
    - `mbrl/environments/backends/diagnostics.py`
    - `tests/test_backend_diagnostics.py`
    - `tests/test_environment_backends.py`

- Run: run_2026_03_05_backend_decouple_phase13
  - Summary: merged delta at 2026-03-04T17:30:47Z
  - Source: runs/run_2026_03_05_backend_decouple_phase13/memory/researcher.delta.md
  - Notes:
    # Memory Delta: researcher
    
    - Run ID: run_2026_03_05_backend_decouple_phase13
    - Source Agent: researcher
    - Recorder: doc-writer
    
    ## 1) Delta Summary
    - What changed in this run:
      - 明确了 phase13 的最小优先方向不是继续改 backend 逻辑，而是补 1 个“无 mock 的 Genesis 真实切换集成测试”（默认 gate，不影响普通 CI）。
      - 证实当前代码已具备 strict true-runtime 与 external probe 能力；主要缺口是 default `ENV_REGISTRY` 没有 genesis 映射，导致 real-switch candidate 发现会 skip。
      - 在本机验证“今天可落地”：`.venv-genesis` 可导入 genesis，注入临时 genesis 映射后 strict real-switch 可达 `would_skip=False` 且 `candidate_runtime_mode=true-runtime`。
    
    ## 2) New Stable Preferences
    - 在“切换后端真实性”议题上，优先要求端到端证据（真实解释器/真实 probe/真实 candidate）而非仅 mock 单元覆盖。
    - 在不改生产代码前提下，优先用 tests 侧临时映射 + env gate 形成最小闭环。
    
    ## 3) Pitfalls Learned
    - 仅设置 `REAL_BACKEND_SWITCH_GENESIS_*` 不足以让 real-switch 执行，若无 genesis 映射仍会 skip。
    - diagnostics 报告里 `physics_backend_readiness` 与 `real_backend_switch_test.implemented_backend_readiness` 可能因 options 来源不同出现状态差异，阅读时需要区分。
    - 仓库中无独立 phase11 变更文档；若外部汇报需标明“phase11 信息来自 phase12 文档引用与代码现状”。
    
    ## 4) Reusable Patterns
    - 真实切换可执行三步法：
      - 先检查 external runtime：`<external_python> -c "find_spec('genesis')"`
      - 再跑 strict diagnostics（带 `REAL_BACKEND_SWITCH_GENESIS_*`）
      - 最后在测试上下文临时注册 genesis 映射并执行 `env_from_string`/`diagnose_real_backend_switch_test`
    - phase 评审模板优先包含：最小改动、验收标准、今日可落地性、风险矩阵、可直接复制的命令。
    
    ## 5) Evidence Links
    - `agent_team/runs/run_2026_03_05_backend_decouple_phase13/logs/researcher.md`
    - `mbrl/environments/backends/physics.py`
    - `mbrl/environments/backends/diagnostics.py`
    - `tests/test_environment_backends.py`
    - `tests/test_genesis_external_runtime_bridge.py`
    - `zh_docs/changes/2026-03-05_real-backend-switch-runtime-policy-phase10.md`
    - `zh_docs/changes/2026-03-05_genesis-external-runtime-bridge-phase12.md`
