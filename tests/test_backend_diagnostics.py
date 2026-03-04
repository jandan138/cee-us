import copy
import os
import unittest
from unittest.mock import patch

from mbrl.environments.backends import register_env_backend
from mbrl.environments.backends import diagnostics as diagnostics_module
from mbrl.environments.backends import physics as physics_backend_module
from mbrl.environments.backends.registry import ENV_REGISTRY


class BackendDiagnosticsTestCase(unittest.TestCase):
    def setUp(self):
        self._registry_snapshot = copy.deepcopy(ENV_REGISTRY)

    def tearDown(self):
        ENV_REGISTRY.clear()
        ENV_REGISTRY.update(self._registry_snapshot)

    def _require_unified_switch_api(self):
        required_symbols = (
            "render_backend_readiness",
            "diagnose_unified_switch_readiness",
            "resolve_real_backend_switch_configuration",
            "resolve_real_backend_switch_execution_target",
        )
        missing = [name for name in required_symbols if not hasattr(diagnostics_module, name)]
        self.assertFalse(
            missing,
            "Unified switch readiness diagnostics API is missing symbols: "
            + ", ".join(sorted(missing)),
        )

    def test_real_switch_configuration_parses_env_vars(self):
        with patch.dict(
            os.environ,
            {
                "REAL_BACKEND_SWITCH_PLUGIN_MODULES": "tests.backend_plugin_example, tests.backend_plugin_example",
                "REAL_BACKEND_SWITCH_PHYSICS_OPTIONS_JSON": "{\"genesis\": {\"skip_dependency_check\": true}}",
                "REAL_BACKEND_SWITCH_REQUIRE_TRUE_RUNTIME": "1",
                "REAL_BACKEND_SWITCH_GENESIS_EXTERNAL_PYTHON": "",
                "REAL_BACKEND_SWITCH_GENESIS_PROBE_TIMEOUT_SEC": "",
                "REAL_BACKEND_SWITCH_GENESIS_PYOPENGL_PLATFORM": "",
            },
            clear=False,
        ):
            config = diagnostics_module.resolve_real_backend_switch_configuration()

        self.assertEqual(config["backend_plugin_modules"], ["tests.backend_plugin_example"])
        self.assertEqual(config["options_by_backend"], {"genesis": {"skip_dependency_check": True}})
        self.assertTrue(config["require_true_runtime"])

    def test_real_switch_configuration_rejects_invalid_json_options(self):
        with patch.dict(
            os.environ,
            {"REAL_BACKEND_SWITCH_PHYSICS_OPTIONS_JSON": "{\"genesis\": "},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                diagnostics_module.resolve_real_backend_switch_configuration()

    def test_real_switch_configuration_rejects_invalid_require_true_runtime(self):
        with patch.dict(
            os.environ,
            {"REAL_BACKEND_SWITCH_REQUIRE_TRUE_RUNTIME": "strict"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                diagnostics_module.resolve_real_backend_switch_configuration()

    def test_real_switch_configuration_bridges_genesis_external_probe_env_vars(self):
        with patch.dict(
            os.environ,
            {
                "REAL_BACKEND_SWITCH_GENESIS_EXTERNAL_PYTHON": ".venv-genesis/bin/python",
                "REAL_BACKEND_SWITCH_GENESIS_PROBE_TIMEOUT_SEC": "15",
                "REAL_BACKEND_SWITCH_GENESIS_PYOPENGL_PLATFORM": "egl",
            },
            clear=False,
        ):
            config = diagnostics_module.resolve_real_backend_switch_configuration()

        genesis_options = config["options_by_backend"]["genesis"]
        self.assertEqual(genesis_options["external_python"], ".venv-genesis/bin/python")
        self.assertEqual(genesis_options["external_probe_timeout_sec"], 15.0)
        self.assertEqual(genesis_options["PYOPENGL_PLATFORM"], "egl")

    def test_real_switch_configuration_rejects_invalid_genesis_probe_timeout(self):
        with patch.dict(
            os.environ,
            {"REAL_BACKEND_SWITCH_GENESIS_PROBE_TIMEOUT_SEC": "0"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                diagnostics_module.resolve_real_backend_switch_configuration()

    def test_collect_backend_diagnostics_allows_real_switch_option_overrides(self):
        with patch.dict(
            os.environ,
            {
                "REAL_BACKEND_SWITCH_GENESIS_EXTERNAL_PYTHON": "",
                "REAL_BACKEND_SWITCH_GENESIS_PROBE_TIMEOUT_SEC": "",
                "REAL_BACKEND_SWITCH_GENESIS_PYOPENGL_PLATFORM": "",
            },
            clear=False,
        ):
            report = diagnostics_module.collect_backend_diagnostics(
                backend_names=["genesis"],
                real_switch_options_by_backend={"genesis": {"skip_dependency_check": True}},
            )
        real_switch = report["real_backend_switch_test"]
        genesis_options = real_switch["configured_physics_options_by_backend"].get("genesis")
        self.assertEqual(genesis_options.get("skip_dependency_check"), True)

    def test_real_switch_reports_candidate_runtime_mode(self):
        register_env_backend(
            "GenesisRuntimeModeEnv",
            "genesis",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
            overwrite=True,
        )
        with patch.object(physics_backend_module.GenesisPhysicsBackend, "_is_genesis_available", return_value=False):
            report = diagnostics_module.diagnose_real_backend_switch_test(
                enable_real_backend_tests=1,
                options_by_backend={"genesis": {"skip_dependency_check": True}},
            )

        self.assertEqual(report["candidate"]["env_name"], "GenesisRuntimeModeEnv")
        self.assertEqual(report["candidate"]["backend_name"], "genesis")
        self.assertEqual(report["candidate_runtime_mode"], "synthetic-runtime")
        self.assertEqual(report["candidate_runtime_mode_reason"], "skip_dependency_check=true")
        self.assertFalse(report["require_true_runtime_flag"]["enabled"])

    def test_real_switch_strict_policy_blocks_synthetic_candidate_via_api(self):
        register_env_backend(
            "GenesisStrictPolicyEnv",
            "genesis",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
            overwrite=True,
        )
        with patch.object(physics_backend_module.GenesisPhysicsBackend, "_is_genesis_available", return_value=False):
            report = diagnostics_module.diagnose_real_backend_switch_test(
                enable_real_backend_tests=1,
                options_by_backend={"genesis": {"skip_dependency_check": True}},
                require_true_runtime=True,
            )

        self.assertTrue(report["require_true_runtime_flag"]["enabled"])
        self.assertTrue(report["require_true_runtime_flag"]["violated"])
        self.assertTrue(report["would_skip"])
        self.assertIn("Strict true-runtime policy is enabled", report["first_skip_reason"])
        self.assertIn("synthetic-runtime", report["first_skip_reason"])

    def test_real_switch_reports_external_runtime_mode_when_readiness_uses_external_dependency(self):
        register_env_backend(
            "GenesisExternalRuntimeModeEnv",
            "genesis",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
            overwrite=True,
        )
        with patch.object(physics_backend_module.GenesisPhysicsBackend, "_is_genesis_available", return_value=False):
            with patch.object(
                physics_backend_module.GenesisPhysicsBackend,
                "_is_genesis_available_via_external_runtime",
                return_value=(True, "module import probe succeeded in external runtime"),
            ):
                report = diagnostics_module.diagnose_real_backend_switch_test(
                    enable_real_backend_tests=1,
                    options_by_backend={"genesis": {"external_python": ".venv-genesis/bin/python"}},
                    require_true_runtime=True,
                )

        self.assertIsNotNone(report["candidate"])
        self.assertEqual(report["candidate"]["backend_name"], "genesis")
        self.assertEqual(report["candidate_runtime_mode"], "external-runtime")
        self.assertEqual(report["candidate_runtime_mode_reason"], "dependency_source=external")
        self.assertFalse(report["require_true_runtime_flag"]["violated"])
        self.assertFalse(report["would_skip"])

    def test_resolve_real_backend_switch_execution_target_selects_candidate(self):
        report = {
            "candidate": {"env_name": "GenesisCandidateEnv", "backend_name": "genesis"},
            "candidate_physics_backend_options": {"external_python": ".venv-genesis/bin/python"},
            "candidate_runtime_mode": "external-runtime",
            "candidate_runtime_mode_reason": "dependency_source=external",
            "would_skip": False,
            "first_skip_reason": "",
        }
        with patch.object(diagnostics_module, "diagnose_real_backend_switch_test", return_value=report):
            target = diagnostics_module.resolve_real_backend_switch_execution_target(enable_real_backend_tests=1)

        self.assertTrue(target["selected"])
        self.assertEqual(target["skip_reason"], "")
        self.assertEqual(target["env_name"], "GenesisCandidateEnv")
        self.assertEqual(target["backend_name"], "genesis")
        self.assertEqual(target["physics_backend_options"], {"external_python": ".venv-genesis/bin/python"})
        self.assertEqual(target["candidate_runtime_mode"], "external-runtime")

    def test_resolve_real_backend_switch_execution_target_propagates_skip_reason(self):
        report = {
            "candidate": {"env_name": "GenesisSyntheticEnv", "backend_name": "genesis"},
            "candidate_physics_backend_options": {"skip_dependency_check": True},
            "candidate_runtime_mode": "synthetic-runtime",
            "candidate_runtime_mode_reason": "skip_dependency_check=true",
            "would_skip": True,
            "first_skip_reason": "Strict true-runtime policy is enabled",
        }
        with patch.object(diagnostics_module, "diagnose_real_backend_switch_test", return_value=report):
            target = diagnostics_module.resolve_real_backend_switch_execution_target(enable_real_backend_tests=1)

        self.assertFalse(target["selected"])
        self.assertEqual(target["skip_reason"], "Strict true-runtime policy is enabled")
        self.assertIsNone(target["env_name"])
        self.assertIsNone(target["backend_name"])
        self.assertEqual(target["physics_backend_options"], {})

    def test_diagnostics_cli_accepts_real_switch_require_true_runtime_flag(self):
        with patch.object(diagnostics_module, "collect_backend_diagnostics", return_value={"ok": True}) as collect_fn:
            with patch("builtins.print"):
                diagnostics_module.main(["--real-switch-require-true-runtime", "--json"])

        self.assertTrue(collect_fn.call_args.kwargs["real_switch_require_true_runtime"])

    def test_diagnostics_cli_accepts_genesis_external_probe_flags(self):
        with patch.object(diagnostics_module, "collect_backend_diagnostics", return_value={"ok": True}) as collect_fn:
            with patch("builtins.print"):
                diagnostics_module.main(
                    [
                        "--real-switch-genesis-external-python",
                        ".venv-genesis/bin/python",
                        "--real-switch-genesis-probe-timeout-sec",
                        "12",
                        "--real-switch-genesis-pyopengl-platform",
                        "egl",
                        "--json",
                    ]
                )

        kwargs = collect_fn.call_args.kwargs
        self.assertEqual(kwargs["real_switch_genesis_external_python"], ".venv-genesis/bin/python")
        self.assertEqual(kwargs["real_switch_genesis_probe_timeout_sec"], 12.0)
        self.assertEqual(kwargs["real_switch_genesis_pyopengl_platform"], "egl")

    def test_unified_switch_readiness_reports_missing_env_mapping(self):
        self._require_unified_switch_api()
        report = diagnostics_module.diagnose_unified_switch_readiness(
            "GenesisMissingMappingEnv",
            physics_backend_name="genesis",
            render_backend_name="none",
            physics_backend_options={"skip_dependency_check": True},
        )

        self.assertFalse(report["ready"])
        self.assertTrue(report["physics_backend_readiness"]["ready"])
        self.assertTrue(report["render_backend_readiness"]["ready"])
        self.assertFalse(report["env_mapping"]["exists"])
        self.assertEqual(report["env_mapping"]["error_type"], "KeyError")
        self.assertIn("GenesisMissingMappingEnv", report["env_mapping"]["reason"])
        self.assertIn("ENV_REGISTRY", report["env_mapping"]["reason"])

    def test_unified_switch_readiness_reports_render_backend_unimplemented(self):
        self._require_unified_switch_api()
        register_env_backend(
            "GenesisRenderUnimplementedEnv",
            "genesis",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
            overwrite=True,
        )
        report = diagnostics_module.diagnose_unified_switch_readiness(
            "GenesisRenderUnimplementedEnv",
            physics_backend_name="genesis",
            render_backend_name="genesis",
            physics_backend_options={"skip_dependency_check": True},
        )

        self.assertFalse(report["ready"])
        self.assertTrue(report["physics_backend_readiness"]["ready"])
        self.assertTrue(report["env_mapping"]["exists"])
        self.assertFalse(report["render_backend_readiness"]["ready"])
        self.assertFalse(report["render_backend_readiness"]["implemented"])
        self.assertEqual(report["render_backend_readiness"]["error_type"], "NotImplementedError")
        self.assertIn("registered but not implemented", report["render_backend_readiness"]["reason"])

    def test_unified_switch_readiness_synthetic_pass_with_skip_dependency_and_render_none(self):
        self._require_unified_switch_api()
        register_env_backend(
            "GenesisSyntheticReadyEnv",
            "genesis",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
            overwrite=True,
        )
        report = diagnostics_module.diagnose_unified_switch_readiness(
            "GenesisSyntheticReadyEnv",
            physics_backend_name="genesis",
            render_backend_name="none",
            physics_backend_options={"skip_dependency_check": True},
        )

        self.assertTrue(report["ready"])
        self.assertTrue(report["physics_backend_readiness"]["ready"])
        self.assertTrue(report["render_backend_readiness"]["ready"])
        self.assertTrue(report["render_backend_readiness"]["implemented"])
        self.assertTrue(report["env_mapping"]["exists"])
        self.assertEqual(report["resolved"]["physics_backend"], "genesis")
        self.assertEqual(report["resolved"]["render_backend"], "none")

    def test_unified_switch_readiness_reports_explicit_physics_reason_messages(self):
        self._require_unified_switch_api()
        register_env_backend(
            "GenesisExplicitReasonEnv",
            "genesis",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
            overwrite=True,
        )

        with patch.object(physics_backend_module.GenesisPhysicsBackend, "_is_genesis_available", return_value=False):
            report = diagnostics_module.diagnose_unified_switch_readiness(
                "GenesisExplicitReasonEnv",
                physics_backend_name="genesis",
                render_backend_name="none",
            )

        self.assertFalse(report["ready"])
        physics_reason = report["physics_backend_readiness"]
        self.assertFalse(physics_reason["ready"])
        self.assertEqual(physics_reason["error_type"], "ImportError")
        self.assertEqual(physics_reason["dependency_source"], "local")
        self.assertIn("Physics backend 'Genesis' selected", physics_reason["reason"])
        self.assertIn("'genesis' package is not available", physics_reason["reason"])
        self.assertIn("skip_dependency_check", physics_reason["reason"])


if __name__ == "__main__":
    unittest.main()
