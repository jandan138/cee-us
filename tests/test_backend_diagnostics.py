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
            },
            clear=False,
        ):
            config = diagnostics_module.resolve_real_backend_switch_configuration()

        self.assertEqual(config["backend_plugin_modules"], ["tests.backend_plugin_example"])
        self.assertEqual(config["options_by_backend"], {"genesis": {"skip_dependency_check": True}})

    def test_real_switch_configuration_rejects_invalid_json_options(self):
        with patch.dict(
            os.environ,
            {"REAL_BACKEND_SWITCH_PHYSICS_OPTIONS_JSON": "{\"genesis\": "},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                diagnostics_module.resolve_real_backend_switch_configuration()

    def test_collect_backend_diagnostics_allows_real_switch_option_overrides(self):
        report = diagnostics_module.collect_backend_diagnostics(
            backend_names=["genesis"],
            real_switch_options_by_backend={"genesis": {"skip_dependency_check": True}},
        )
        real_switch = report["real_backend_switch_test"]
        self.assertEqual(
            real_switch["configured_physics_options_by_backend"].get("genesis"),
            {"skip_dependency_check": True},
        )

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
        self.assertIn("Physics backend 'Genesis' selected", physics_reason["reason"])
        self.assertIn("'genesis' package is not available", physics_reason["reason"])
        self.assertIn("skip_dependency_check", physics_reason["reason"])


if __name__ == "__main__":
    unittest.main()
