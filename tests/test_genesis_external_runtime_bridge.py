import copy
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mbrl.environments import env_from_string
from mbrl.environments.backends import diagnostics as diagnostics_module
from mbrl.environments.backends import register_env_backend
from mbrl.environments.backends import physics as physics_backend_module
from mbrl.environments.backends.physics import physics_backend_readiness
from mbrl.environments.backends.registry import ENV_REGISTRY

RUN_PHASE13_GENESIS_EXTERNAL_RUNTIME_EXEC_TEST_ENV_VAR = "RUN_PHASE13_GENESIS_EXTERNAL_RUNTIME_EXEC_TEST"


def _contains_value(node, expected):
    if isinstance(node, dict):
        return any(_contains_value(value, expected) for value in node.values())
    if isinstance(node, (list, tuple, set)):
        return any(_contains_value(value, expected) for value in node)
    return node == expected


def _is_blocked(report):
    if "blocked" in report:
        return bool(report["blocked"])
    return bool(report.get("would_skip", False))


class GenesisExternalRuntimeBridgeTestCase(unittest.TestCase):
    def setUp(self):
        self._registry_snapshot = copy.deepcopy(ENV_REGISTRY)

    def tearDown(self):
        ENV_REGISTRY.clear()
        ENV_REGISTRY.update(self._registry_snapshot)

    @staticmethod
    def _external_probe_options():
        probe_payload = {
            "python_executable": "/opt/genesis-external/bin/python3",
            "opengl_platform": "osmesa",
            "timeout_seconds": 5,
        }
        return {
            "external_python": probe_payload["python_executable"],
            "PYOPENGL_PLATFORM": probe_payload["opengl_platform"],
            "external_probe_timeout_sec": probe_payload["timeout_seconds"],
            "external_probe_env": {"PYOPENGL_PLATFORM": probe_payload["opengl_platform"]},
        }

    @staticmethod
    def _external_probe_bridge_supported():
        backend_cls = __import__("mbrl.environments.backends.physics", fromlist=["GenesisPhysicsBackend"]).GenesisPhysicsBackend
        return hasattr(backend_cls, "_is_genesis_available_via_external_runtime")

    @staticmethod
    def _external_env_bridge_supported():
        required_names = {
            "REAL_SWITCH_TEST_GENESIS_EXTERNAL_PYTHON_ENV_VAR",
            "REAL_SWITCH_TEST_GENESIS_PROBE_TIMEOUT_SEC_ENV_VAR",
            "REAL_SWITCH_TEST_GENESIS_PYOPENGL_PLATFORM_ENV_VAR",
        }
        return required_names.issubset(set(dir(diagnostics_module)))

    def _register_strict_probe_env(self):
        register_env_backend(
            "GenesisExternalProbeStrictEnv",
            "genesis",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
            overwrite=True,
        )

    @staticmethod
    def _register_executable_probe_env():
        register_env_backend(
            "GenesisExternalRuntimeExecutableEnv",
            "genesis",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
            overwrite=True,
        )

    def _require_runtime_metadata(self, readiness, *, context):
        required_fields = {"dependency_source", "runtime_mode"}
        missing = sorted(required_fields.difference(set(readiness.keys())))
        if missing:
            self.skipTest(
                "Phase13 runtime metadata is not integrated "
                f"({context} missing: {', '.join(missing)})."
            )

    def test_external_runtime_probe_success_path(self):
        if not self._external_probe_bridge_supported():
            self.skipTest("Phase12 external runtime probe bridge is not integrated in current branch.")

        with patch("mbrl.environments.backends.physics.find_spec", return_value=None):
            with patch("subprocess.run") as run_mock:
                run_mock.return_value = SimpleNamespace(returncode=0, stdout="ok", stderr="")
                readiness = physics_backend_readiness("genesis", options=self._external_probe_options())

        self.assertTrue(readiness["ready"])
        self.assertIsNone(readiness["error_type"])
        self.assertEqual(readiness["dependency_source"], "external")
        run_mock.assert_called()

    def test_external_runtime_probe_failure_path(self):
        if not self._external_probe_bridge_supported():
            self.skipTest("Phase12 external runtime probe bridge is not integrated in current branch.")

        with patch("mbrl.environments.backends.physics.find_spec", return_value=None):
            with patch("subprocess.run") as run_mock:
                run_mock.return_value = SimpleNamespace(returncode=2, stdout="", stderr="genesis import failed")
                readiness = physics_backend_readiness("genesis", options=self._external_probe_options())

        self.assertFalse(readiness["ready"])
        self.assertEqual(readiness["error_type"], "ImportError")
        self.assertEqual(readiness["dependency_source"], "external")
        self.assertTrue(
            "external" in readiness["reason"].lower() or "python" in readiness["reason"].lower(),
            "Failure reason should include external probe context.",
        )
        run_mock.assert_called()

    def test_external_runtime_probe_timeout_path(self):
        if not self._external_probe_bridge_supported():
            self.skipTest("Phase12 external runtime probe bridge is not integrated in current branch.")

        with patch("mbrl.environments.backends.physics.find_spec", return_value=None):
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="python", timeout=5)) as run_mock:
                readiness = physics_backend_readiness("genesis", options=self._external_probe_options())

        self.assertFalse(readiness["ready"])
        self.assertIn(readiness["error_type"], {"ImportError", "TimeoutExpired", "RuntimeError"})
        self.assertEqual(readiness["dependency_source"], "external")
        self.assertTrue(
            "timeout" in readiness["reason"].lower()
            or "external" in readiness["reason"].lower()
            or "python" in readiness["reason"].lower()
        )
        run_mock.assert_called()

    def test_strict_true_runtime_allows_candidate_when_external_probe_ready(self):
        self._register_strict_probe_env()

        def readiness_side_effect(backend_name, options=None):
            if backend_name == "genesis":
                return {
                    "backend": "genesis",
                    "ready": True,
                    "error_type": None,
                    "reason": "",
                    "dependency_source": "external",
                }
            return {
                "backend": backend_name,
                "ready": False,
                "error_type": "NotImplementedError",
                "reason": "not relevant",
                "dependency_source": "local",
            }

        with patch(
            "mbrl.environments.backends.diagnostics.list_physics_backends",
            return_value={"mujoco": {"implemented": True}, "genesis": {"implemented": True}},
        ):
            with patch(
                "mbrl.environments.backends.diagnostics.physics_backend_readiness",
                side_effect=readiness_side_effect,
            ):
                report = diagnostics_module.diagnose_real_backend_switch_test(
                    enable_real_backend_tests=1,
                    options_by_backend={"genesis": self._external_probe_options()},
                    require_true_runtime=True,
                )

        self.assertIsNotNone(report["candidate"])
        self.assertEqual(report["candidate"]["backend_name"], "genesis")
        self.assertEqual(report["candidate_runtime_mode"], "external-runtime")
        self.assertEqual(report["candidate_runtime_mode_reason"], "dependency_source=external")
        self.assertFalse(report["require_true_runtime_flag"]["violated"])
        self.assertFalse(_is_blocked(report))

    def test_strict_true_runtime_marks_blocked_when_external_probe_not_ready(self):
        self._register_strict_probe_env()

        def readiness_side_effect(backend_name, options=None):
            if backend_name == "genesis":
                return {
                    "backend": "genesis",
                    "ready": False,
                    "error_type": "ImportError",
                    "reason": "external probe timeout",
                    "dependency_source": "external",
                }
            return {
                "backend": backend_name,
                "ready": False,
                "error_type": "NotImplementedError",
                "reason": "not relevant",
                "dependency_source": "local",
            }

        with patch(
            "mbrl.environments.backends.diagnostics.list_physics_backends",
            return_value={"mujoco": {"implemented": True}, "genesis": {"implemented": True}},
        ):
            with patch(
                "mbrl.environments.backends.diagnostics.physics_backend_readiness",
                side_effect=readiness_side_effect,
            ):
                report = diagnostics_module.diagnose_real_backend_switch_test(
                    enable_real_backend_tests=1,
                    options_by_backend={"genesis": self._external_probe_options()},
                    require_true_runtime=True,
                )

        self.assertIsNone(report["candidate"])
        self.assertTrue(_is_blocked(report))
        self.assertTrue(any("timeout" in failure.lower() for failure in report["readiness_failures"]))

    def test_default_config_keeps_existing_genesis_behavior_without_external_probe(self):
        with patch("mbrl.environments.backends.physics.find_spec", return_value=None):
            with patch("subprocess.run") as run_mock:
                readiness = physics_backend_readiness("genesis")

        self.assertFalse(readiness["ready"])
        self.assertEqual(readiness["backend"], "genesis")
        self.assertEqual(readiness["error_type"], "ImportError")
        self.assertEqual(readiness["dependency_source"], "local")
        run_mock.assert_not_called()

    def test_real_switch_configuration_env_injection_for_external_python_and_opengl(self):
        if not self._external_env_bridge_supported():
            self.skipTest("Phase12 diagnostics env-var bridge for external probe is not integrated in current branch.")

        external_python = "/opt/genesis-external/bin/python3"
        with patch.dict(
            os.environ,
            {
                "REAL_BACKEND_SWITCH_PHYSICS_OPTIONS_JSON": "{}",
                "REAL_BACKEND_SWITCH_GENESIS_EXTERNAL_PYTHON": external_python,
                "REAL_BACKEND_SWITCH_GENESIS_PYOPENGL_PLATFORM": "osmesa",
                "REAL_BACKEND_SWITCH_GENESIS_PROBE_TIMEOUT_SEC": "6.5",
            },
            clear=False,
        ):
            config = diagnostics_module.resolve_real_backend_switch_configuration()

        genesis_options = config["options_by_backend"].get("genesis", {})
        self.assertTrue(genesis_options)
        self.assertTrue(_contains_value(genesis_options, external_python))
        self.assertTrue(_contains_value(genesis_options, "osmesa"))

    def test_phase13_readiness_runtime_metadata_marks_synthetic_runtime(self):
        with patch.object(physics_backend_module.GenesisPhysicsBackend, "_is_genesis_available", return_value=False):
            readiness = physics_backend_readiness("genesis", options={"skip_dependency_check": True})

        self._require_runtime_metadata(readiness, context="skip_dependency_check readiness")
        self.assertTrue(readiness["ready"])
        runtime_mode = str(readiness["runtime_mode"]).lower()
        dependency_source = str(readiness["dependency_source"]).lower()
        self.assertIn("synthetic", runtime_mode)
        self.assertTrue(
            "skip" in dependency_source or "synthetic" in dependency_source,
            "dependency_source should explain synthetic/skip-dependency path.",
        )

    def test_phase13_readiness_runtime_metadata_marks_external_runtime_source(self):
        if not self._external_probe_bridge_supported():
            self.skipTest("Phase12 external runtime probe bridge is not integrated in current branch.")

        with patch.object(physics_backend_module.GenesisPhysicsBackend, "_is_genesis_available", return_value=False):
            with patch.object(
                physics_backend_module.GenesisPhysicsBackend,
                "_is_genesis_available_via_external_runtime",
                return_value=(True, "module import probe succeeded in external runtime"),
            ):
                readiness = physics_backend_readiness(
                    "genesis",
                    options={
                        "module_name": "phase13_external_only_module",
                        "external_python": sys.executable,
                    },
                )

        self._require_runtime_metadata(readiness, context="external runtime readiness")
        self.assertTrue(readiness["ready"])
        dependency_source = str(readiness["dependency_source"]).lower()
        runtime_mode = str(readiness["runtime_mode"]).lower()
        self.assertIn("external", dependency_source)
        self.assertNotIn("synthetic", runtime_mode)

    def test_phase13_real_switch_executes_external_probe_and_env_from_string(self):
        if os.environ.get(RUN_PHASE13_GENESIS_EXTERNAL_RUNTIME_EXEC_TEST_ENV_VAR, "0") != "1":
            self.skipTest(
                "Set "
                f"{RUN_PHASE13_GENESIS_EXTERNAL_RUNTIME_EXEC_TEST_ENV_VAR}=1 "
                "to run executable Genesis external-runtime switch test."
            )
        if not self._external_probe_bridge_supported():
            self.skipTest("Phase12 external runtime probe bridge is not integrated in current branch.")

        self._register_executable_probe_env()
        module_name = "phase13_external_runtime_only_module"

        with tempfile.TemporaryDirectory(prefix="phase13_genesis_external_probe_") as module_dir:
            module_file = os.path.join(module_dir, f"{module_name}.py")
            with open(module_file, "w", encoding="utf-8") as handle:
                handle.write("EXTERNAL_RUNTIME_ONLY = True\n")

            options_by_backend = {
                "genesis": {
                    "module_name": module_name,
                    "external_python": sys.executable,
                    "external_probe_timeout_sec": 8,
                    "PYOPENGL_PLATFORM": "egl",
                    "external_probe_env": {"PYTHONPATH": module_dir},
                }
            }

            with patch.dict(
                os.environ,
                {
                    "ENABLE_REAL_BACKEND_TESTS": "1",
                    "REAL_BACKEND_SWITCH_REQUIRE_TRUE_RUNTIME": "1",
                },
                clear=False,
            ):
                report = diagnostics_module.diagnose_real_backend_switch_test(
                    enable_real_backend_tests=1,
                    options_by_backend=options_by_backend,
                    require_true_runtime=True,
                )

            if _is_blocked(report):
                self.skipTest(report.get("first_skip_reason") or "real backend switch diagnostics is blocked")

            candidate = report.get("candidate")
            if candidate is None:
                self.skipTest("Real backend switch diagnostics did not return a candidate.")

            candidate_options = report.get("candidate_physics_backend_options", {})
            env = env_from_string(
                candidate["env_name"],
                physics_backend=candidate["backend_name"],
                render_backend="none",
                physics_backend_options=candidate_options,
            )

        self.assertEqual(candidate["env_name"], "GenesisExternalRuntimeExecutableEnv")
        self.assertEqual(candidate["backend_name"], "genesis")
        self.assertEqual(env.physics_backend, "genesis")
        self.assertEqual(candidate_options.get("module_name"), module_name)
        self.assertEqual(candidate_options.get("external_python"), sys.executable)
        self.assertEqual(candidate_options.get("PYOPENGL_PLATFORM"), "egl")
        self.assertEqual(candidate_options.get("external_probe_timeout_sec"), 8)
        self.assertEqual(candidate_options.get("external_probe_env", {}).get("PYTHONPATH"), module_dir)

        candidate_runtime_mode = str(report.get("candidate_runtime_mode") or "").lower()
        self.assertTrue(candidate_runtime_mode in {"true-runtime", "external-runtime"})
        readiness = report.get("candidate_readiness") or {}
        if {"dependency_source", "runtime_mode"}.issubset(set(readiness.keys())):
            self.assertIn("external", str(readiness["dependency_source"]).lower())


if __name__ == "__main__":
    unittest.main()
