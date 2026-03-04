import copy
import importlib
import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import unittest

from mbrl.environments import env_from_string
from mbrl.environments.backends import (
    ENV_REGISTRY,
    available_env_backends,
    diagnose_real_backend_switch_test,
    dispatch_render,
    load_backend_plugins,
    normalize_plugin_modules,
    physics_backend_readiness,
    register_env_backend,
    reset_loaded_backend_plugins,
)
from mbrl.environments.backends import physics as physics_backend_module
from mbrl.environments.backends import render as render_backend_module


class EnvironmentBackendsTestCase(unittest.TestCase):
    def setUp(self):
        self._registry_snapshot = copy.deepcopy(ENV_REGISTRY)
        self._physics_backends_snapshot = copy.deepcopy(physics_backend_module._PHYSICS_BACKENDS)
        self._physics_aliases_snapshot = copy.deepcopy(physics_backend_module._PHYSICS_ALIASES)
        self._render_backends_snapshot = copy.deepcopy(render_backend_module._RENDER_BACKENDS)
        self._render_aliases_snapshot = copy.deepcopy(render_backend_module._RENDER_ALIASES)
        reset_loaded_backend_plugins()
        plugin_module = importlib.import_module("tests.backend_plugin_example")
        plugin_module.reset_register_call_count()

    def tearDown(self):
        ENV_REGISTRY.clear()
        ENV_REGISTRY.update(self._registry_snapshot)
        physics_backend_module._PHYSICS_BACKENDS.clear()
        physics_backend_module._PHYSICS_BACKENDS.update(self._physics_backends_snapshot)
        physics_backend_module._PHYSICS_ALIASES.clear()
        physics_backend_module._PHYSICS_ALIASES.update(self._physics_aliases_snapshot)
        render_backend_module._RENDER_BACKENDS.clear()
        render_backend_module._RENDER_BACKENDS.update(self._render_backends_snapshot)
        render_backend_module._RENDER_ALIASES.clear()
        render_backend_module._RENDER_ALIASES.update(self._render_aliases_snapshot)
        reset_loaded_backend_plugins()

    def _require_real_backend_switch_candidate(self, *, backend_plugin_modules=None, options_by_backend=None):
        report = diagnose_real_backend_switch_test(
            backend_plugin_modules=backend_plugin_modules,
            options_by_backend=options_by_backend,
        )
        if report["would_skip"]:
            self.skipTest(report["first_skip_reason"])

        candidate = report["candidate"]
        if candidate is None:
            self.skipTest("Real backend switch diagnostics did not return a candidate.")
        return (
            candidate["env_name"],
            candidate["backend_name"],
            report.get("candidate_physics_backend_options", {}),
        )

    def test_register_env_backend_and_overwrite(self):
        register_env_backend(
            "DummyBackendEnv",
            "mujoco",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
        )
        self.assertEqual(available_env_backends("DummyBackendEnv"), ["mujoco"])

        with self.assertRaises(KeyError):
            register_env_backend(
                "DummyBackendEnv",
                "mujoco",
                "mbrl.environments.testsupport_dummy_env",
                "OtherDummyTestEnv",
            )

        register_env_backend(
            "DummyBackendEnv",
            "mujoco",
            "mbrl.environments.testsupport_dummy_env",
            "OtherDummyTestEnv",
            overwrite=True,
        )
        self.assertEqual(
            ENV_REGISTRY["DummyBackendEnv"]["mujoco"],
            ("mbrl.environments.testsupport_dummy_env", "OtherDummyTestEnv"),
        )

    def test_env_factory_accepts_alias_keys(self):
        register_env_backend(
            "DummyFactoryEnv",
            "mujoco",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
        )
        env = env_from_string(
            "DummyFactoryEnv",
            simulator_backend="mujoco",
            renderer_backend="native",
            seed_value=7,
        )

        self.assertEqual(env.name, "DummyFactoryEnv")
        self.assertEqual(env.physics_backend, "mujoco")
        self.assertEqual(env.render_backend, "native")
        self.assertEqual(env.seed_value, 7)
        self.assertEqual(env.init_kwargs["physics_backend"], "mujoco")
        self.assertEqual(env.init_kwargs["render_backend"], "native")
        self.assertIn("wrappers", env.init_kwargs)

    def test_render_dispatch_headless_and_none(self):
        register_env_backend(
            "DummyRenderEnv",
            "mujoco",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
        )

        headless_env = env_from_string("DummyRenderEnv", render_backend="headless")
        frame = dispatch_render(headless_env, width=12, height=6, camera_name="cam0")
        self.assertEqual(frame.shape, (6, 12, 3))

        none_env = env_from_string("DummyRenderEnv", render_backend="none")
        self.assertIsNone(dispatch_render(none_env))
        with self.assertRaises(RuntimeError):
            dispatch_render(none_env, mode="rgb_array")

    def test_unavailable_backend_error_guides_registration(self):
        register_env_backend(
            "DummyFutureEnv",
            "mujoco",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
        )
        with self.assertRaises(NotImplementedError) as error:
            env_from_string("DummyFutureEnv", physics_backend="isaacsim")
        self.assertIn("register_env_backend", str(error.exception))

    def test_plugin_module_loading_registers_backends(self):
        env = env_from_string(
            "PluginEnv",
            physics_backend="plugin",
            render_backend="plugin_renderer",
            backend_plugin_modules=["tests.backend_plugin_example"],
            seed_value=11,
        )
        render_result = dispatch_render(env, mode="human", width=5, height=4, camera_name="cam1")

        self.assertEqual(env.physics_backend, "pluginphysics")
        self.assertEqual(env.render_backend, "pluginrender")
        self.assertEqual(render_result["backend"], "pluginrender")
        self.assertEqual(render_result["camera_name"], "cam1")
        self.assertEqual(env.init_kwargs["backend_plugin_modules"], ["tests.backend_plugin_example"])

    def test_normalize_plugin_modules(self):
        self.assertEqual(
            normalize_plugin_modules([" tests.backend_plugin_example ", "tests.backend_plugin_example", ""]),
            ["tests.backend_plugin_example"],
        )
        self.assertEqual(normalize_plugin_modules("tests.backend_plugin_example"), ["tests.backend_plugin_example"])
        with self.assertRaises(TypeError):
            normalize_plugin_modules({"module": "tests.backend_plugin_example"})

    def test_plugin_loader_registers_once(self):
        plugin_module = importlib.import_module("tests.backend_plugin_example")

        load_backend_plugins(["tests.backend_plugin_example"])
        load_backend_plugins(["tests.backend_plugin_example"])

        self.assertEqual(plugin_module.REGISTER_CALL_COUNT, 1)

    def test_plugin_loader_thread_safe_single_registration(self):
        plugin_module = importlib.import_module("tests.backend_plugin_example")

        def _load(_):
            load_backend_plugins(["tests.backend_plugin_example"])

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_load, range(32)))

        self.assertEqual(plugin_module.REGISTER_CALL_COUNT, 1)

    def test_plugin_loader_import_error(self):
        with self.assertRaises(ImportError) as error:
            load_backend_plugins(["tests.module_does_not_exist_for_backend_plugin"])
        self.assertIn("Failed to import backend plugin module", str(error.exception))

    def test_real_backend_switch_when_enabled(self):
        env_name, backend_name, backend_options = self._require_real_backend_switch_candidate()
        env = env_from_string(
            env_name,
            physics_backend=backend_name,
            render_backend="none",
            physics_backend_options=backend_options,
        )
        self.assertEqual(env.physics_backend, backend_name)

    def test_real_backend_switch_skip_reason_reports_genesis_readiness_failure(self):
        register_env_backend(
            "GenesisReadinessSkipEnv",
            "genesis",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
            overwrite=True,
        )
        readiness_failure = {
            "backend": "genesis",
            "ready": False,
            "error_type": "ImportError",
            "reason": "Physics backend 'Genesis' selected, but the 'genesis' package is not available.",
        }
        backend_info = {
            "mujoco": {"implemented": True},
            "genesis": {"implemented": True},
        }
        with patch.dict(os.environ, {"ENABLE_REAL_BACKEND_TESTS": "1"}, clear=False):
            with patch("mbrl.environments.backends.diagnostics.list_physics_backends", return_value=backend_info):
                with patch(
                    "mbrl.environments.backends.diagnostics.physics_backend_readiness",
                    return_value=readiness_failure,
                ):
                    with self.assertRaises(unittest.SkipTest) as skipped:
                        self._require_real_backend_switch_candidate()

        skip_reason = str(skipped.exception)
        self.assertIn("No ready non-MuJoCo backend candidate could be selected", skip_reason)
        self.assertIn("readiness failures:", skip_reason)
        self.assertIn("genesis", skip_reason)
        self.assertIn("ImportError", skip_reason)
        self.assertIn("not available", skip_reason)

    def test_real_backend_switch_candidate_discovery_loads_plugin_from_env_var(self):
        with patch.dict(
            os.environ,
            {
                "ENABLE_REAL_BACKEND_TESTS": "1",
                "REAL_BACKEND_SWITCH_PLUGIN_MODULES": "tests.backend_plugin_example",
            },
            clear=False,
        ):
            env_name, backend_name, backend_options = self._require_real_backend_switch_candidate()

        self.assertEqual(env_name, "PluginEnv")
        self.assertEqual(backend_name, "pluginphysics")
        self.assertEqual(backend_options, {})

    def test_real_backend_switch_candidate_discovery_accepts_explicit_plugin_modules(self):
        with patch.dict(os.environ, {"ENABLE_REAL_BACKEND_TESTS": "1"}, clear=False):
            env_name, backend_name, backend_options = self._require_real_backend_switch_candidate(
                backend_plugin_modules=["tests.backend_plugin_example"]
            )

        self.assertEqual(env_name, "PluginEnv")
        self.assertEqual(backend_name, "pluginphysics")
        self.assertEqual(backend_options, {})

    def test_real_backend_switch_env_construction_uses_backend_options_from_env_var(self):
        register_env_backend(
            "GenesisRealSwitchOptionsEnv",
            "genesis",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
            overwrite=True,
        )
        with patch.object(physics_backend_module.GenesisPhysicsBackend, "_is_genesis_available", return_value=False):
            with patch.dict(
                os.environ,
                {
                    "ENABLE_REAL_BACKEND_TESTS": "1",
                    "REAL_BACKEND_SWITCH_PHYSICS_OPTIONS_JSON": "{\"genesis\": {\"skip_dependency_check\": true}}",
                },
                clear=False,
            ):
                env_name, backend_name, backend_options = self._require_real_backend_switch_candidate()
                env = env_from_string(
                    env_name,
                    physics_backend=backend_name,
                    render_backend="none",
                    physics_backend_options=backend_options,
                )

        self.assertEqual(env_name, "GenesisRealSwitchOptionsEnv")
        self.assertEqual(backend_name, "genesis")
        self.assertEqual(backend_options, {"skip_dependency_check": True})
        self.assertEqual(env.physics_backend, "genesis")

    def test_genesis_backend_can_switch_when_mapped_and_dependency_check_skipped(self):
        register_env_backend(
            "GenesisDummyEnv",
            "genesis",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
            overwrite=True,
        )
        env = env_from_string(
            "GenesisDummyEnv",
            physics_backend="genesis",
            render_backend="none",
            physics_backend_options={"skip_dependency_check": True},
        )
        self.assertEqual(env.physics_backend, "genesis")

    def test_genesis_backend_requires_dependency_by_default(self):
        register_env_backend(
            "GenesisDependencyEnv",
            "genesis",
            "mbrl.environments.testsupport_dummy_env",
            "DummyTestEnv",
            overwrite=True,
        )
        with self.assertRaises(ImportError) as error:
            env_from_string(
                "GenesisDependencyEnv",
                physics_backend="genesis",
                render_backend="none",
            )
        self.assertIn("Genesis", str(error.exception))

    def test_physics_backend_readiness_reports_genesis_missing_dependency(self):
        with patch.object(physics_backend_module.GenesisPhysicsBackend, "_is_genesis_available", return_value=False):
            readiness = physics_backend_readiness("genesis")

        self.assertFalse(readiness["ready"])
        self.assertEqual(readiness["backend"], "genesis")
        self.assertEqual(readiness["error_type"], "ImportError")
        self.assertIn("Genesis", readiness["reason"])

    def test_physics_backend_readiness_can_skip_genesis_dependency_check(self):
        with patch.object(physics_backend_module.GenesisPhysicsBackend, "_is_genesis_available", return_value=False):
            readiness = physics_backend_readiness("genesis", options={"skip_dependency_check": True})

        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["backend"], "genesis")
        self.assertEqual(readiness["error_type"], None)


if __name__ == "__main__":
    unittest.main()
