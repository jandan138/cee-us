import copy
import importlib
import os
from concurrent.futures import ThreadPoolExecutor
import unittest

from mbrl.environments import env_from_string
from mbrl.environments.backends import (
    ENV_REGISTRY,
    available_env_backends,
    dispatch_render,
    list_physics_backends,
    load_backend_plugins,
    normalize_plugin_modules,
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
        if os.environ.get("ENABLE_REAL_BACKEND_TESTS", "0") != "1":
            self.skipTest("Set ENABLE_REAL_BACKEND_TESTS=1 to run real non-MuJoCo backend switch tests.")

        implemented_non_mujoco = [
            name for name, info in list_physics_backends().items() if info["implemented"] and name != "mujoco"
        ]
        if not implemented_non_mujoco:
            self.skipTest("No implemented non-MuJoCo physics backend is registered yet.")

        candidate = None
        for env_name, env_backends in ENV_REGISTRY.items():
            for backend_name in implemented_non_mujoco:
                if backend_name in env_backends:
                    candidate = (env_name, backend_name)
                    break
            if candidate is not None:
                break

        if candidate is None:
            self.skipTest("No ENV_REGISTRY entry exists for implemented non-MuJoCo backends.")

        env_name, backend_name = candidate
        env = env_from_string(env_name, physics_backend=backend_name, render_backend="none")
        self.assertEqual(env.physics_backend, backend_name)


if __name__ == "__main__":
    unittest.main()
