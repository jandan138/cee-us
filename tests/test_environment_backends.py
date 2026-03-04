import copy
import unittest

from mbrl.environments import env_from_string
from mbrl.environments.backends import (
    ENV_REGISTRY,
    available_env_backends,
    dispatch_render,
    normalize_plugin_modules,
    register_env_backend,
)


class EnvironmentBackendsTestCase(unittest.TestCase):
    def setUp(self):
        self._registry_snapshot = copy.deepcopy(ENV_REGISTRY)

    def tearDown(self):
        ENV_REGISTRY.clear()
        ENV_REGISTRY.update(self._registry_snapshot)

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


if __name__ == "__main__":
    unittest.main()
