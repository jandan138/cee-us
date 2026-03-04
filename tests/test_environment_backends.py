import copy
import unittest

from mbrl.environments import env_from_string
from mbrl.environments.backends import (
    ENV_REGISTRY,
    available_env_backends,
    dispatch_render,
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


if __name__ == "__main__":
    unittest.main()
