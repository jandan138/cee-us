from mbrl.environments.backends import (
    register_env_backend,
    register_physics_backend,
    register_render_backend,
)
from mbrl.environments.backends.physics import PhysicsBackend
from mbrl.environments.backends.render import RenderBackend


class DummyPluginPhysicsBackend(PhysicsBackend):
    backend_name = "dummyplugin"
    display_name = "Dummy Plugin Physics"
    implemented = True


class DummyPluginRenderBackend(RenderBackend):
    backend_name = "dummypluginrender"
    display_name = "Dummy Plugin Renderer"
    implemented = True

    def render(self, env, *, mode=None, width=None, height=None, camera_name=None, **kwargs):
        return {
            "plugin_backend": self.backend_name,
            "mode": mode or "human",
            "width": width,
            "height": height,
            "camera_name": camera_name,
            "extra": kwargs,
        }


def register_backends():
    register_physics_backend(
        "dummyplugin",
        DummyPluginPhysicsBackend,
        aliases=["dp"],
        override=True,
    )
    register_render_backend(
        "dummypluginrender",
        DummyPluginRenderBackend,
        aliases=["dpr"],
        override=True,
    )
    register_env_backend(
        "DummyPluginEnv",
        "dummyplugin",
        "mbrl.environments.testsupport_dummy_env",
        "DummyTestEnv",
        overwrite=True,
    )
