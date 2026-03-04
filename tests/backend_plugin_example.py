from mbrl.environments.backends import (
    register_env_backend,
    register_physics_backend,
    register_render_backend,
)
from mbrl.environments.backends.physics import PhysicsBackend
from mbrl.environments.backends.render import RenderBackend


class PluginPhysicsBackend(PhysicsBackend):
    backend_name = "pluginphysics"
    display_name = "Plugin Physics"
    implemented = True


class PluginRenderBackend(RenderBackend):
    backend_name = "pluginrender"
    display_name = "Plugin Render"
    implemented = True

    def render(self, env, *, mode=None, width=None, height=None, camera_name=None, **kwargs):
        return {
            "backend": self.backend_name,
            "mode": mode,
            "width": width,
            "height": height,
            "camera_name": camera_name,
            "extra": kwargs,
        }


REGISTER_CALL_COUNT = 0


def register_backends():
    global REGISTER_CALL_COUNT
    REGISTER_CALL_COUNT += 1

    register_physics_backend("pluginphysics", PluginPhysicsBackend, aliases=["plugin"], override=True)
    register_render_backend("pluginrender", PluginRenderBackend, aliases=["plugin_renderer"], override=True)
    register_env_backend(
        "PluginEnv",
        "pluginphysics",
        "mbrl.environments.testsupport_dummy_env",
        "DummyTestEnv",
        overwrite=True,
    )


def reset_register_call_count():
    global REGISTER_CALL_COUNT
    REGISTER_CALL_COUNT = 0
