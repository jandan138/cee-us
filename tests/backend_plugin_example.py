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


register_physics_backend("pluginphysics", PluginPhysicsBackend, aliases=["plugin"], override=False)
register_render_backend("pluginrender", PluginRenderBackend, aliases=["plugin_renderer"], override=False)
register_env_backend(
    "PluginEnv",
    "pluginphysics",
    "mbrl.environments.testsupport_dummy_env",
    "DummyTestEnv",
    overwrite=False,
)
