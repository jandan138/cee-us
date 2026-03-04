from .backends import (
    ENV_REGISTRY,
    load_backend_plugins,
    normalize_plugin_modules,
    physics_backend_from_string,
    render_backend_from_string,
)
from .wrappers import env_wrapper_from_string


def env_from_string(env_string, wrappers=None, **env_params):
    wrappers = [] if wrappers is None else list(wrappers)

    physics_backend_name = env_params.pop("physics_backend", env_params.pop("simulator_backend", "mujoco"))
    render_backend_name = env_params.pop("render_backend", env_params.pop("renderer_backend", "native"))
    backend_plugin_modules = normalize_plugin_modules(env_params.pop("backend_plugin_modules", []))
    physics_backend_options = env_params.pop("physics_backend_options", {})
    render_backend_options = env_params.pop("render_backend_options", {})

    load_backend_plugins(backend_plugin_modules)

    physics_backend = physics_backend_from_string(physics_backend_name)
    physics_backend.configure(physics_backend_options)
    env = physics_backend.create_env(
        env_string=env_string,
        env_registry=ENV_REGISTRY,
        env_params=env_params,
    )

    render_backend = render_backend_from_string(render_backend_name)
    render_backend.attach(env, options=render_backend_options)

    for env_wrapper in wrappers:
        env = env_wrapper_from_string(
            wrapper_string=env_wrapper["env_wrapper"],
            env=env,
            **env_wrapper["env_wrapper_params"],
        )

    if not hasattr(env, "init_kwargs"):
        env.init_kwargs = {}
    env.init_kwargs["wrappers"] = wrappers
    env.init_kwargs["physics_backend"] = env.physics_backend
    env.init_kwargs["render_backend"] = env.render_backend
    env.init_kwargs["backend_plugin_modules"] = backend_plugin_modules
    env.init_kwargs["physics_backend_options"] = physics_backend_options
    env.init_kwargs["render_backend_options"] = render_backend_options

    return env
