ENV_REGISTRY = {
    # - PLAYGROUND - #
    "PlaygroundwGoals": {
        "mujoco": (".playground_env_wgoals", "PlaygroundwGoals"),
    },
    # - CONSTRUCTION - #
    "FetchPickAndPlace": {
        "mujoco": (".robotics", "FetchPickAndPlace"),
    },
    "FetchReach": {
        "mujoco": (".robotics", "FetchReach"),
    },
    "FetchPickAndPlaceConstruction": {
        "mujoco": (".fpp_construction_env", "FetchPickAndPlaceConstruction"),
    },
    # - ROBODESK - #
    "Robodesk": {
        "mujoco": (".robodesk_env", "Robodesk"),
    },
    "RobodeskFlat": {
        "mujoco": (".robodesk_env", "RobodeskFlat"),
    },
}


def register_env_backend(env_name, physics_backend, module_path, class_name, *, overwrite=False):
    if not env_name:
        raise ValueError("env_name must be non-empty")
    if not physics_backend:
        raise ValueError("physics_backend must be non-empty")
    if not module_path or not class_name:
        raise ValueError("module_path and class_name must be non-empty")

    backend_name = physics_backend.strip().lower()
    backend_entries = ENV_REGISTRY.setdefault(env_name, {})
    mapping = (module_path, class_name)

    if not overwrite and backend_name in backend_entries and backend_entries[backend_name] != mapping:
        raise KeyError(
            f"Env '{env_name}' already has backend '{backend_name}' mapped to {backend_entries[backend_name]}"
        )

    backend_entries[backend_name] = mapping


def available_env_backends(env_name):
    if env_name not in ENV_REGISTRY:
        return []
    return sorted(ENV_REGISTRY[env_name].keys())
