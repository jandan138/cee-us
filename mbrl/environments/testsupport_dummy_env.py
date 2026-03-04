class DummyTestEnv:
    supports_live_rendering = True

    def __init__(self, *, name, seed_value=0, **kwargs):
        self.name = name
        self.seed_value = seed_value
        self.init_kwargs = {"seed_value": seed_value, **kwargs}
        self.render_calls = []
        self.physics_backend = None
        self.render_backend = None

    def reset(self):
        return 0

    def reset_with_mode(self, mode):
        return self.reset()

    def step(self, action):
        return 0, 0.0, False, {}

    def render(self, mode="human", render_width=None, render_height=None, camera_name=None):
        call = {
            "mode": mode,
            "render_width": render_width,
            "render_height": render_height,
            "camera_name": camera_name,
        }
        self.render_calls.append(call)
        if mode == "rgb_array":
            width = 64 if render_width is None else int(render_width)
            height = 48 if render_height is None else int(render_height)
            return DummyFrame(height, width)
        return call

    def get_fps(self):
        return 30

    def close(self):
        return None


class OtherDummyTestEnv(DummyTestEnv):
    pass


class DummyFrame:
    def __init__(self, height, width):
        self.shape = (height, width, 3)
