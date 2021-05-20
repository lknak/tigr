from meta_rand_envs.metaworld import MetaWorldEnv
from . import register_env
from gym import utils
import mujoco_py
import numpy as np


@register_env('metaworld')
@register_env('metaworld-ml10')
@register_env('metaworld-ml45')
class MetaWorldWrappedEnv(MetaWorldEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):

        MetaWorldEnv.__init__(self, *args, **kwargs)

        self.env_buffer = {}

        utils.EzPickle.__init__(self, *args, **kwargs)

    def reset_task(self, idx, keep_buffered=False):

        # Close window to avoid multiple windows open at once
        if hasattr(self, 'viewer'):
            self.close()

        task = self.tasks[int(idx)]
        if keep_buffered and idx in self.env_buffer.keys():
            self.metaworld_env = self.env_buffer[idx]
        else:
            if task.env_name in self.ml_env.train_classes:
                self.metaworld_env = self.ml_env.train_classes[task.env_name]()
            elif task.env_name in self.ml_env.test_classes:
                self.metaworld_env = self.ml_env.test_classes[task.env_name]()
            if keep_buffered: self.env_buffer[idx] = self.metaworld_env

        self.metaworld_env.viewer_setup = self.viewer_setup
        self.metaworld_env.set_task(task)
        self.metaworld_env.reset()
        self.active_env_name = task.env_name

        self.reset()

    def set_task(self, idx):
        assert idx in self.env_buffer.keys()

        # close window to avoid mulitple windows open at once
        if hasattr(self, 'viewer'):
            self.close()

        self.metaworld_env = self.env_buffer[idx]
        self.metaworld_env.viewer_setup = self.viewer_setup
        self.active_env_name = self.tasks[int(idx)].env_name

    def clear_buffer(self):
        self.env_buffer = {}

    def viewer_setup(self):
        self.viewer.cam.azimuth = -185
        self.viewer.cam.elevation = -10
        self.viewer.cam.distance = 1.7
        self.viewer.cam.lookat[1] = 0.5

    def render(self, mode='rgb_array', width=512, height=512, camera_id=-1):
        if mode == 'human':
            self.viewer = self._viewers.get(mode)
            self.viewer.render()
        elif mode == 'rgb_array':
            self.viewer = self._viewers.get(mode)
            if self.viewer is None:
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, camera_id)
                self._viewers[mode] = self.viewer
            self.viewer_setup()

            self.viewer.render(width, height)
            return np.asarray(self.viewer.read_pixels(width, height, depth=False)[::-1, :, :], dtype=np.uint8)
        else:
            raise ValueError("mode can only be either 'human' or 'rgb_array'")