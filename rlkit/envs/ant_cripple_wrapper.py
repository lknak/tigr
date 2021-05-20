import numpy as np
from meta_rand_envs.ant_cripple import AntCrippleEnv

from . import register_env


@register_env('ant-cripple')
class AntCrippleWrappedEnv(AntCrippleEnv):
    def __init__(self, n_tasks=2, randomize_tasks=True, hfield_mode='gentle', log_scale_limit=3.0, change_prob=0.01):
        super(AntCrippleWrappedEnv, self).__init__(log_scale_limit=log_scale_limit, mode=hfield_mode, change_prob=change_prob)
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()