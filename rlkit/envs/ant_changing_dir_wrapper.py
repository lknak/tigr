import numpy as np
from meta_rand_envs.ant_changing_dir import AntChangingDirEnv

from . import register_env


@register_env('ant-changing-dir')
class AntChangingDirWrappedEnv(AntChangingDirEnv):
    def __init__(self, n_tasks=2, *args, **kwargs):
        super(AntChangingDirWrappedEnv, self).__init__(*args, **kwargs)
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self.goal_direction_start = self._task['direction']
        self.reset()