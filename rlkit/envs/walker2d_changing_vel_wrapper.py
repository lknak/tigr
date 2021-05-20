import numpy as np
from meta_rand_envs.walker2d_changing_vel import Walker2DChangingVelEnv

from . import register_env


@register_env('walker2d-changing-vel')
class Walker2DChangingVelWrappedEnv(Walker2DChangingVelEnv):
    def __init__(self, *args, **kwargs):
        super(Walker2DChangingVelWrappedEnv, self).__init__(*args, **kwargs)
        self.tasks = self.sample_tasks(kwargs['n_tasks'])
        self.train_tasks = self.tasks[:kwargs['n_train_tasks']]
        self.test_tasks = self.tasks[kwargs['n_train_tasks']:]
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self.goal_velocity = self._task['velocity']
        self.recolor()
        self.reset()
