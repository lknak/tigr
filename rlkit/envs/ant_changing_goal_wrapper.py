import numpy as np
from meta_rand_envs.ant_changing_goal import AntChangingGoalEnv

from . import register_env


@register_env('ant-changing-goal')
class AntChangingGoalWrappedEnv(AntChangingGoalEnv):
    def __init__(self, *args, **kwargs):
        super(AntChangingGoalWrappedEnv, self).__init__(*args, **kwargs)
        self.tasks = self.sample_tasks(kwargs['n_tasks'])
        self.train_tasks = self.tasks[:kwargs['n_train_tasks']]
        self.test_tasks = self.tasks[kwargs['n_train_tasks']:]
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self.goal = self._task
        self.recolor()
        self.reset()
