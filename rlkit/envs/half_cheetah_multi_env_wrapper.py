import numpy as np
from meta_rand_envs.half_cheetah_multi_env import HalfCheetahMixtureEnv

from . import register_env


@register_env('cheetah-changing-multi-env')
class HalfCheetahMultiEnvWrappedEnv(HalfCheetahMixtureEnv):
    def __init__(self, *args, **kwargs):
        super(HalfCheetahMultiEnvWrappedEnv, self).__init__(*args, **kwargs)
        self.train_tasks = self.sample_tasks(kwargs['n_train_tasks'])
        self.test_tasks = self.sample_tasks(kwargs['n_eval_tasks'])
        self.tasks = self.train_tasks + self.test_tasks
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self.base_task = self._task['base_task']
        self.task_specification = self._task['specification']
        self.color = self._task['color']
        self.recolor()
        self.reset()
