import metaworld
import random
import numpy as np


from . import register_env

# based on Repo Master from https://github.com/rlworkgroup/metaworld on 7/23/2020 at 12:12 AM
@register_env('metaworld-benchmark-ml')
class MetaWorldWrappedEnv:
    def __init__(self, *args, **kwargs):
        ml10or45 = kwargs['ml10or45']
        if ml10or45 == 10:
            self.ml_env = metaworld.ML10()
            num_train_tasks_per_base_task = int(kwargs['n_train_tasks'] / 10)
            num_test_tasks_per_base_task = int(kwargs['n_eval_tasks'] / 5)
        elif ml10or45 == 45:
            self.ml_env = metaworld.ML45()
            num_train_tasks_per_base_task = int(kwargs['n_train_tasks'] / 45)
            num_test_tasks_per_base_task = int(kwargs['n_eval_tasks'] / 5)
        else:
            raise NotImplementedError

        self.name2number = {}
        counter = 0
        for name, env_cls in self.ml_env.train_classes.items():
            self.name2number[name] = counter
            counter += 1
        for name, env_cls in self.ml_env.test_classes.items():
            self.name2number[name] = counter
            counter += 1

        self.train_tasks = []
        for name, env_cls in self.ml_env.train_classes.items():
            tasks = random.sample([task for task in self.ml_env.train_tasks if task.env_name == name], num_train_tasks_per_base_task)
            self.train_tasks += tasks

        self.test_tasks = []
        for name, env_cls in self.ml_env.test_classes.items():
            tasks = random.sample([task for task in self.ml_env.test_tasks if task.env_name == name], num_test_tasks_per_base_task)
            self.test_tasks += tasks

        self.tasks = self.train_tasks + self.test_tasks
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        task = self.tasks[idx]
        if task.env_name in self.ml_env.train_classes:
            self.metaworld_env = self.ml_env.train_classes[task.env_name]()
        elif task.env_name in self.ml_env.test_classes:
            self.metaworld_env = self.ml_env.test_classes[task.env_name]()
        self.metaworld_env.viewer_setup = self.viewer_setup
        self.metaworld_env.set_task(task)
        self.metaworld_env.reset()
        self.active_env_name = task.env_name
        #print(task.env_name + str(self.metaworld_env._state_goal))
        self.reset()

    def set_meta_mode(self, mode):
        self.meta_mode = mode

    def step(self, action):
        ob, reward, done, info = self.metaworld_env.step(action)
        info['true_task'] = dict(base_task=self.name2number[self.active_env_name], specification=self.metaworld_env._state_goal.sum())
        return ob.astype(np.float32), reward, done, info

    def reset(self):
        a = self.metaworld_env.reset()
        return a.astype(np.float32)

    def viewer_setup(self):
        self.metaworld_env.viewer.cam.azimuth = -20
        self.metaworld_env.viewer.cam.elevation = -20

    def __getattr__(self, attrname):
        return getattr(self.metaworld_env, attrname)
