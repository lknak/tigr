from meta_rand_envs.half_cheetah_non_stationary_multi_task import HalfCheetahNonStationaryMultiTaskEnv
from . import register_env


@register_env('cheetah-non-stationary-multi-task')
@register_env('cheetah-stationary-multi-task')
class HalfCheetahNonStationaryMultiTaskWrappedEnv(HalfCheetahNonStationaryMultiTaskEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryMultiTaskEnv.__init__(self, *args, **kwargs)
