from meta_rand_envs.half_cheetah_non_stationary_direction import HalfCheetahNonStationaryDirectionEnv
from . import register_env

@register_env('cheetah-stationary-dir')
@register_env('cheetah-non-stationary-dir')
@register_env('cheetah-continuous-learning-dir')
class HalfCheetahNonStationaryDirWrappedEnv(HalfCheetahNonStationaryDirectionEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryDirectionEnv.__init__(self, *args, **kwargs)
