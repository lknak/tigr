from meta_rand_envs.walker2d_non_stationary_rand_params import Walker2DNonStationaryRandParamEnv
from . import register_env


@register_env('walker2d-non-stationary-rand-params')
class Walker2DNonStationaryRandParamWrappedEnv(Walker2DNonStationaryRandParamEnv):
    def __init__(self, *args, **kwargs):
        Walker2DNonStationaryRandParamEnv.__init__(self, *args, **kwargs)
