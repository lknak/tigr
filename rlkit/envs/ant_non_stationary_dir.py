from meta_rand_envs.ant_non_stationary_direction import AntNonStationaryDirectionEnv
from . import register_env


@register_env('ant-non-stationary-dir')
class HalfCheetahNonStationaryDirWrappedEnv(AntNonStationaryDirectionEnv):
    def __init__(self, *args, **kwargs):
        AntNonStationaryDirectionEnv.__init__(self, *args, **kwargs)
