from meta_rand_envs.ant_non_stationary_velocity import AntNonStationaryVelocityEnv
from . import register_env


@register_env('ant-non-stationary-vel')
class AntNonStationaryVelWrappedEnv(AntNonStationaryVelocityEnv):
    def __init__(self, *args, **kwargs):
        AntNonStationaryVelocityEnv.__init__(self, *args, **kwargs)
