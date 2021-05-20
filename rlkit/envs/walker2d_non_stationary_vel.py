from meta_rand_envs.walker2d_non_stationary_velocity import Walker2DNonStationaryVelocityEnv
from . import register_env


@register_env('walker2d-non-stationary-vel')
class Walker2DNonStationaryVelWrappedEnv(Walker2DNonStationaryVelocityEnv):
    def __init__(self, *args, **kwargs):
        Walker2DNonStationaryVelocityEnv.__init__(self, *args, **kwargs)
