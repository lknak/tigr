from meta_rand_envs.hopper_non_stationary_velocity import HopperNonStationaryVelocityEnv
from . import register_env


@register_env('hopper-non-stationary-vel')
class HopperNonStationaryVelWrappedEnv(HopperNonStationaryVelocityEnv):
    def __init__(self, *args, **kwargs):
        HopperNonStationaryVelocityEnv.__init__(self, *args, **kwargs)
