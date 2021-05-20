from meta_rand_envs.half_cheetah_non_stationary_rand_mass_params import HalfCheetahNonStationaryRandMassParamEnv
from . import register_env


@register_env('cheetah-non-stationary-rand-mass-params')
class HalfCheetahNonStationaryRandMassParamWrappedEnv(HalfCheetahNonStationaryRandMassParamEnv):
    def __init__(self, *args, **kwargs):
        HalfCheetahNonStationaryRandMassParamEnv.__init__(self, *args, **kwargs)
