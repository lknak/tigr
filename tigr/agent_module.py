import torch
import torch.nn as nn
import numpy as np
from rlkit.torch.core import np_ify
import rlkit.torch.pytorch_util as ptu

from tigr.scripted_policies import policies

class Agent(nn.Module):
    def __init__(self,
                 encoder,
                 policy
                 ):
        super(Agent, self).__init__()
        self.encoder = encoder
        self.policy = policy

    def get_action(self, encoder_input, state, deterministic=False, z_debug=None, env=None):
        state = ptu.from_numpy(state).view(1, -1)
        z, _ = self.encoder(encoder_input)
        if z_debug is not None:
            z = z_debug
        policy_input = torch.cat([state, z], dim=1)
        return self.policy.get_action(policy_input, deterministic=deterministic), np_ify(z.clone().detach())[0, :]

    def get_actions(self, encoder_input, state, deterministic=False):
        z, _ = self.encoder(encoder_input)
        policy_input = torch.cat([state, z], dim=-1)

        return (self.policy.get_actions(policy_input, deterministic=deterministic), [{}] * state.shape[0]), np_ify(z)

class ScriptedPolicyAgent(nn.Module):
    def __init__(self,
                 encoder,
                 policy
                 ):
        super(ScriptedPolicyAgent, self).__init__()
        self.encoder = encoder
        self.policy = policy
        self.latent_dim = encoder.latent_dim

    def get_action(self, encoder_input, state, deterministic=False, z_debug=None, env=None):
        env_name = env.active_env_name
        oracle_policy = policies[env_name]()
        action = oracle_policy.get_action(state)
        return (action.astype('float32'), {}), np.zeros(self.latent_dim, dtype='float32')

