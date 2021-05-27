import torch
from torch import nn as nn
from rlkit.torch.networks import Mlp


class DecoderMDP(nn.Module):
    '''
    Uses data (state, action, reward, task_hypothesis z) from the replay buffer or online
    and computes estimates for the next state and reward.
    Through that it reconstructs the MDP and gives gradients back to the task hypothesis.
    '''
    def __init__(self,
                 action_dim,
                 state_dim,
                 reward_dim,
                 z_dim,
                 net_complex,
                 state_reconstruction_clip
    ):
        super(DecoderMDP, self).__init__()

        self.state_decoder_input_size = state_dim + action_dim + z_dim
        self.state_decoder_hidden_size = int(self.state_decoder_input_size * net_complex)

        self.state_reconstruction_clip = state_reconstruction_clip if state_reconstruction_clip is not None and 0 < state_reconstruction_clip < state_dim else state_dim

        self.reward_decoder_input_size = state_dim + action_dim + z_dim
        self.reward_decoder_hidden_size = int(self.reward_decoder_input_size * net_complex)

        self.net_state_decoder = Mlp(
            input_size=self.state_decoder_input_size,
            hidden_sizes=[self.state_decoder_hidden_size, self.state_decoder_hidden_size],
            output_size=self.state_reconstruction_clip
        )
        self.net_reward_decoder = Mlp(
            input_size=self.reward_decoder_input_size,
            hidden_sizes=[self.reward_decoder_hidden_size, self.reward_decoder_hidden_size],
            output_size=reward_dim
        )

    def forward(self, state, action, next_state, z):
        state_estimate = self.net_state_decoder(torch.cat([state, action, z], dim=-1))
        reward_estimate = self.net_reward_decoder(torch.cat([state, action, z], dim=-1))

        return state_estimate, reward_estimate
