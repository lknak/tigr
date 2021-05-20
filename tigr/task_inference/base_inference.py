from torch import nn as nn
from rlkit.torch.networks import Mlp
from tigr.task_inference.improved_encoder import GruAttentionEncoder, ConvAttentionEncoder, TransformerModel

class DecoupledEncoder(nn.Module):
    def __init__(self,
                 shared_dim,
                 encoder_input_dim,
                 latent_dim,
                 num_classes,
                 time_steps,
                 encoding_mode,
                 timestep_combination,
                 encoder_type='mlp'
                 ):
        super(DecoupledEncoder, self).__init__()
        self.shared_dim = shared_dim
        self.encoder_input_dim = encoder_input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.encoding_mode = encoding_mode
        self.timestep_combination = timestep_combination

        self.encoder_type = encoder_type

        self.sigma_ops = 'softplus'

        # Define shared encoder to extract features
        if encoder_type == 'mlp':
            self.shared_encoder = Mlp(input_size=self.encoder_input_dim,
                                      hidden_sizes=[self.shared_dim, self.shared_dim],
                                      output_size=self.shared_dim)
            # TODO: Test batchnorm
            self.shared_encoder = nn.Sequential(
                # nn.BatchNorm1d(self.encoder_input_dim),
                self.shared_encoder
            )
        elif encoder_type == 'gru':
            assert self.encoding_mode == 'trajectory'
            self.shared_encoder = GruAttentionEncoder(self.encoder_input_dim, self.shared_dim, self.time_steps, self.encoding_mode)
        elif encoder_type == 'conv':
            assert self.encoding_mode == 'transitionSharedY'
            self.shared_encoder = ConvAttentionEncoder(self.encoder_input_dim, self.shared_dim, self.time_steps, self.encoding_mode)
        elif encoder_type == 'transformer':
            assert self.encoding_mode == 'transitionSharedY'
            self.shared_encoder = TransformerModel(self.encoder_input_dim, self.shared_dim, self.time_steps, self.encoding_mode)
        else:
            raise NotImplementedError(f'Encoder type "{encoder_type}" is not implemented!')

    def forward(self, x, sampler='mean'):
        raise NotImplementedError('Function "forward" must be implemented for task inference.')

    def encode(self, x):
        raise NotImplementedError('Function "forward" must be implemented for task inference.')

    def sample(self, latent_distributions, sampler='random'):
        # Select from which Gaussian to sample

        if sampler == 'random':
            # Sample from specified Gaussian using reparametrization trick
            sampled = latent_distributions.rsample()
        else:
            sampled = latent_distributions.mean

        return sampled
