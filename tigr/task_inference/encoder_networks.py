import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from tigr.utils import generate_gaussian


class ClassEncoder(nn.Module):
    def __init__(self,
                 num_classes,
                 shared_dim
    ):
        super(ClassEncoder, self).__init__()

        self.shared_dim = shared_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(self.shared_dim, self.num_classes)

    def forward(self, m):
        return F.softmax(self.linear(m), dim=-1)


class PriorPz(nn.Module):
    def __init__(self,
                 num_classes,
                 latent_dim
                 ):
        super(PriorPz, self).__init__()
        self.latent_dim = latent_dim
        # feed cluster number y as one-hot, get mu_sigma out
        self.linear = nn.Linear(num_classes, self.latent_dim * 2)

    def forward(self, m):
        return self.linear(m)


class EncoderMixtureModelTrajectory(nn.Module):
    '''
    Overall encoder network, putting a shared_encoder, class_encoder and gauss_encoder together.
    '''
    def __init__(self,
                 shared_dim,
                 encoder_input_dim,
                 latent_dim,
                 batch_size,
                 num_classes,
                 time_steps,
                 merge_mode='add'
    ):
        super(EncoderMixtureModelTrajectory, self).__init__()
        self.shared_dim = shared_dim
        self.encoder_input_dim = encoder_input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes_init = num_classes
        self.num_classes = num_classes
        self.shared_encoder = Mlp(
                                    hidden_sizes=[shared_dim, shared_dim],
                                    input_size=encoder_input_dim,
                                    output_size=shared_dim,
                                )
        self.class_encoder = ClassEncoder(self.num_classes, self.shared_dim)
        self.gauss_encoder_list = nn.ModuleList([nn.Linear(self.shared_dim, self.latent_dim * 2) for _ in range(self.num_classes)])

    def forward(self, x):
        y_distribution, z_distributions = self.encode(x)
        # TODO: could be more efficient if only compute the Gaussian layer of the y that we pick later
        return self.sample_z(y_distribution, z_distributions, y_usage="most_likely", sampler="mean")

    def encode(self, x):
        # Compute shared encoder forward pass
        m = self.shared_encoder(x)

        # Compute class encoder forward pass
        y = self.class_encoder(m)
        y_distribution = torch.distributions.categorical.Categorical(probs=y)

        # Compute every gauss_encoder forward pass
        all_mu_sigma = []
        for net in self.gauss_encoder_list:
            all_mu_sigma.append(net(m))
        z_distributions = [generate_gaussian(mu_sigma, self.latent_dim) for mu_sigma in all_mu_sigma]

        return y_distribution, z_distributions

    def sample_z(self, y_distribution, z_distributions, y_usage="specific", y=None, sampler="random"):
        # Select from which Gaussian to sample

        # Used for individual sampling when computing ELBO
        if y_usage == "specific":
            y = ptu.ones(self.batch_size, dtype=torch.long) * y
        # Used while inference
        elif y_usage == "most_likely":
            y = torch.argmax(y_distribution.probs, dim=1)
        else:
            raise RuntimeError("Sampling strategy not specified correctly")

        mask = y.view(-1, 1).unsqueeze(2).repeat(1, 1, self.latent_dim)

        if sampler == "random":
            # Sample from specified Gaussian using reparametrization trick
            # this operation samples from each Gaussian for every class first
            # (diag_embed not possible for distributions), put it back to tensor with shape [class, batch, latent]
            sampled = torch.cat([torch.unsqueeze(z_distributions[i].rsample(), 0) for i in range(self.num_classes)], dim=0)

        elif sampler == "mean":
            sampled = torch.cat([torch.unsqueeze(z_distributions[i].mean, 0) for i in range(self.num_classes)], dim=0)

        # tensor with shape [batch, class, latent]
        permute = sampled.permute(1, 0, 2)
        z = torch.squeeze(torch.gather(permute, 1, mask), 1)
        return z, y


class EncoderMixtureModelTransitionSharedY(nn.Module):
    '''
    Overall encoder network, putting a shared_encoder, class_encoder and gauss_encoder together.
    '''
    def __init__(self,
                 shared_dim,
                 encoder_input_dim,
                 latent_dim,
                 batch_size,
                 num_classes,
                 time_steps,
                 merge_mode='add'
    ):
        super(EncoderMixtureModelTransitionSharedY, self).__init__()
        self.shared_dim = shared_dim
        self.encoder_input_dim = encoder_input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes_init = num_classes
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.merge_mode = merge_mode
        self.shared_encoder = Mlp(
                                    hidden_sizes=[shared_dim, shared_dim],
                                    input_size=encoder_input_dim,
                                    output_size=shared_dim,
                                )
        if self.merge_mode == 'linear':
            self.pre_class_encoder = nn.Linear(self.time_steps * self.shared_dim, shared_dim)
        elif self.merge_mode == 'mlp':
            self.pre_class_encoder = Mlp(hidden_sizes=[self.shared_dim], input_size=self.time_steps * self.shared_dim, output_size=shared_dim)
        self.class_encoder = ClassEncoder(self.num_classes, self.shared_dim)
        self.gauss_encoder_list = nn.ModuleList([nn.Linear(self.shared_dim, self.latent_dim * 2) for _ in range(self.num_classes)])

    def forward(self, x):
        y_distribution, z_distributions = self.encode(x)
        # TODO: could be more efficient if only compute the Gaussian layer of the y that we pick later
        return self.sample_z(y_distribution, z_distributions, y_usage="most_likely", sampler="mean")

    def encode(self, x):
        # Compute shared encoder forward pass
        m = self.shared_encoder(x)

        # Compute class encoder forward pass
        # Variant 1: Pre class encoder
        if self.merge_mode == 'linear' or self.merge_mode == 'mlp':
            flat = torch.flatten(m, start_dim=1)
            pre_class = self.pre_class_encoder(flat)
            y = self.class_encoder(pre_class)
        # Variant 2: Add logits
        elif self.merge_mode == "add":
            y = self.class_encoder(m)
            y = y.sum(dim=-2) / y.shape[1]  # add the outcome of individual samples, scale down
        elif self.merge_mode == "add_softmax":
            y = self.class_encoder(m)
            y = F.softmax(y.sum(dim=-2), dim=-1)  # add the outcome of individual samples, softmax
        # Variant 2: Multiply logits
        elif self.merge_mode == "multiply":
            y = self.class_encoder(m)
            y = F.softmax(y.prod(dim=-2), dim=-1)  # multiply the outcome of individual samples
        y_distribution = torch.distributions.categorical.Categorical(probs=y)

        # Compute every gauss_encoder forward pass
        all_mu_sigma = []
        for net in self.gauss_encoder_list:
            all_mu_sigma.append(net(m))
        z_distributions = [generate_gaussian(mu_sigma, self.latent_dim, mode='multiplication') for mu_sigma in all_mu_sigma]

        return y_distribution, z_distributions

    def sample_z(self, y_distribution, z_distributions, y_usage="specific", y=None, sampler="random"):
        # Select from which Gaussian to sample

        # Used for individual sampling when computing ELBO
        if y_usage == "specific":
            y = ptu.ones(self.batch_size, dtype=torch.long) * y
        # Used while inference
        elif y_usage == "most_likely":
            y = torch.argmax(y_distribution.probs, dim=1)
        else:
            raise RuntimeError("Sampling strategy not specified correctly")

        mask = y.view(-1, 1).unsqueeze(2).repeat(1, 1, self.latent_dim)

        if sampler == "random":
            # Sample from specified Gaussian using reparametrization trick
            # this operation samples from each Gaussian for every class first
            # (diag_embed not possible for distributions), put it back to tensor with shape [class, batch, latent]
            sampled = torch.cat([torch.unsqueeze(z_distributions[i].rsample(), 0) for i in range(self.num_classes)], dim=0)

        elif sampler == "mean":
            sampled = torch.cat([torch.unsqueeze(z_distributions[i].mean, 0) for i in range(self.num_classes)], dim=0)

        # tensor with shape [batch, class, latent]
        permute = sampled.permute(1, 0, 2)
        z = torch.squeeze(torch.gather(permute, 1, mask), 1)
        return z, y


class EncoderMixtureModelTransitionIndividualY(nn.Module):
    '''
    Overall encoder network, putting a shared_encoder, class_encoder and gauss_encoder together.
    '''
    def __init__(self,
                 shared_dim,
                 encoder_input_dim,
                 latent_dim,
                 batch_size,
                 num_classes,
                 time_steps,
                 merge_mode='add'
    ):
        super(EncoderMixtureModelTransitionIndividualY, self).__init__()
        self.shared_dim = shared_dim
        self.encoder_input_dim = encoder_input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes_init = num_classes
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.shared_encoder = Mlp(
                                    hidden_sizes=[shared_dim, shared_dim],
                                    input_size=encoder_input_dim,
                                    output_size=shared_dim,
                                )
        self.class_encoder = ClassEncoder(self.num_classes, self.shared_dim)
        self.gauss_encoder_list = nn.ModuleList([nn.Linear(self.shared_dim, self.latent_dim * 2) for _ in range(self.num_classes)])

    def forward(self, x):
        y_distribution, z_distributions = self.encode(x)
        # TODO: could be more efficient if only compute the Gaussian layer of the y that we pick later
        return self.sample_z(y_distribution, z_distributions, y_usage="most_likely", sampler="mean")

    def encode(self, x):
        # Compute shared encoder forward pass
        m = self.shared_encoder(x)

        # Compute class encoder forward pass
        y = self.class_encoder(m)
        y_distribution = torch.distributions.categorical.Categorical(probs=y)

        all_mu_sigma = []
        for net in self.gauss_encoder_list:
            all_mu_sigma.append(net(m))
        z_distributions = [generate_gaussian(mu_sigma, self.latent_dim) for mu_sigma in all_mu_sigma]

        return y_distribution, z_distributions

    def sample_z(self, y_distribution, z_distribution, y_usage="specific", y=None, sampler="random"):
        z_distributions = ptu.zeros((len(self.gauss_encoder_list), z_distribution[0].mean.shape[0], z_distribution[0].mean.shape[1], 2 * self.latent_dim))
        for i, dist in enumerate(z_distribution):
            z_distributions[i] = torch.cat((dist.mean, dist.scale), dim=-1)

        # Select from which Gaussian to sample
        # Used for individual sampling when computing ELBO
        if y_usage == "specific":
            y = ptu.ones(self.batch_size, self.time_steps, dtype=torch.long) * y
        # Used while inference
        elif y_usage == "most_likely":
            y = torch.argmax(y_distribution.probs, dim=-1)
        else:
            raise RuntimeError("Sampling strategy not specified correctly")

        # values as [batch, timestep, class, latent_dim]
        mask = y.unsqueeze(2).unsqueeze(3).repeat(1, 1, 1, 2 * self.latent_dim)

        # values as [batch, timestep, class, latent_dim]
        z_distributions_permuted = z_distributions.permute(1, 2, 0, 3)
        # gather at dim 2, which is the class dimension, selected by y
        mu_sigma = torch.squeeze(torch.gather(z_distributions_permuted, 2, mask), 2)

        gaussians = generate_gaussian(mu_sigma, self.latent_dim, sigma_ops=None, mode='multiplication')

        if sampler == "random":
            # Sample from specified Gaussian using reparametrization trick
            # this operation samples from each Gaussian for every class first
            # (diag_embed not possible for distributions), put it back to tensor with shape [class, batch, latent]
            z = gaussians.rsample()

        elif sampler == "mean":
            z = gaussians.mean

        y_return = torch.argmax(torch.prod(y_distribution.probs, dim=1), dim=-1)
        return z, y_return
