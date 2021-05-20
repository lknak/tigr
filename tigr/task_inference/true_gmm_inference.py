import torch
from torch import nn as nn
import torch.nn.functional as F
from tigr.utils import generate_mvn_gaussian

from tigr.task_inference.base_inference import DecoupledEncoder as BaseEncoder


class DecoupledEncoder(BaseEncoder):
	def __init__(self, *args, **kwargs):
		super(DecoupledEncoder, self).__init__(*args, **kwargs)

		# Define mixture model
		self.mixture_model = nn.Sequential(
			nn.Linear(self.shared_dim, self.shared_dim),
			nn.ReLU(),
			nn.Linear(self.shared_dim, self.num_classes * (self.latent_dim * 2 + 1))
		)

		if self.timestep_combination == 'network':
			self.gmm_combiner = nn.Sequential(
				nn.Linear(self.time_steps, self.time_steps),
				nn.ReLU(),
				nn.Linear(self.time_steps, 1)
			)
			self.logit_combiner = nn.Sequential(
				nn.Linear(self.time_steps, self.time_steps),
				nn.ReLU(),
				nn.Linear(self.time_steps, 1)
			)

		self.encode = self.encode_shared_y if self.encoding_mode == 'transitionSharedY' else self.encode_trajectory

	def forward(self, x, sampler='mean', return_probabilities=False):
		# Encode
		latent_distributions, logits = self.encode(x)
		gammas = F.softmax(logits, dim=-1)
		# Sample
		latent_samples = self.sample(latent_distributions, sampler=sampler)
		# Compute latent variable according to activation
		# TODO: Think about if multiplication makes sense here
		latent_variables = torch.sum(latent_samples * gammas[:, :, None], dim=1)

		if not return_probabilities:
			# Calculate max class
			return latent_variables, torch.argmax(gammas, dim=-1)
		else:
			# Calculate max class
			return latent_variables, gammas

	def encode_trajectory(self, x):

		# Compute shared encoder forward pass
		m = self.shared_encoder(x)

		# Compute class encoder forward pass
		mm_out = self.mixture_model(m).view(-1, self.num_classes, self.latent_dim * 2 + 1)

		latent_distributions = mm_out[:, :, :-1]
		logits = mm_out[:, :, -1]

		return generate_mvn_gaussian(latent_distributions, self.latent_dim, sigma_ops=self.sigma_ops), logits

	def encode_shared_y(self, x):

		# Compute shared encoder forward pass
		m = self.shared_encoder(x)

		# Compute class encoder forward pass
		mm_out = self.mixture_model(m).view(-1, self.time_steps, self.num_classes, self.latent_dim * 2 + 1)

		latent_distributions = mm_out[:, :, :, :-1]
		logits = mm_out[:, :, :, -1]

		# TODO: Implement other possibilities for estimating the class from each timestep logit
		if self.timestep_combination == 'network':
			logits = self.logit_combiner(logits.permute(0, 2, 1)).squeeze(dim=-1)
			latent_distributions = self.gmm_combiner(latent_distributions.permute(0, 2, 3, 1)).squeeze(dim=-1)
			return generate_mvn_gaussian(latent_distributions, self.latent_dim, sigma_ops=self.sigma_ops), logits
		elif self.timestep_combination == 'multiplication':
			# Multiply = Sum in log space!
			logits = logits.sum(dim=1)
			return generate_mvn_gaussian(latent_distributions, self.latent_dim, sigma_ops=self.sigma_ops, mode='multiplication'), logits
		else:
			raise NotImplementedError(f'Timestep combination {self.time_step_combination} has not been implemented yet.')