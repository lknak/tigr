import numpy as np
import torch
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu


def product_of_gaussians3D(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=1)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=1)
    return mu, sigma_squared

def generate_gaussian(mu_sigma, latent_dim, sigma_ops="softplus", mode=None):
    """
    Generate a Gaussian distribution given a selected parametrization.
    """
    mus, sigmas = torch.split(mu_sigma, split_size_or_sections=latent_dim, dim=-1)

    if sigma_ops == 'softplus':
        # Softplus, s.t. sigma is always positive
        # sigma is assumed to be st. dev. not variance
        sigmas = F.softplus(sigmas)
    if mode == 'multiplication':
        mu, sigma = product_of_gaussians3D(mus, sigmas)
    else:
        mu = mus
        sigma = sigmas
    return torch.distributions.normal.Normal(mu, sigma)


def generate_mvn_gaussian(mu_sigma, latent_dim, sigma_ops='softplus', mode=None):
    """
    Generate a Gaussian distribution given a selected parametrization.
    """
    mus, sigmas = torch.split(mu_sigma, split_size_or_sections=latent_dim, dim=-1)

    if sigma_ops == 'softplus':
        # Softplus, s.t. sigma is always positive
        # sigma is assumed to be st. dev. not variance
        sigmas = F.softplus(sigmas)
    # TODO: Test if abs is better than softplus
    elif sigma_ops == 'abs':
        sigmas = torch.abs(sigmas)

    if mode == 'multiplication':
        mus, sigmas = product_of_gaussians3D(mus, sigmas)

    # Avoid singular covariance matrix by adding 1e-8 uncertainty
    return torch.distributions.multivariate_normal.MultivariateNormal(loc=mus, covariance_matrix=(sigmas.unsqueeze(-1) + 1e-8) * ptu.from_numpy(np.eye(latent_dim)).view(*([1 for _ in range(sigmas.ndim - 1)] + [latent_dim, latent_dim])))