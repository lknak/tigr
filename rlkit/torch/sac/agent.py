import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_ify
from scipy.stats import multivariate_normal as normal


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class Task:
    def __init__(self, latent_dim):
        self.z_means = torch.zeros(latent_dim)
        self.z_vars = torch.ones(latent_dim)
        self.buffer = None

class PEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy

        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

        # list of tasks that will be expanded when seeing new tasks
        self.task_list = []
        self.best_matching_tasks = []
        self.task_numbers = []
        self.num_online_tasks = -1
        self.current_online_task = -1

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])

        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def get_last_n_context_elements(self, n):
        return self.context[:,-n:,:].detach().clone()

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])  # Encoder could give negative values, but sigma must be positive
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z()

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]  # rsample() samples using the reparametrization trick
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic), np_ify(z.clone().detach())[0, :]

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context)
        self.sample_z()

        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparametrize=True, return_log_prob=True)

        return policy_outputs, task_z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    # ---------------- Functions by DL ---------------------------

    def update_task_buffer(self, task_number, data):
        if self.task_list[task_number].buffer is None:
            self.task_list[task_number].buffer = data
        else:
            self.task_list[task_number].buffer = torch.cat([self.task_list[task_number].buffer, data], dim=1)

    def initialize_new_online_task(self, z_means, z_vars, new_context):
        self.task_list.append(Task(self.latent_dim))
        self.task_list[-1].z_means = z_means
        self.task_list[-1].z_vars = torch.tensor([[10.0]])
        self.update_task_buffer(-1, new_context)
        self.num_online_tasks += 1
        self.current_online_task = self.num_online_tasks

    def get_latent_embedding(self, context):
        '''basically like infer posterior, but no sampling'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        mu = params[..., :self.latent_dim]
        sigma_squared = F.softplus(
            params[..., self.latent_dim:])  # Encoder could give negative values, but sigma must be positive
        z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        z_means = torch.stack([p[0] for p in z_params])
        z_vars = torch.stack([p[1] for p in z_params])
        return z_means, z_vars

    def kl_divergence(self, mu1, sigma_1, mu2, sigma_2):
        sigma_diag_1 = np.eye(sigma_1.shape[0]) * sigma_1
        sigma_diag_2 = np.eye(sigma_2.shape[0]) * sigma_2

        sigma_diag_2_inv = np.linalg.inv(sigma_diag_2)

        kl = 0.5 * (np.log(np.linalg.det(sigma_diag_2) / np.linalg.det(sigma_diag_2))
                    - mu1.shape[0] + np.trace(np.matmul(sigma_diag_2_inv, sigma_diag_1))
                    + np.matmul(np.matmul(np.transpose(mu2 - mu1), sigma_diag_2_inv), (mu2 - mu1))
                    )
        return kl

    def get_best_matching_task(self, z_means_new, z_vars_new, mode='likelihood'):
        #z_means_new = z_means_new.detach().numpy()
        #z_vars_new = z_vars_new.detach().numpy()

        if mode == 'likelihood':
            # Calculates log-likelihood of the mean of the measured new point w.r.t. Normal distribution of the classes
            # puts threshold on the log-likelihood
            assignment_threshold = -10
            likelihood = torch.zeros(len(self.task_list))

            for i in range(len(self.task_list)):
                mu = self.task_list[i].z_means
                sigma = self.task_list[i].z_vars
                likelihood[i] = torch.distributions.normal.Normal(mu, torch.sqrt(sigma)).log_prob(z_means_new)

            best_matching_task = torch.argmax(likelihood)
            best_likelihood = likelihood[best_matching_task]

            if best_likelihood < assignment_threshold:
                best_matching_task = -1

        if mode == 'sigma':
            # Calculates log-likelihood of the mean of the measured new point w.r.t. Normal distribution of the classes
            # Checks if the mean of the measured new point is within some standard deviation distance
            sigma_multi = 3
            likelihood = torch.zeros(len(self.task_list))

            for i in range(len(self.task_list)):
                mu = self.task_list[i].z_means
                sigma = self.task_list[i].z_vars
                likelihood[i] = torch.distributions.normal.Normal(mu, torch.sqrt(sigma)).log_prob(z_means_new)

            best_matching_task = torch.argmax(likelihood)
            best_likelihood = likelihood[best_matching_task]

            # Check if new point is out of some sigma in both directions for best task
            mu = self.task_list[best_matching_task].z_means
            sigma = self.task_list[best_matching_task].z_vars
            if torch.ge(z_means_new, mu + sigma_multi * sigma).any() or torch.le(z_means_new, mu - sigma_multi * sigma).any():
                best_matching_task = -1

        elif mode == 'kl':
            # calculates the symmetric variant of the KL divergence
            # D_kl ( P || Q ) + D_kl (Q || P)
            assignment_threshold = 0.1
            distance = np.zeros(len(self.task_list))

            for i in range(len(self.task_list)):
                mu = self.task_list[i].z_means.detach().numpy()
                sigma = np.diag(self.task_list[i].z_vars.detach().numpy())
                distance[i] = self.kl_divergence(mu, sigma, z_means_new, z_vars_new) + self.kl_divergence(z_means_new, z_vars_new, mu, sigma)

            best_matching_task = np.argmin(distance)
            best_task_distance = distance[best_matching_task]

            if best_task_distance < assignment_threshold:
                best_matching_task = -1

        return best_matching_task


    def do_task_assignment(self, context_length):
        # for newest context get latent (Gaussian parameters)
        new_context = self.get_last_n_context_elements(context_length)
        z_means_new, z_vars_new = self.get_latent_embedding(new_context)

        # first task
        if not self.task_list:
            self.initialize_new_online_task(z_means_new, z_vars_new, new_context)
            self.z_means = z_means_new
            self.z_vars = z_vars_new
        else:
            # check likelihood w.r.t. existing tasks
            best_matching_task = self.get_best_matching_task(z_means_new, z_vars_new)
            self.best_matching_tasks.append(best_matching_task)
            self.task_numbers.append(len(self.task_list))

            # if not well fitting: initialize new task
            if best_matching_task == -1:
                self.initialize_new_online_task(z_means_new, z_vars_new, new_context)
                self.z_means = z_means_new
                self.z_vars = z_vars_new
            else:
                # put newest context in buffer of best fitting
                self.current_online_task = best_matching_task
                self.update_task_buffer(self.current_online_task, new_context) # TODO: currently single transistions are copied multiple times into buffer
                # recompute overall mean and variance
                # TODO: make more efficient: do not encode again but make product of Gaussians
                #context = self.task_list[self.current_online_task].buffer
                #z_mean, z_var = self.get_latent_embedding(context)
                mu_old = self.task_list[self.current_online_task].z_means
                var_old = self.task_list[self.current_online_task].z_vars
                mu_new = z_means_new
                #var_new = z_vars_new
                var_new = torch.tensor([[10.0]])

                z_params = [_product_of_gaussians(torch.cat([mu_old, mu_new]), torch.cat([var_old, var_new]))]

                self.task_list[self.current_online_task].z_means = torch.stack([z_params[0][0]])
                self.task_list[self.current_online_task].z_vars = torch.stack([z_params[0][1]])
                self.z_means = torch.stack([z_params[0][0]])
                self.z_vars = torch.stack([z_params[0][1]])
        self.sample_z()

    @property
    def networks(self):
        return [self.context_encoder, self.policy]




