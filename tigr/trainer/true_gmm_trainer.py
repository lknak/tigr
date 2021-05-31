import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.kl as kl
from tigr.utils import generate_mvn_gaussian
import rlkit.torch.pytorch_util as ptu

from rlkit.core.eval_util import create_stats_ordered_dict

from tigr import PCGradOptimizer

from tigr.trainer.base_trainer import AugmentedTrainer as BaseTrainer

import vis_utils.tb_logging as TB


class AugmentedTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(AugmentedTrainer, self).__init__(*args, **kwargs)

        self.optimizer_mixture_model = self.optimizer_class(
            [{'params': self.encoder.shared_encoder.parameters()},
             {'params': self.encoder.mixture_model.parameters()}],
            lr=self.lr_encoder
        )

        if self.use_PCGrad:
            # Wrap both optimizers in PCGrad
            self.PCGrad_mixture_model_optimizer = PCGradOptimizer.PCGradOptimizer(
                [self.optimizer_mixture_model, self.optimizer_decoder],
                verbose=False
            )

        self.loss_ce = nn.CrossEntropyLoss(reduction='none')

    def mixture_training_step(self, indices):

        '''
        Computes a forward pass to encoder and decoder with sampling at the encoder
        '''

        # Get data from real replay buffer
        # TODO: Think about normalization
        data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size, normalize=self.use_data_normalization)

        # Prepare for usage in decoder
        actions = ptu.from_numpy(data['actions'])[:, 1:, :]
        states = ptu.from_numpy(data['observations'])[:, 1:, :]
        next_states = ptu.from_numpy(data['next_observations'])[:, 1:, :]
        rewards = ptu.from_numpy(data['rewards'])[:, 1:, :]
        terminals = ptu.from_numpy(data['terminals'])[:, 1:, :]

        # Remove last (trailing) dimension here
        true_task = np.array([a['base_task'] for a in data['true_tasks'][:, -1, 0]], dtype=np.int)
        unique_tasks = torch.unique(ptu.from_numpy(true_task).long()).tolist()
        targets = ptu.from_numpy(true_task).long()

        decoder_state_target = next_states[:, :, :self.state_reconstruction_clip]

        '''
        MIXTURE MODEL TRAINING
        '''

        # Prepare for usage in encoder
        encoder_input = self.replay_buffer.make_encoder_data(data, self.batch_size)

        # Forward pass through encoder
        latent_distributions, logits = self.encoder.encode(encoder_input)
        gammas = F.softmax(logits, dim=-1)

        # Sample latent variables
        latent_samples = self.encoder.sample(latent_distributions)
        latent_variables = torch.sum(latent_samples * gammas[:, :, None], dim=1)

        '''
        Dynamics Prediction Loss
        '''

        # Calculate standard losses
        # Put in decoder to get likelihood
        state_estimate, reward_estimate = self.decoder(states, actions, decoder_state_target, latent_variables.unsqueeze(1).repeat(1, self.timesteps, 1))
        mixture_state_loss = torch.mean((decoder_state_target - state_estimate) ** 2, dim=[-2, -1])
        mixture_reward_loss = torch.mean((rewards - reward_estimate) ** 2, dim=[-2, -1])

        # TODO: Reactivate q loss?
        # TODO: Normal weighting
        mixture_nll = self.loss_weight_state * mixture_state_loss + self.loss_weight_reward * mixture_reward_loss

        assert not torch.isnan(latent_distributions.mean).any(), latent_distributions.mean
        assert not torch.isnan(latent_distributions.stddev).any(), latent_distributions.stddev
        assert not torch.isnan(logits).any(), logits
        assert not torch.isnan(latent_variables).any(), latent_variables
        assert not torch.isnan(state_estimate).any(), state_estimate
        assert not torch.isnan(reward_estimate).any(), reward_estimate
        assert not torch.isnan(mixture_state_loss).any(), mixture_state_loss
        assert not torch.isnan(mixture_reward_loss).any(), mixture_reward_loss

        # Calculate extra losses

        # KL ( q(z | x,y=k) || p(z|y=k) )
        # Use uniform prior!
        kl_qz_pz = torch.sum(
            kl.kl_divergence(
                latent_distributions,
                generate_mvn_gaussian(
                    torch.cat([ptu.zeros([latent_distributions._batch_shape[0], self.num_classes, self.latent_dim]),
                               ptu.ones([latent_distributions._batch_shape[0], self.num_classes, self.latent_dim])],
                              dim=-1),
                    self.latent_dim, sigma_ops=None)),
            dim=-1)

        clustering_loss = self.alpha_kl_z * kl_qz_pz

        # Clustering loss for maximum sparsity, meaning only one dimension should be used!
        sparsity_loss = torch.abs(latent_variables).sum(dim=-1)
        clustering_loss = clustering_loss + self.gamma_sparsity * sparsity_loss

        # Clustering loss maximizing the distance between latent variable mean for different classes, scale by standard deviation
        if self.num_classes > 1:
            # Sum over all classes in 2nd dimension --> (batch_size x classes x latend_dims)
            distances = torch.sum((latent_distributions.mean[:, :, None, :] - latent_distributions.mean[:, None, :, :]) ** 2, dim=[2])
            # Inversely scale distance with stddev, prevent division by 0 and sum over all classes and latent dims, divide by 2 since we have distance of k1-k2 and k2-k1 (twice)
            euclid_loss = torch.sum(latent_distributions.stddev / (distances + 1e-8), dim=[1, 2]) / 2
            clustering_loss = clustering_loss + self.beta_euclid * euclid_loss

        # Component constraint classification learning
        reg_loss = ptu.zeros(1)

        if self.use_regularization_loss:
            if (targets < self.num_classes).all():
                # Calculate class regularization loss
                reg_loss = self.loss_ce(logits, targets)

                # Total mixture loss after reg
                clustering_loss = clustering_loss + self.regularization_lambda * reg_loss
            else:
                self.use_regularization_loss = False
                print('Warning: Encountered target outside of given class range, disabling regularization loss!')

        # Overall elbo, but weight KL div takes up self.alpha_kl_z fraction of the loss!
        elbo = mixture_nll + clustering_loss

        # Take mean over each true task so one task won't dominate
        mixture_loss = torch.sum(elbo)

        # TODO: Remove check for nan
        assert not torch.isnan(reg_loss).any(), reg_loss
        assert not torch.isnan(mixture_nll).any(), mixture_nll
        assert not torch.isnan(kl_qz_pz).any(), kl_qz_pz
        assert not torch.isnan(elbo).any(), elbo
        assert not torch.isnan(mixture_loss).any(), mixture_loss

        if self.use_PCGrad:
            # Find according class for every sample
            if self.PCGrad_option == 'true_task':
                task_indices = targets
            elif self.PCGrad_option == 'most_likely_task':
                task_indices = torch.argmax(gammas, dim=-1)
            elif self.PCGrad_option == 'random_prob_task':
                task_indices = torch.distributions.categorical.Categorical(gammas).sample()
            else:
                raise NotImplementedError(f'Option {self.PCGrad_option} for PCGrad was not implemented yet.')

            # Group all elements according to class, also elbo should be maximized, and backward function assumes minimization
            per_class_total_loss = [torch.sum(elbo[task_indices == current_class]) for current_class in unique_tasks]

            self.PCGrad_mixture_model_optimizer.minimize(per_class_total_loss)

        else:
            # Optimize mixture model first and afterwards do the same with activation encoder
            self.optimizer_mixture_model.zero_grad()
            self.optimizer_decoder.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            mixture_loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            self.optimizer_mixture_model.step()
            self.optimizer_decoder.step()


        total_state_loss = ptu.get_numpy(torch.sum(mixture_state_loss)) / self.batch_size
        total_reward_loss = ptu.get_numpy(torch.sum(mixture_reward_loss)) / self.batch_size

        if False and self._n_train_steps_mixture == 0:
            import vis_utils.helper_functions as helper_functions
            helper_functions.print_loss_params_relation(
                [('Elbo', elbo),
                 ('Mixture State Loss', mixture_state_loss),
                 ('Mixture Reward Loss', mixture_reward_loss),
                 ('Reg Loss', reg_loss),
                 ('KL Loss', kl_qz_pz),
                 ('Total Loss', mixture_loss)],
                [('Mixture Model', self.encoder.mixture_model.named_parameters()),
                 ('Shared Encoder', self.encoder.shared_encoder.named_parameters()),
                 ('Decoder', self.decoder.named_parameters())])


        if TB.LOG_INTERVAL > 0 and TB.TI_LOG_STEP % TB.LOG_INTERVAL == 0:
            # Write new stats to TB
            # Normalize all with batch size
            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_loss', (torch.sum(mixture_loss) / self.batch_size).item(), global_step=TB.TI_LOG_STEP)
            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_elbo_loss', (torch.sum(elbo) / self.batch_size).item(), global_step=TB.TI_LOG_STEP)
            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_state_losses', total_state_loss.item(), global_step=TB.TI_LOG_STEP)
            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_reward_losses', total_reward_loss.item(), global_step=TB.TI_LOG_STEP)

            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_nll', (torch.sum(mixture_nll) / self.batch_size).item(), global_step=TB.TI_LOG_STEP)

            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_clustering_losses', (torch.sum(clustering_loss) / self.batch_size).item(), global_step=TB.TI_LOG_STEP)

            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_klz_loss', (torch.sum(kl_qz_pz) / self.batch_size).item(), global_step=TB.TI_LOG_STEP)
            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_sparsity_loss', (torch.sum(sparsity_loss) / self.batch_size).item(), global_step=TB.TI_LOG_STEP)
            if self.num_classes > 1:
                TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_euclid_loss', (torch.sum(euclid_loss) / self.batch_size).item(), global_step=TB.TI_LOG_STEP)

            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_classification_acc', (torch.argmax(gammas, dim=-1) == targets).float().mean().item(), global_step=TB.TI_LOG_STEP)

            if self.use_regularization_loss:
                TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_regularization_loss', reg_loss.mean().item(), global_step=TB.TI_LOG_STEP)
        TB.TI_LOG_STEP += 1

        return ((torch.sum(mixture_loss) / self.batch_size),
                total_state_loss,
                total_reward_loss)

    def validate_mixture(self, indices):

        # Get data from real replay buffer
        data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size, normalize=self.use_data_normalization)

        # Prepare for usage in decoder
        actions = ptu.from_numpy(data['actions'])[:, -1, :]
        states = ptu.from_numpy(data['observations'])[:, -1, :]
        next_states = ptu.from_numpy(data['next_observations'])[:, -1, :]
        rewards = ptu.from_numpy(data['rewards'])[:, -1, :]
        terminals = ptu.from_numpy(data['terminals'])[:, -1, :]

        # Remove last (trailing) dimension here
        true_task = np.array([a['base_task'] for a in data['true_tasks'][:, -1, 0]], dtype=np.int)
        targets = ptu.from_numpy(true_task).long()

        decoder_state_target = next_states[:, :self.state_reconstruction_clip]

        '''
        MIXTURE MODEL
        '''

        with torch.no_grad():
            # Prepare for usage in encoder
            encoder_input = self.replay_buffer.make_encoder_data(data, self.batch_size)
            # Forward pass through encoder
            latent_distributions, logits = self.encoder.encode(encoder_input)
            gammas = F.softmax(logits, dim=-1)

            # Sample latent variables
            latent_samples = self.encoder.sample(latent_distributions)
            latent_variables = torch.sum(latent_samples * gammas[:, :, None], dim=1)

            '''
            Dynamics Prediction Loss
            '''

            # Calculate standard losses
            # Put in decoder to get likelihood
            state_estimate, reward_estimate = self.decoder(states, actions, decoder_state_target, latent_variables)
            mixture_state_loss = torch.mean((state_estimate - decoder_state_target) ** 2, dim=-1)
            mixture_reward_loss = torch.mean((reward_estimate - rewards) ** 2, dim=-1)

            # TODO: Reactivate q loss?
            mixture_nll = self.loss_weight_state * mixture_state_loss + self.loss_weight_reward * mixture_reward_loss

            assert not torch.isnan(latent_distributions.mean).any(), latent_distributions.mean
            assert not torch.isnan(latent_distributions.stddev).any(), latent_distributions.stddev
            assert not torch.isnan(logits).any(), logits
            assert not torch.isnan(latent_variables).any(), latent_variables
            assert not torch.isnan(state_estimate).any(), state_estimate
            assert not torch.isnan(reward_estimate).any(), reward_estimate
            assert not torch.isnan(mixture_state_loss).any(), mixture_state_loss
            assert not torch.isnan(mixture_reward_loss).any(), mixture_reward_loss

            # Calculate extra losses
            # Calculate class regularization loss
            reg_loss = ptu.zeros(1)

            if self.use_regularization_loss:
                loss_ce = nn.CrossEntropyLoss(reduction='none')
                reg_loss = loss_ce(logits, targets)

                # Total mixture loss after reg
                mixture_nll = mixture_nll + self.regularization_lambda * reg_loss

            # KL ( q(z | x,y=k) || p(z|y=k) )
            # Use uniform prior!
            kl_qz_pz = torch.sum(kl.kl_divergence(latent_distributions,
                                                  generate_mvn_gaussian(torch.cat([ptu.zeros(
                                                      [latent_distributions._batch_shape[0], self.num_classes, self.latent_dim]),
                                                                               ptu.ones([latent_distributions._batch_shape[0],
                                                                                         self.num_classes,
                                                                                         self.latent_dim])], dim=-1),
                                                                    self.latent_dim, sigma_ops=None)),
                                 dim=-1)

            # Overall elbo, but weight KL div takes up self.alpha_kl_z fraction of the loss!
            elbo = - mixture_nll - self.alpha_kl_z * kl_qz_pz

            # Take mean over each true task so one task won't dominate
            mixture_loss = -torch.sum(elbo)

            # TODO: Remove check for nan
            assert not torch.isnan(latent_distributions.mean).any()
            assert not torch.isnan(latent_distributions.stddev).any()
            assert not torch.isnan(logits).any(), logits
            assert not torch.isnan(latent_variables).any(), latent_variables
            assert not torch.isnan(reg_loss).any(), reg_loss
            assert not torch.isnan(mixture_nll).any(), mixture_nll
            assert not torch.isnan(kl_qz_pz).any(), kl_qz_pz
            assert not torch.isnan(elbo).any(), elbo
            assert not torch.isnan(mixture_loss).any(), mixture_loss
        # TODO: Reactivate q loss?
        return (ptu.get_numpy(mixture_loss) / self.batch_size,
                ptu.get_numpy(torch.sum(mixture_state_loss)) / self.batch_size,
                ptu.get_numpy(torch.sum(mixture_reward_loss)) / self.batch_size,
                0.)#ptu.get_numpy(torch.sum(mixture_qf_loss)) / self.batch_size)
