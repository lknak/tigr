import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu

from rlkit.core import logger
from tqdm import tqdm

from tigr import stacked_replay_buffer, PCGradOptimizer
from tigr.task_inference import base_inference as task_inference
from tigr.task_inference import prediction_networks

import vis_utils.tb_logging as TB


def weighting_fun(loss_array, c=1., m=1.):
    weights = loss_array / (np.sum(loss_array) + 1e-8)
    return c - m * weights


class AugmentedTrainer:
    def __init__(self,
                 encoder : task_inference.DecoupledEncoder,
                 decoder : prediction_networks.DecoderMDP,
                 replay_buffer : stacked_replay_buffer.StackedReplayBuffer,
                 replay_buffer_augmented : stacked_replay_buffer.StackedReplayBuffer,
                 batch_size,
                 num_classes,
                 latent_dim,
                 timesteps,
                 lr_decoder,
                 lr_encoder,
                 alpha_kl_z,
                 beta_euclid,
                 gamma_sparsity,
                 regularization_lambda,
                 use_state_diff,
                 state_reconstruction_clip,
                 use_data_normalization,

                 train_val_percent,
                 eval_interval,
                 early_stopping_threshold,
                 experiment_log_dir,

                 use_regularization_loss,

                 use_PCGrad=False,
                 PCGrad_option='random_prob_task',
                 optimizer_class=optim.Adam,
                 ):
        self.encoder = encoder
        self.decoder = decoder
        self.replay_buffer = replay_buffer
        self.replay_buffer_augmented = replay_buffer_augmented

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.timesteps = timesteps

        self.lr_decoder = lr_decoder
        self.lr_encoder = lr_encoder
        self.alpha_kl_z = alpha_kl_z
        self.beta_euclid = beta_euclid
        self.gamma_sparsity = gamma_sparsity

        self.use_state_diff = use_state_diff
        self.state_reconstruction_clip = state_reconstruction_clip
        self.use_data_normalization = use_data_normalization

        self.train_val_percent = train_val_percent
        self.eval_interval = eval_interval
        self.early_stopping_threshold = early_stopping_threshold
        self.experiment_log_dir = experiment_log_dir

        self.use_regularization_loss = use_regularization_loss
        self.regularization_lambda = regularization_lambda

        self.loss_weight_state = 1 / 3
        self.loss_weight_reward = 1 / 3
        self.loss_weight_qf = 1 / 3

        self.lowest_loss = np.inf
        self.lowest_loss_epoch = 0

        self.temp_path = os.path.join(os.getcwd(), '.temp', self.experiment_log_dir.split('/')[-1])
        self.encoder_path = os.path.join(os.getcwd(), self.temp_path, 'encoder.pth')
        self.decoder_path = os.path.join(os.getcwd(), self.temp_path, 'decoder.pth')

        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        self.optimizer_class = optimizer_class

        self.sigma_ops = 'softplus'

        # Optimization options
        self.use_PCGrad = use_PCGrad
        self.PCGrad_option = PCGrad_option

        self.optimizer_decoder = self.optimizer_class(
            self.decoder.parameters(),
            lr=self.lr_decoder,
        )

        self._n_train_steps_mixture = 0


    def train(self, mixture_steps, w_method='val_value_based'):

        train_indices, val_indices = self.replay_buffer.get_train_val_indices(self.train_val_percent)

        # Reset lowest loss for mixture
        self.lowest_loss_epoch = 0
        self.lowest_loss = np.inf

        '''
        MIXTURE TRAINING EPOCHS
        '''
        mixture_training_step = 0
        for mixture_training_step in tqdm(range(mixture_steps), desc='Reconstruction trainer'):

            # Perform training step
            self.mixture_training_step(train_indices)

            self._n_train_steps_mixture += 1

            # Evaluate with validation set for early stopping
            if mixture_training_step % self.eval_interval == 0:
                losses_ = self.validate_mixture(val_indices)
                if len(losses_) == 3:
                    val_total_loss, val_state_loss, val_reward_loss, val_qf_loss = list(losses_) + [0.]
                else:
                    val_total_loss, val_state_loss, val_reward_loss, val_qf_loss = losses_

                # Change loss weighting
                if w_method == 'val_value_based':
                    # Normalize the other way: Lower values with higher weight
                    temp = weighting_fun(np.array([val_state_loss, val_reward_loss, val_qf_loss]))
                    self.loss_weight_state, self.loss_weight_reward, self.loss_weight_qf = temp

                if False and self.early_stopping(mixture_training_step, val_total_loss):
                    print('Mixture: Early stopping at epoch ' + str(mixture_training_step))
                    break

        logger.record_tabular('Mixture_steps', mixture_training_step + 1)

        return self.lowest_loss_epoch

    def mixture_training_step(self, indices):

        raise NotImplementedError('Function "mixture_training_step" must be implemented for augmented trainer.')

    def policy_training_step(self, indices, use_real_data):

        raise NotImplementedError('Function "policy_training_step" must be implemented for augmented trainer.')

    def validate_mixture(self, indices):

        raise NotImplementedError('Function "validate_mixture" must be implemented for augmented trainer.')

    def validate_policy(self, indices):

        raise NotImplementedError('Function "validate_policy" must be implemented for augmented trainer.')

    def calculate_regularization_distances(self, z_distributions, stddev_factor=1.0):
        op_ = torch.abs if self.sigma_ops == 'abs' else F.softplus

        means = z_distributions.mean
        stddevs = z_distributions.stddev

        mean_matrix = torch.abs(means[:, None, :] - means[None, :, :])
        stddev_matrix = op_(stddevs[:, None, :]) + op_(stddevs[None, :, :])

        distances_matrix = torch.sum(torch.clamp(mean_matrix - stddev_factor * stddev_matrix, min=0) ** 2, dim=-1)
        per_class_distances = distances_matrix.sum(dim=1)

        return per_class_distances


    def early_stopping(self, epoch, loss):
        if loss < self.lowest_loss:

            if int(os.environ['DEBUG']) == 1:
                print('Found new minimum at Epoch ' + str(epoch))

            self.lowest_loss = loss
            self.lowest_loss_epoch = epoch

            torch.save(self.encoder.state_dict(), self.encoder_path)
            torch.save(self.decoder.state_dict(), self.decoder_path)

        if epoch - self.lowest_loss_epoch > self.early_stopping_threshold:
            return True
        else:
            return False
