# This code is based on rlkit sac_v2 implementation.

from collections import OrderedDict
import os

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from tqdm import tqdm

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict

import vis_utils.tb_logging as TB

from tigr import PCGradOptimizer


class PolicyTrainer:
    def __init__(
            self,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            alpha_net,
            encoder,

            replay_buffer,
            replay_buffer_augmented,
            batch_size,

            env_action_space,
            data_usage_sac,
            use_data_normalization,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=3e-4,
            qf_lr=3e-4,
            optimizer_class=optim.Adam,

            soft_target_tau=5e-3,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=False,
            use_parametrized_alpha=False,
            target_entropy=None,
            target_entropy_factor=1.0,
            alpha=1.0,

            use_PCGrad=False,
            PCGrad_option='random_prob_task'

    ):
        super().__init__()
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.alpha_net = alpha_net
        self.encoder = encoder

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.replay_buffer = replay_buffer
        self.replay_buffer_augmented = replay_buffer_augmented
        self.batch_size = batch_size

        self.env_action_space = env_action_space
        self.data_usage_sac= data_usage_sac
        self.use_data_normalization = use_data_normalization

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.use_parametrized_alpha = use_parametrized_alpha
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -self.env_action_space  # heuristic value from Tuomas
            self.target_entropy = self.target_entropy * target_entropy_factor

            if self.use_parametrized_alpha:
                self.alpha_optimizer = optimizer_class(
                    self.alpha_net.parameters(),
                    lr=policy_lr,
                )
            else:
                self.log_alpha = ptu.zeros(1, requires_grad=True)
                self.alpha_optimizer = optimizer_class(
                    [self.log_alpha],
                    lr=policy_lr,
                )
        self._alpha = ptu.ones(1) * alpha

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss(reduction='none')
        self.vf_criterion = nn.MSELoss(reduction='none')

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        # Optimization options
        self.use_PCGrad = use_PCGrad
        self.PCGrad_option = PCGrad_option

        if self.use_PCGrad:
            # Wrap all optimizers in PCGrad
            self.PCGrad_policy_optimizer = PCGradOptimizer.PCGradOptimizer(
                [self.policy_optimizer],
                verbose=False
            )

            self.PCGrad_qf1_optimizer = PCGradOptimizer.PCGradOptimizer(
                [self.qf1_optimizer],
                verbose=False
            )
            self.PCGrad_qf2_optimizer = PCGradOptimizer.PCGradOptimizer(
                [self.qf2_optimizer],
                verbose=False
            )

    def train(self, policy_steps, augmented_buffer=False):

        if not augmented_buffer:
            indices = np.array(self.replay_buffer.get_allowed_points())
        else:
            indices = np.array(self.replay_buffer_augmented.get_allowed_points())
        
        policy_losses = []
        alphas = []
        log_pis = []
        for step in tqdm(range(policy_steps), desc='Policy trainer'):

            policy_loss, alpha, log_pi = self.training_step(indices, augmented_buffer)
            policy_losses.append(policy_loss/1.0)
            alphas.append(alpha / 1.0)
            log_pis.append((-1) * log_pi.mean() / 1.0)
            if step % 100 == 0 and int(os.environ['DEBUG']) == 1:
                print('Epoch: ' + str(step) + ', policy loss: ' + str(policy_losses[-1]))

        self.eval_statistics['policy_train_steps_total'] = self._n_train_steps_total
        self._need_to_update_eval_statistics = True

        return policy_losses[-1] if len(policy_losses) > 0 else [], self.get_diagnostics()

    def training_step(self, indices, augmented_buffer=False):

        if not augmented_buffer:
            # get data from replay buffer
            encoder_data, sac_data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size, normalize=self.use_data_normalization, return_sac_data=True)

            encoder_input = self.replay_buffer.make_encoder_data(encoder_data, self.batch_size)
            z, gammas = self.encoder(encoder_input, return_probabilities=True)
            task_z = z.detach().clone()
        else:
            sac_data = self.replay_buffer_augmented.sample_sac_data_batch(indices, self.batch_size)
            task_z = ptu.from_numpy(sac_data['task_indicators'])

        rewards = ptu.from_numpy(sac_data['rewards'])
        terminals = ptu.from_numpy(sac_data['terminals'])
        obs = ptu.from_numpy(sac_data['observations'])
        actions = ptu.from_numpy(sac_data['actions'])
        next_obs = ptu.from_numpy(sac_data['next_observations'])

        new_task_z = task_z.clone().detach()

        # Variant 1: train the SAC as if there was no encoder and the state is just extended to be [state , z]
        obs = torch.cat((obs, task_z), dim=1)
        next_obs = torch.cat((next_obs, new_task_z), dim=1)

        '''
        Policy and Alpha Loss
        '''
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparametrize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            if self.use_parametrized_alpha:
                self.log_alpha = self.alpha_net(task_z)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            if self.use_parametrized_alpha:
                alpha = self.log_alpha.exp().detach()
            else:
                alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self._alpha

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )

        policy_loss = (alpha*log_pi - q_new_actions)

        '''
        QF Loss
        '''
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparametrize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        '''
        Update networks
        '''

        # Note: Do not use PCGrad when using augmented buffer since the data is noisy anyway
        if self.use_PCGrad and not augmented_buffer:
            true_task = np.array([a['base_task'] for a in sac_data['true_tasks'][:, 0]], dtype=np.int)
            unique_tasks = torch.unique(ptu.from_numpy(true_task).long()).tolist()
            targets = ptu.from_numpy(true_task).long()

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
            per_class_policy_loss = [torch.mean(policy_loss[task_indices == current_class]) for current_class in unique_tasks]
            per_class_qf1_loss = [torch.mean(qf1_loss[task_indices == current_class]) for current_class in unique_tasks]
            per_class_qf2_loss = [torch.mean(qf2_loss[task_indices == current_class]) for current_class in unique_tasks]

            self.PCGrad_policy_optimizer.minimize(per_class_policy_loss)
            self.PCGrad_qf1_optimizer.minimize(per_class_qf1_loss)
            self.PCGrad_qf2_optimizer.minimize(per_class_qf2_loss)

            # Take mean for all losses for logging
            policy_loss = policy_loss.mean()
            qf1_loss = qf1_loss.mean()
            qf2_loss = qf2_loss.mean()

        else:
            # Take mean for all losses
            policy_loss = policy_loss.mean()
            qf1_loss = qf1_loss.mean()
            qf2_loss = qf2_loss.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer.step()

        '''
        Soft Updates
        '''
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        '''
        Save some statistics for eval
        '''
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            '''
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            '''
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.mean().item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.mean().item()
        self._n_train_steps_total += 1

        if TB.LOG_INTERVAL > 0 and TB.TRAINING_LOG_STEP % TB.LOG_INTERVAL == 0:
            # Write new stats to TB
            TB.TENSORBOARD_LOGGER.add_scalar('rl/alpha', alpha.mean().item(), global_step=TB.TRAINING_LOG_STEP)
            TB.TENSORBOARD_LOGGER.add_scalar('rl/policy_loss', policy_loss.item(), global_step=TB.TRAINING_LOG_STEP)
            TB.TENSORBOARD_LOGGER.add_scalar('rl/qf1_loss', qf1_loss.item(), global_step=TB.TRAINING_LOG_STEP)
            TB.TENSORBOARD_LOGGER.add_scalar('rl/qf2_loss', qf2_loss.item(), global_step=TB.TRAINING_LOG_STEP)
        TB.TRAINING_LOG_STEP += 1

        return ptu.get_numpy(policy_loss), ptu.get_numpy(alpha), ptu.get_numpy(log_pi)

    def get_diagnostics(self):
        return self.eval_statistics


    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )

