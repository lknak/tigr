import numpy as np
import torch
from collections import OrderedDict
from rlkit.core import logger
import gtimer as gt
import json
import os
import ray

import rlkit.torch.pytorch_util as ptu

import cv2

import vis_utils.tb_logging as TB


class TrainingAlgorithm:
    def __init__(self,
                 replay_buffer,
                 replay_buffer_augmented,
                 rollout_coordinator,
                 reconstruction_trainer,
                 policy_trainer,

                 agent,
                 networks,
                 train_tasks,
                 test_tasks,
                 task_distribution,

                 latent_dim,
                 num_classes,
                 use_data_normalization,

                 num_epochs,
                 initial_epoch,
                 num_reconstruction_steps,
                 num_policy_steps,
                 num_train_tasks_per_episode,
                 num_transistions_per_episode,

                 augmented_start_percentage,
                 augmented_every,
                 augmented_rollout_length,
                 augmented_rollout_batch_size,

                 num_eval_trajectories,
                 test_evaluation_every,
                 num_showcase,

                 experiment_log_dir,
                 name2number
                 ):
        self.replay_buffer = replay_buffer
        self.replay_buffer_augmented = replay_buffer_augmented

        self.rollout_coordinator = rollout_coordinator

        self.reconstruction_trainer = reconstruction_trainer
        self.policy_trainer = policy_trainer

        self.agent = agent
        self.networks = networks

        self.train_tasks = sorted(train_tasks)
        self.test_tasks = sorted(test_tasks)
        self.task_distribution = task_distribution

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.use_data_normalization = use_data_normalization

        self.num_epochs = num_epochs
        self.initial_epoch = initial_epoch
        self.num_reconstruction_steps = num_reconstruction_steps
        self.num_policy_steps = num_policy_steps
        self.num_transitions_initial = num_transistions_per_episode
        self.num_train_tasks_per_episode = num_train_tasks_per_episode
        self.num_transitions_per_episode = num_transistions_per_episode
        self.num_eval_trajectories = num_eval_trajectories

        self.augmented_start = augmented_start_percentage * self.num_epochs
        self.augmented_every = augmented_every
        self.augmented_rollout_length = augmented_rollout_length
        self.augmented_rollout_batch_size = augmented_rollout_batch_size
        # Calculate num times how often we have to repeat the rollout so we get same amount of samples as for real roll-out
        self.augmented_rollout_steps = round((self.num_transitions_per_episode * self.num_train_tasks_per_episode) / (self.augmented_rollout_length * self.augmented_rollout_batch_size))

        self.experiment_log_dir = experiment_log_dir
        self.latent_dim = latent_dim

        self.num_eval_trajectories = num_eval_trajectories
        self.experiment_log_dir = experiment_log_dir
        self.number2name = {el : key for key, el in name2number.items()} if name2number is not None else None
        # In case we have more classes given than base tasks, add undefined so the code won't crash
        for i in range(len(self.number2name), self.num_classes):
            self.number2name[i] = f'undefined_{i}'

        self.test_evaluation_every = test_evaluation_every
        self.num_showcase = num_showcase

        self._n_env_steps_total = 0

    def train(self):
        params = self.get_epoch_snapshot()
        logger.save_itr_params(-1, params)
        previous_epoch_end = 0
        rng = np.random.default_rng()
        gt.reset_root()

        i_list = np.linspace(0, np.log2(self.num_showcase), int(np.log2(self.num_showcase) + 1))
        showcase_epochs = [self.initial_epoch + el for l in [np.linspace(round((self.num_epochs - 1) / np.log2(self.num_showcase) * i_list[i + 0]),
                                                    round((self.num_epochs - 1) / np.log2(self.num_showcase) * i_list[i + 1]),
                                                    int(2 ** i_list[i + 0] + 1) + (i == 0)).astype(np.int)[1:]
                                        for i in range(len(i_list) - 1)]
                           for el in l]

        print('Collecting initial samples ...')
        self._n_env_steps_total += self.rollout_coordinator.collect_replay_data(np.random.permutation(self.train_tasks), num_samples_per_task=self.num_transitions_per_episode)

        for epoch in gt.timed_for(range(self.initial_epoch, self.initial_epoch + self.num_epochs), save_itrs=True):
            tabular_statistics = OrderedDict()

            # 1. collect data with rollout coordinator
            print('Collecting samples ...')

            # Note: Only calculate gradient once there are 3 samples, otherwise normalization only gives -/+ 2
            if self.task_distribution is None or min([len(self.replay_buffer.task_info_dict[base_tasks][task_nr]) for base_tasks in self.replay_buffer.task_info_dict.keys() for task_nr in self.replay_buffer.task_info_dict[base_tasks].keys()]) < 3:
                data_collection_tasks = rng.choice(self.train_tasks, self.num_train_tasks_per_episode, replace=True if self.num_train_tasks_per_episode > len(self.train_tasks) else False)
            else:
                sorted_indices, task_rewards, occorence_bonus = [], [], []
                for base_tasks in self.replay_buffer.task_info_dict.keys():
                    for task_nr in self.replay_buffer.task_info_dict[base_tasks].keys():
                        sorted_indices.append(task_nr)
                        task_rewards.append(self.replay_buffer.task_info_dict[base_tasks][task_nr])
                        occorence_bonus.append(len(self.replay_buffer.task_info_dict[base_tasks][task_nr]))

                sorted_indices = np.argsort(sorted_indices)
                occorence_bonus = np.array(occorence_bonus)[sorted_indices]

                task_gradients = []
                for tr in task_rewards:
                    tr = np.array(tr)
                    # Normalize
                    tr = (tr - tr.mean()) / (tr.std()) if tr.std() > 0 else tr * 0.
                    # Calculate weights
                    weighting = np.exp(np.arange(-len(tr)+2, 1) * 1 / 10)
                    weighting = weighting / weighting.sum()
                    # Append weighted gradient
                    task_gradients.append(((tr[1:] - tr[:-1]) * weighting).sum())

                task_rewards = np.array(task_gradients)[sorted_indices]
                # Get task rewards to range [0.05, 0.95] for lowest/highest gradients
                task_rewards = (task_rewards - task_rewards.min()) / (task_rewards.max() - task_rewards.min()) * 0.9 + 0.05 if task_rewards.max() != task_rewards.min() else task_rewards * 0. + 0.5

                occorence_bonus = -(occorence_bonus - occorence_bonus.min()) / (occorence_bonus.max() - occorence_bonus.min()) * 0.1 + 0.1 if occorence_bonus.max() != occorence_bonus.min() else occorence_bonus * 0.

                if self.task_distribution == 'worst':
                    data_collection_tasks = rng.choice(self.train_tasks,
                                                       self.num_train_tasks_per_episode,
                                                       replace=True,
                                                       p=(1 - task_rewards + occorence_bonus) / (1 - task_rewards + occorence_bonus).sum())
                elif self.task_distribution == 'best':
                    data_collection_tasks = rng.choice(self.train_tasks,
                                                       self.num_train_tasks_per_episode,
                                                       replace=True,
                                                       p=(task_rewards + occorence_bonus) / (task_rewards + occorence_bonus).sum())
                else:
                    raise NotImplementedError(f'Task distribution {self.task_distribution} has not been implemented yet.')

            self._n_env_steps_total += self.rollout_coordinator.collect_replay_data(data_collection_tasks, num_samples_per_task=self.num_transitions_per_episode)
            tabular_statistics['n_env_steps_total'] = self._n_env_steps_total
            gt.stamp('data_collection')

            # replay buffer stats
            print('Training ...')
            self.replay_buffer.stats_dict = self.replay_buffer.get_stats()

            # 2. encoder - decoder training with reconstruction trainer
            self.reconstruction_trainer.train(self.num_reconstruction_steps)
            gt.stamp('reconstruction_trainer')

            # 4. train policy via SAC with data from the replay buffer
            temp, sac_stats = self.policy_trainer.train(self.num_policy_steps)
            tabular_statistics.update(sac_stats)
            gt.stamp('policy_trainer')

            # 3. Augmentation
            # TODO: Rework
            if epoch >= self.augmented_start and self.augmented_every > 0 and epoch % self.augmented_every == 0:

                # 4. Generate augmented data
                print('Generating augmented samples...')
                num_samples_generated = self.generate_augmented_data()
                print(f'\tGenerated {num_samples_generated} augmented samples...')

                # 5. Train with augmented data
                print('Training with augmented samples...')
                self.policy_trainer.train(self.num_policy_steps, augmented_buffer=True)
            gt.stamp('augmentation_and_training')


            # 6. Evaluation
            if self.test_evaluation_every > 0 and epoch % self.test_evaluation_every == 0:
                print('Evaluation ...')
                # Evaluate on training tasks
                eval_output = self.rollout_coordinator.evaluate('train', self.train_tasks, self.num_eval_trajectories, deterministic=True, animated=False)
                average_test_reward, std_test_reward, max_test_reward, min_test_reward, eval_stats = eval_output
                tabular_statistics.update(eval_stats)

                # Write new stats to TB
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/train/average_reward', average_test_reward, global_step=self._n_env_steps_total)
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/train/std_reward', std_test_reward, global_step=self._n_env_steps_total)
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/train/max_reward', max_test_reward, global_step=self._n_env_steps_total)
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/train/min_reward', min_test_reward, global_step=self._n_env_steps_total)
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/train/success_rate', eval_stats['train_eval_success_rate'], global_step=self._n_env_steps_total)

                # Evaluate on test tasks with nd policy
                eval_output = self.rollout_coordinator.evaluate('test', self.test_tasks, self.num_eval_trajectories, deterministic=False, animated=False)
                average_test_reward, std_test_reward, max_test_reward, min_test_reward, eval_stats = eval_output
                tabular_statistics.update(eval_stats)

                # Write new stats to TB
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/nd_test/average_reward', average_test_reward, global_step=self._n_env_steps_total)
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/nd_test/std_reward', std_test_reward, global_step=self._n_env_steps_total)
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/nd_test/max_reward', max_test_reward, global_step=self._n_env_steps_total)
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/nd_test/min_reward', min_test_reward, global_step=self._n_env_steps_total)
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/nd_test/success_rate', eval_stats['test_eval_success_rate'], global_step=self._n_env_steps_total)

                # Evaluate on test tasks
                eval_output = self.rollout_coordinator.evaluate('test', self.test_tasks, self.num_eval_trajectories, deterministic=True, animated=False)
                average_test_reward, std_test_reward, max_test_reward, min_test_reward, eval_stats = eval_output
                tabular_statistics.update(eval_stats)

                # Write new stats to TB
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/test/average_reward', average_test_reward, global_step=self._n_env_steps_total)
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/test/std_reward', std_test_reward, global_step=self._n_env_steps_total)
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/test/max_reward', max_test_reward, global_step=self._n_env_steps_total)
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/test/min_reward', min_test_reward, global_step=self._n_env_steps_total)
                TB.TENSORBOARD_LOGGER.add_scalar('evaluation/test/success_rate', eval_stats['test_eval_success_rate'], global_step=self._n_env_steps_total)

            gt.stamp('evaluation')

            # 7. Showcase if wanted
            if epoch in showcase_epochs:
                # Display all test tasks in last showcase
                if epoch != showcase_epochs[-1]:
                    # Use gathered data from rollout to choose task for display
                    if len(self.replay_buffer.task_info_dict) > 0:
                        tasks_ = []
                        for k in self.replay_buffer.task_info_dict.keys():
                            tasks_ += list(np.random.choice(list(self.replay_buffer.task_info_dict[k].keys()), 1))
                    else:
                        tasks_ = rng.choice(self.train_tasks, np.min([self.num_classes, len(self.train_tasks)]),
                                         replace=False)
                else:
                    tasks_ = rng.choice(self.test_tasks, np.min([self.num_classes * 5, len(self.test_tasks)]), replace=False)

                print('Rendering and saving training showcase ...')
                # Assuming train tasks are equally filled with different base tasks
                images = self.rollout_coordinator.evaluate('train',
                                                           tasks_,
                                                           self.num_eval_trajectories, deterministic=True, animated=False, save_frames=True)

                for p_nr, path in enumerate(images):

                    dir_ = os.path.join(self.experiment_log_dir, 'videos', f'task_{path[0]["base_task"] if self.number2name is None else self.number2name[path[0]["base_task"]]}')
                    os.makedirs(dir_, exist_ok=True)

                    out = cv2.VideoWriter(os.path.join(dir_, f'epoch_{epoch}_path_{p_nr}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))

                    for im_nr, info in enumerate(path):
                        open_cv_image = np.array(info['image'])
                        # Convert RGB to BGR
                        open_cv_image = open_cv_image[:, :, ::-1].copy()

                        cv2.putText(open_cv_image, f'{path[im_nr]["base_task"] if self.number2name is None else self.number2name[path[im_nr]["base_task"]]}'.upper() + f' | SPEC: {int(info["specification"] * 1000) / 1000 if type(info["specification"]) is np.float64 else [int(k * 1000) / 1000 for k in info["specification"]]}', (0, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                        cv2.putText(open_cv_image, 'reward: ' + str(info["reward"]), (0, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))
                        # cv2.putText(open_cv_image, 'z: ' + str(info["z"]), (0, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))

                        # write the flipped frame
                        out.write(open_cv_image)

                    # Release everything if job is finished
                    out.release()

            gt.stamp('showcase')

            # 8. Logging
            # Network parameters
            params = self.get_epoch_snapshot()
            file_names = logger.save_itr_params(epoch, params)

            # Only keep newest params (remove all other params in folder)
            if len(file_names) > 0:
                for el in os.listdir(os.path.join(logger.get_snapshot_dir(), 'weights')):
                    if '.pth' in el and os.path.join(logger.get_snapshot_dir(), 'weights', el) not in file_names:
                        os.remove(os.path.join(logger.get_snapshot_dir(), 'weights', el), )

            # Replay buffer stats dict
            if self.replay_buffer.stats_dict is not None:
                with open(os.path.join(logger.get_snapshot_dir(), 'weights', 'stats_dict.json'), 'w') as f:
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            return json.JSONEncoder.default(self, obj)
                    json.dump(self.replay_buffer.stats_dict, f, cls=NumpyEncoder)

            # Gaussians
            print('Storing gaussian factors ...')
            self.get_gaussian_factors()

            gt.stamp('logging')

            # 8. Time
            times_itrs = gt.get_times().stamps.itrs
            tabular_statistics['time_data_collection'] = times_itrs['data_collection'][-1]
            tabular_statistics['time_reconstruction_trainer'] = times_itrs['reconstruction_trainer'][-1]
            tabular_statistics['time_policy_trainer'] = times_itrs['policy_trainer'][-1]
            tabular_statistics['time_evaluation'] = times_itrs['evaluation'][-1]
            tabular_statistics['time_showcase'] = times_itrs['showcase'][-1]
            tabular_statistics['time_logging'] = times_itrs['logging'][-1]
            total_time = gt.get_times().total
            epoch_time = total_time - previous_epoch_end
            previous_epoch_end = total_time
            tabular_statistics['time_epoch'] = epoch_time
            tabular_statistics['time_total'] = total_time

            # other
            tabular_statistics['n_env_steps_total'] = self._n_env_steps_total
            tabular_statistics['epoch'] = epoch

            for key, value in tabular_statistics.items():
                logger.record_tabular(key, value)

            logger.dump_tabular(with_prefix=False, with_timestamp=False)

        ray.shutdown()

    def get_gaussian_factors(self, batch_size=1024, num_samples_per_class=100):

        # Estimate mean gaussian factors by sampling from replay buffer
        train_indices, _ = self.replay_buffer.get_train_val_indices(self.reconstruction_trainer.train_val_percent)

        data = self.replay_buffer.sample_random_few_step_batch(train_indices, self.num_classes * num_samples_per_class, normalize=self.use_data_normalization)

        # Prepare for usage in encoder
        encoder_input = self.replay_buffer.make_encoder_data(data, self.num_classes * num_samples_per_class)

        true_task = np.array([a['base_task'] for a in data['true_tasks'][:, -1, 0]], dtype=np.int)
        specs = [a['specification'] for a in data['true_tasks'][:, -1, 0]]
        targets = ptu.from_numpy(true_task).long()

        evidence_variables = torch.zeros([encoder_input.shape[0], self.latent_dim])
        predicted_classes = torch.zeros([encoder_input.shape[0]], dtype=torch.long)

        for i in range(int(np.ceil(self.num_classes * num_samples_per_class / batch_size))):
            with torch.no_grad():
                # Forward pass through encoder
                evidence_variables_, predicted_classes_ = self.reconstruction_trainer.encoder(encoder_input[i * batch_size : (i + 1) * batch_size])

                evidence_variables[i * batch_size : (i + 1) * batch_size] = evidence_variables_
                predicted_classes[i * batch_size : (i + 1) * batch_size] = predicted_classes_

        d = []
        with torch.no_grad():
            for class_nr in range(self.num_classes):
                d.append({'values' : ptu.get_numpy(evidence_variables[targets == class_nr, :]),
                          'classes' : ptu.get_numpy(predicted_classes[targets == class_nr])})

        # Add embedding to tf
        emb = evidence_variables
        if evidence_variables.shape[-1] < 3:
            emb = torch.zeros(evidence_variables.shape[0], 3)
            emb[:, :evidence_variables.shape[1]] = evidence_variables
        TB.TENSORBOARD_LOGGER.add_embedding(emb,
                                            metadata=[f'{self.number2name[t.item()]} [{int(s * 100) / 100 if type(s) is np.float64 else [int(k * 100) / 100 for k in s]}] -> {self.number2name[predicted_classes[i].item()]}'
                                                      if self.number2name is not None
                                                      else f'{t.item()} [{int(s * 100) / 100 if type(s) is np.float64 else [int(k * 100) / 100 for k in s]}] -> {predicted_classes[i].item()}'
                                                      for i, (t, s) in enumerate(zip(targets, specs))],
                                            global_step=TB.TI_LOG_STEP)

        return d

    def generate_augmented_data(self):

        samples_counter = 0

        for step_nr in range(self.augmented_rollout_steps):
            # Randomly sample starting point from experienced data
            indices = np.array(self.replay_buffer.get_allowed_points())
            data = self.replay_buffer.sample_random_few_step_batch(indices, self.augmented_rollout_batch_size,
                                                                   normalize=self.use_data_normalization)

            encoder_input = self.replay_buffer.make_encoder_data(data, self.augmented_rollout_batch_size)
            with torch.no_grad():
                z, gammas = self.policy_trainer.encoder(encoder_input, return_probabilities=True)

            obs = ptu.from_numpy(data['observations'])[:, -1, :]
            actions = data['actions'][:, -1, :]
            rewards = data['rewards'][:, -1, :]

            task_z = z.detach().clone()
            true_tasks = data['true_tasks'][:, -1, :]

            # Create arrays to store rollout data
            paths_observations = np.zeros(list(obs.shape) + [self.augmented_rollout_length])
            paths_actions = np.zeros(list(actions.shape) + [self.augmented_rollout_length])
            paths_rewards = np.zeros(list(rewards.shape) + [self.augmented_rollout_length])
            paths_next_observations = np.zeros(list(obs.shape) + [self.augmented_rollout_length])

            # TODO: Think about if it makes sense to recalculate task indicators
            paths_task_indicators = ptu.get_numpy(task_z).copy()[:, :, None].repeat(self.augmented_rollout_length, axis=2)
            paths_next_task_indicators = ptu.get_numpy(task_z).copy()[:, :, None].repeat(self.augmented_rollout_length, axis=2)
            paths_terminals = np.zeros([obs.shape[0], 1, self.augmented_rollout_length], dtype=np.bool)
            paths_true_tasks = true_tasks.copy()[:, None].repeat(self.augmented_rollout_length, axis=2)

            # Perform roll-outs using sampled z
            for timestep in range(self.augmented_rollout_length):

                with torch.no_grad():
                    actions = self.policy_trainer.policy(torch.cat([obs, task_z], dim=-1), deterministic=False)[0]
                    next_obs, rewards = self.reconstruction_trainer.decoder(obs, actions, None, task_z)

                paths_observations[:, :, timestep] = ptu.get_numpy(obs)
                paths_actions[:, :, timestep] = ptu.get_numpy(actions)
                paths_rewards[:, :, timestep] = ptu.get_numpy(rewards)
                paths_next_observations[:, :, timestep] = ptu.get_numpy(next_obs)

                # Set obs to next state
                obs = next_obs

                samples_counter += self.augmented_rollout_batch_size

            # Add to augmented Replay-buffer
            for path_nr in range(self.augmented_rollout_steps):
                episode = {
                    "observations": paths_observations[path_nr, :, :].transpose(),
                    "task_indicators": paths_task_indicators[path_nr, :, :].transpose(),
                    "actions": paths_actions[path_nr, :, :].transpose(),
                    "rewards": paths_rewards[path_nr, :, :].transpose(),
                    "next_observations": paths_next_observations[path_nr, :, :].transpose(),
                    "next_task_indicators": paths_next_task_indicators[path_nr, :, :].transpose(),
                    "terminals": paths_terminals[path_nr, :, :].transpose(),
                    "true_tasks": paths_true_tasks[path_nr, :, :].transpose(),
                }
                self.replay_buffer_augmented.add_episode(episode)

        return samples_counter

    def showcase_task_inference(self):

        results = self.rollout_coordinator.collect_data(self.test_tasks, 'train', deterministic=True, num_samples_per_task=self.num_transitions_per_episode,
                                    animated=False)
        with open(os.path.join(logger.get_snapshot_dir(), 'roll_out_results.json'), 'w') as f:
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return json.JSONEncoder.default(self, obj)

            json.dump(results, f, cls=NumpyEncoder)

    def showcase_all(self):

        print('Rendering environment interaction in test envs ...')
        # Assuming train tasks are equally filled with different base tasks
        rng = np.random.default_rng()
        images = self.rollout_coordinator.evaluate('test', self.test_tasks, 1, deterministic=True, animated=False,
                                                   save_frames=True)

        print('Saving training showcase ...')
        for p_nr, path in enumerate(images):

            dir_ = os.path.join(self.experiment_log_dir, 'videos', f'showcase_all')
            os.makedirs(dir_, exist_ok=True)

            out = cv2.VideoWriter(os.path.join(dir_, f'path_{p_nr}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                  20.0, (640, 480))

            for im_nr, info in enumerate(path):
                open_cv_image = np.array(info['image'])
                # Convert RGB to BGR
                open_cv_image = open_cv_image[:, :, ::-1].copy()

                cv2.putText(open_cv_image,
                            f'{path[im_nr]["base_task"] if self.number2name is None else self.number2name[path[im_nr]["base_task"]]}'.upper() + f' | SPEC: {int(info["specification"] * 1000) / 1000 if type(info["specification"]) is np.float64 else [int(k * 1000) / 1000 for k in info["specification"]]}',
                            (0, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                cv2.putText(open_cv_image, 'reward: ' + str(info["reward"]), (0, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.3,
                            (0, 0, 255))
                # cv2.putText(open_cv_image, 'z: ' + str(info["z"]), (0, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))

                # write the flipped frame
                out.write(open_cv_image)

                # Empty memory
                path[im_nr]['image'] = None

            # Release everything if job is finished
            out.release()
        print(f'\tShowcasing finished, average return: {np.mean(np.sum([[transition["reward"] for transition in path] for path in images], axis=1))}. Writing results json.')

        with open(os.path.join(logger.get_snapshot_dir(), 'showcase_all_results.json'), 'w') as f:
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return json.JSONEncoder.default(self, obj)

            json.dump(images, f, cls=NumpyEncoder)

    def showcase_non_stationary_env(self):

        for i in range(self.rollout_coordinator.num_workers):
            self.rollout_coordinator.workers[i].env.wrapped_env.change_mode = 'time'
            self.rollout_coordinator.workers[i].env.wrapped_env.meta_mode = 'test'
            self.rollout_coordinator.workers[i].env.wrapped_env.change_steps = 80
            self.rollout_coordinator.max_path_length = 560

        print('Rendering environment interaction in non-stationary env ...')
        # Assuming train tasks are equally filled with different base tasks
        rng = np.random.default_rng()
        images = self.rollout_coordinator.evaluate('test',
                                                   rng.choice(self.test_tasks, np.min([5, len(self.test_tasks)]), replace=False),
                                                   self.num_eval_trajectories, deterministic=True, animated=False,
                                                   save_frames=True)

        print('Saving training showcase ...')
        for p_nr, path in enumerate(images):

            dir_ = os.path.join(self.experiment_log_dir, 'videos', f'non_stationary_tasks')
            os.makedirs(dir_, exist_ok=True)

            out = cv2.VideoWriter(os.path.join(dir_, f'path_{p_nr}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                  20.0, (640, 480))

            for im_nr, info in enumerate(path):
                open_cv_image = np.array(info['image'])
                # Convert RGB to BGR
                open_cv_image = open_cv_image[:, :, ::-1].copy()

                cv2.putText(open_cv_image,
                            f'{path[im_nr]["base_task"] if self.number2name is None else self.number2name[path[im_nr]["base_task"]]}'.upper() + f' | SPEC: {int(info["specification"] * 1000) / 1000 if type(info["specification"]) is np.float64 else [int(k * 1000) / 1000 for k in info["specification"]]}',
                            (0, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                cv2.putText(open_cv_image, 'reward: ' + str(info["reward"]), (0, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.3,
                            (0, 0, 255))
                # cv2.putText(open_cv_image, 'z: ' + str(info["z"]), (0, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255))

                # write the flipped frame
                out.write(open_cv_image)

                # Empty memory
                path[im_nr]['image'] = None

            # Release everything if job is finished
            out.release()
        print(f'\tShowcasing finished, avg return: {np.mean(np.sum([[transition["reward"] for transition in path] for path in images], axis=1))}. Writing results json')

        with open(os.path.join(logger.get_snapshot_dir(), 'non_stationary_results.json'), 'w') as f:
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return json.JSONEncoder.default(self, obj)

            json.dump(images, f, cls=NumpyEncoder)

    def get_epoch_snapshot(self):
        snapshot = OrderedDict()
        for name, net in self.networks.items():
            snapshot[name] = net.state_dict()
        return snapshot

    def to(self, device=None):
        if device is None:
            device = ptu.device
        for net in self.networks:
            self.networks[net].to(device)
        self.agent.to(device)