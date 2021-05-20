import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu


class StackedReplayBuffer:
    def __init__(self, max_replay_buffer_size,
                 time_steps,
                 observation_dim,
                 action_dim,
                 task_indicator_dim,
                 permute_samples,
                 encoding_mode,
                 sampling_mode=None):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._task_indicator_dim = task_indicator_dim
        self._max_replay_buffer_size = max_replay_buffer_size

        self._observations = np.zeros((max_replay_buffer_size, observation_dim), dtype=np.float32)
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((max_replay_buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros((max_replay_buffer_size, 1), dtype=np.float32)
        # task indicator computed through encoder
        self._base_task_indicators = np.zeros(max_replay_buffer_size, dtype=np.float32)
        self._task_indicators = np.zeros((max_replay_buffer_size, task_indicator_dim), dtype=np.float32)
        self._next_task_indicators = np.zeros((max_replay_buffer_size, task_indicator_dim), dtype=np.float32)
        self._true_task = np.zeros((max_replay_buffer_size, 1),
                                   dtype=object)  # filled with dicts with keys 'base', 'specification'

        self._sparse_rewards = np.zeros((max_replay_buffer_size, 1), dtype=np.float32)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')

        self.time_steps = time_steps
        self._top = 0
        self._size = 0

        # allowed points specify locations in the buffer, that, alone or together with the <self.time_step> last entries
        # can be sampled
        self._allowed_points = np.zeros(max_replay_buffer_size, dtype=np.bool)
        self._first_timestep = -np.ones(max_replay_buffer_size, dtype=np.int)

        self._train_indices = []
        self._val_indices = []
        self.stats_dict = None
        self.task_info_dict = {}

        self.permute_samples = permute_samples
        self.encoding_mode = encoding_mode
        self.sampling_mode = sampling_mode

    def add_episode(self, episode, task_nr=None):
        # Assume all array are same length (as they come from same rollout)
        length = episode['observations'].shape[0]

        # check, if whole episode fits into buffer
        if length >= self._max_replay_buffer_size:
            error_string = \
                "-------------------------------------------------------------------------------------------\n\n" \
                "ATTENTION:\n" \
                "The current episode was longer than the replay buffer and could not be fitted in.\n" \
                "Please consider decreasing the maximum episode length or increasing the task buffer size.\n\n" \
                "-------------------------------------------------------------------------------------------"
            print(error_string)
            return

        indices_list = np.array([(i + self._top) % self._max_replay_buffer_size for i in range(length)])

        self._observations[indices_list] = episode['observations']
        self._next_obs[indices_list] = episode['next_observations']
        self._actions[indices_list] = episode['actions']
        self._rewards[indices_list] = episode['rewards']
        self._task_indicators[indices_list] = episode['task_indicators']
        self._next_task_indicators[indices_list] = episode['next_task_indicators']
        self._terminals[indices_list] = episode['terminals']
        self._true_task[indices_list] = episode['true_tasks']
        # Note: Base task as true task read in since we are going to overwrite this using the relabeler and if not,
        #       we want to have reasonable numbers in there and no zeros
        self._base_task_indicators[indices_list] = np.array(
            [a['base_task'] for a in episode['true_tasks'].squeeze(axis=1)])

        # Update allowed points with new indices
        self._allowed_points[indices_list] = True
        self._first_timestep[indices_list] = self._top
        # Reset start for next episode in buffer in case we overwrite the start
        next_index = (indices_list[-1] + 1) % self._max_replay_buffer_size
        if -1 < self._first_timestep[next_index]:
            self._first_timestep[self._first_timestep == self._first_timestep[next_index]] = next_index

        # Increase buffer size and set _top to new end
        self._advance_multi(length)

        # Store info about task
        if task_nr is not None:
            bt = episode['true_tasks'][0, 0]['base_task']
            if bt in self.task_info_dict.keys():
                if task_nr in self.task_info_dict[bt].keys():
                    self.task_info_dict[bt][task_nr].append(np.sum(episode['rewards']))
                else:
                    self.task_info_dict[bt][task_nr] = [np.sum(episode['rewards'])]
            else:
                self.task_info_dict[bt] = {task_nr: [np.sum(episode['rewards'])]}

    def _advance_multi(self, length):
        self._top = (self._top + length) % self._max_replay_buffer_size
        self._size = min(self._size + length, self._max_replay_buffer_size)

    def size(self):
        return self._size

    def get_allowed_points(self):
        return np.where(self._allowed_points)[0]

    def sample_data(self, indices):
        return dict(
            observations=self._observations[indices],
            next_observations=self._next_obs[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            task_indicators=self._task_indicators[indices],
            next_task_indicators=self._next_task_indicators[indices],
            sparse_rewards=self._sparse_rewards[indices],
            terminals=self._terminals[indices],
            true_tasks=self._true_task[indices],
            base_tasks_indicators=self._base_task_indicators[indices]
        )

    def get_indices(self, points, batch_size, prio=None):
        rng = np.random.default_rng()
        prio = self.sampling_mode if prio is None else prio

        if prio == 'linear':
            # prioritized version: later samples get more weight
            weights = np.linspace(0.9, 0.1, self._size)[(self._top - 1) - points]
            weights = weights / np.sum(weights)
            indices = rng.choice(points, batch_size, replace=True if batch_size > points.shape[0] else False, p=weights)
        elif prio is None:
            indices = rng.choice(points, batch_size, replace=True if batch_size > points.shape[0] else False)
        else:
            raise NotImplementedError(f'Sampling method {prio} has not been implemented yet.')

        return indices


    # Single transition sample functions

    def sample_random_batch(self, indices, batch_size, prio=None):
        ''' batch of unordered transitions '''
        indices = self.get_indices(indices, batch_size, prio=prio)
        return self.sample_data(indices)

    def sample_sac_data_batch(self, indices, batch_size, prio=None):
        return self.sample_random_batch(indices, batch_size, prio=prio)

        # Sequence sample functions

    def sample_few_step_batch(self, points, batch_size, normalize=True):
        # the points in time together with their <time_step> many entries from before are sampled
        # check if the current point is still in the same 'episode' else take episode start
        all_indices = points[:, None] + np.arange(-self.time_steps, 1)[None, :]
        match_map = (self._first_timestep[all_indices] != self._first_timestep[points][:, None])

        data = self.sample_data(all_indices)

        if normalize:
            data = self.normalize_data(data)

        # TODO: Change in case keys are changed
        # Set data outside of trajectory to 0
        data['observations'][match_map] = 0.
        data['next_observations'][match_map] = 0.
        data['actions'][match_map] = 0.
        data['rewards'][match_map] = 0.

        for key in data:
            data[key] = np.reshape(data[key], (batch_size, self.time_steps + 1, -1))

        return data

    def sample_random_few_step_batch(self, points, batch_size, normalize=True, prio=None, return_sac_data=False):
        ''' batch of unordered small sequences of transitions '''
        indices = self.get_indices(points, batch_size, prio=prio)
        if not return_sac_data:
            return self.sample_few_step_batch(indices, batch_size, normalize=normalize)
        else:
            return self.sample_few_step_batch(indices, batch_size, normalize=normalize), self.sample_data(indices)

    # Relabeler util function
    def relabel_z(self, start, batch_size, z, next_z, y):
        points = self.get_allowed_points()[start:start + batch_size]
        self._task_indicators[points] = z
        self._next_task_indicators[points] = next_z
        self._base_task_indicators[points] = y

    def get_train_val_indices(self, train_val_percent):
        # Split all data from replay buffer into training and validation set
        # not very efficient but hopefully readable code in this function
        points = np.array(self.get_allowed_points())

        train_indices = np.array(self._train_indices)
        val_indices = np.array(self._val_indices)

        points = points[np.isin(points, train_indices, invert=True)]
        points = points[np.isin(points, val_indices, invert=True)]
        points = np.random.permutation(points)
        splitter = int(points.shape[0] * train_val_percent)
        new_train_indices = points[:splitter]
        new_val_indices = points[splitter:]
        self._train_indices += new_train_indices.tolist()
        self._val_indices += new_val_indices.tolist()
        self._train_indices.sort()
        self._val_indices.sort()

        return np.array(self._train_indices), np.array(self._val_indices)

    def make_encoder_data(self, data, batch_size, mode='multiply'):
        # MLP encoder input: state of last timestep + state, action, reward of all timesteps before
        # input is in form [[t-N], ... [t-1], [t]]
        # therefore set action and reward of last timestep = 0
        # Returns: [batch_size, timesteps, obs+action+reward dim]
        # assumes, that a flat encoder flattens the data itself

        observations = torch.from_numpy(data['observations'])
        actions = torch.from_numpy(data['actions'])
        rewards = torch.from_numpy(data['rewards'])
        next_observations = torch.from_numpy((data['next_observations']))

        observations_encoder_input = observations.detach().clone()[:, :-1, :]
        actions_encoder_input = actions.detach().clone()[:, :-1, :]
        rewards_encoder_input = rewards.detach().clone()[:, :-1, :]
        next_observations_encoder_input = next_observations.detach().clone()[:, :-1, :]

        # size: [batch_size, time_steps, obs+action+reward]
        encoder_input = torch.cat(
            [observations_encoder_input, actions_encoder_input, rewards_encoder_input, next_observations_encoder_input],
            dim=-1)

        if self.permute_samples:
            perm = torch.randperm(encoder_input.shape[1]).long()
            encoder_input = encoder_input[:, perm]

        if self.encoding_mode == 'trajectory':
            # size: [batch_size, time_steps * (obs+action+reward)]
            encoder_input = encoder_input.view(batch_size, -1)
        elif self.encoding_mode == 'transitionSharedY' or self.encoding_mode == 'transitionIndividualY':
            pass

        return encoder_input.to(ptu.device)

    def get_stats(self):
        values_dict = dict(
            observations=self._observations[:self._size],
            next_observations=self._next_obs[:self._size],
            actions=self._actions[:self._size],
            rewards=self._rewards[:self._size],
        )
        stats_dict = dict(
            observations={},
            next_observations={},
            actions={},
            rewards={},
        )
        for key in stats_dict.keys():
            stats_dict[key]["max"] = values_dict[key].max(axis=0)
            stats_dict[key]["min"] = values_dict[key].min(axis=0)
            stats_dict[key]["mean"] = values_dict[key].mean(axis=0)
            stats_dict[key]["std"] = values_dict[key].std(axis=0)
        return stats_dict

    def normalize_data(self, data):
        for key in self.stats_dict.keys():
            data[key] = (data[key] - self.stats_dict[key]["mean"]) / (self.stats_dict[key]["std"] + 1e-8)
        return data

    def check_enc(self):

        indices = self.get_allowed_points()
        true_task_list = np.squeeze(self._true_task[indices])
        # Use arrays that are created once
        base_tasks_array = np.array([a['base_task'] for a in true_task_list])
        spec_tasks_array = np.array([a['specification'] for a in true_task_list])
        # Find unique base tasks
        base_tasks = np.unique(base_tasks_array)

        base_spec_dict = {}
        for base_task in base_tasks:
            # Find all unique specifications per base task
            spec_list = np.unique(spec_tasks_array[base_tasks_array == base_task])
            base_spec_dict[base_task] = spec_list

        encoding_storage = {}
        for base in base_spec_dict.keys():
            spec_encoding_dict = {}
            for i, spec in enumerate(base_spec_dict[base]):
                task_indices = np.where(np.logical_and(base_tasks_array == base, spec_tasks_array == spec))[0]

                # Get mean and std of estimated specs
                encodings = self._task_indicators[task_indices]
                mean = np.mean(encodings, axis=0)
                std = np.std(encodings, axis=0)
                # Get bincount of base tasks
                base_task_estimate = np.bincount(self._base_task_indicators[task_indices].astype(int))
                # Store estimated values in dict
                spec_encoding_dict[spec] = dict(mean=mean, std=std, base=base_task_estimate)

            encoding_storage[base] = spec_encoding_dict

        return encoding_storage