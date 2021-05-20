import numpy as np
import torch
import ray
import os

from collections import OrderedDict

import rlkit.torch.pytorch_util as ptu

from PIL import Image

class RolloutCoordinator:
    def __init__(self,
                 env,
                 env_name,
                 env_args,
                 train_or_showcase,
                 agent,
                 replay_buffer,

                 batch_size,

                 time_steps,
                 max_path_length,

                 permute_samples,
                 encoding_mode,

                 use_multiprocessing,
                 use_data_normalization,
                 num_workers,
                 gpu_id,
                 scripted_policy
                 ):
        self.env = env
        self.env_name = env_name
        self.env_args = env_args
        self.train_or_showcase = train_or_showcase
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.time_steps = time_steps
        self.max_path_length = max_path_length
        self.permute_samples = permute_samples
        self.encoding_mode = encoding_mode
        self.use_data_normalization = use_data_normalization

        self.batch_size = batch_size
        self.use_multiprocessing = use_multiprocessing
        self.num_workers = num_workers if self.use_multiprocessing else 1
        self.gpu_id = gpu_id

        self.num_env_steps = 0

        self.action_space = self.env.action_space.low.size
        self.obs_space = self.env.observation_space.low.size

        if self.use_multiprocessing:
            ray.init(
                #log_to_driver=False
                # memory=16000 * 1024 * 1024,
                # object_store_memory=9000 * 1024 * 1024,
                # driver_object_store_memory=1000 * 1024 * 1024
            )

            self.workers = [RemoteRolloutWorker.remote(self.env) for _ in range(self.num_workers)]

        else:
            self.workers = [RolloutWorker(self.env) for _ in range(self.num_workers)]

    def collect_data(self, tasks, train_test, deterministic=False, num_samples_per_task=np.inf, num_trajs_per_task=np.inf, animated=False, save_frames=False):

        assert num_samples_per_task < np.inf or num_trajs_per_task < np.inf, 'either num_samples_per_task or num_trajs_per_task must be finite'

        # Distribute tasks over workers
        tasks_for_worker = np.array_split(np.array(tasks), self.num_workers)

        num_max_batches = max([int(np.ceil(arr.size / self.batch_size)) for arr in tasks_for_worker])
        tasks_for_batch = [np.array_split(np.array(tasks_), num_max_batches) for tasks_ in tasks_for_worker]

        n_steps_total = 0
        n_trajs = 0
        results = []

        # Roll-out paths
        while n_steps_total < num_samples_per_task * len(tasks) and n_trajs < num_trajs_per_task * len(tasks):
            for i in range(num_max_batches):
                active_tasks = [list(tasks_for_batch[j][i]) for j in range(self.num_workers)]

                # Stop if we have enough samples
                if n_steps_total > num_samples_per_task * len(tasks) or n_trajs > num_trajs_per_task * len(tasks): break

                # Set worker index map
                num_tasks = 0
                self.index_table = []
                for l in active_tasks:
                    t_l = [i + num_tasks for i, _ in enumerate(l)]
                    self.index_table.append(np.array(t_l))
                    num_tasks += len(t_l)

                # Reset workers to current tasks
                if self.use_multiprocessing:
                    assert sum(ray.get([worker.reset_buffer.remote(temp_tasks) for worker, temp_tasks in zip(self.workers, active_tasks)])) == num_tasks
                else:
                    assert self.workers[0].reset_buffer(active_tasks[0]) == num_tasks

                results_, n_steps_total_, n_trajs_ = self.rollout(num_tasks, deterministic, animated, save_frames)

                results += results_
                n_steps_total += n_steps_total_
                n_trajs += n_trajs_

        return results


    def rollout(self, n_tasks, deterministic, animated, save_frames):

        n_steps_total = 0
        n_trajs = 0

        results = []

        paths = [dict(
            observations=[],
            task_indicators=[],
            actions=[],
            rewards=[],
            next_observations=[],
            next_task_indicators=[],
            terminals=[],
            agent_infos=[],
            env_infos=[],
            true_tasks=[],
        ) for _ in range(n_tasks)]

        path_length = 0

        # Obtain starting obs and flatten list
        starting_obs = self.step_workers(None, None, False, False)

        obs = np.array(starting_obs)
        terminals = np.zeros(n_tasks, dtype=np.bool)
        self.contexts = torch.zeros((n_tasks, self.time_steps, self.obs_space + self.action_space + 1 + self.obs_space))

        while path_length < self.max_path_length:

            # Terminals map
            terminals_map = np.where(terminals == False)[0]

            # Reshape context to be used in policy
            self.contexts = self.contexts[terminals_map]
            agent_input = self.build_agent_input()

            # Step
            out = self.agent.get_actions(agent_input, ptu.from_numpy(obs), deterministic=deterministic)
            actions, agent_infos = out[0]
            task_indicators = out[1]

            # TODO: Remove
            assert not np.isnan(actions).any(), f'Found nan in actions at path_length {path_length}'
            assert not np.isinf(actions).any(), f'Found inf in actions at path_length {path_length}'

            # Obtain observations and env infos from env
            outputs = self.step_workers(actions, terminals, animated, save_frames)

            next_obs, rewards, new_terminals, env_infos = [list(temp) for temp in zip(*outputs)]

            # Update terminals only at the positions where step env gave a new output (old terminals were not True)
            terminals[terminals_map] = np.array(new_terminals, dtype=np.bool)

            next_obs = np.array(next_obs, dtype=np.float32)
            rewards = np.expand_dims(np.array(rewards, dtype=np.float32), axis=1)
            true_tasks = np.array([env_info['true_task'] for env_info in env_infos])

            # Attention: Actions have different size than obs and rewards
            self.update_context(obs, actions[terminals_map], rewards, next_obs)

            # Only append unterminated tasks
            for active_path_nr, path_nr in enumerate(terminals_map):
                #TODO: Check if all p and path_nr are done correctly
                # Obs, rewards, next_obs, agent_infos, env_infos, true_tasks and (next)task_indicators should get smaller when terminal is reached
                # terminals and actions should always keep size n_tasks
                paths[path_nr]['observations'].append(obs[active_path_nr].copy())
                paths[path_nr]['task_indicators'].append(task_indicators[active_path_nr].copy())
                paths[path_nr]['actions'].append(actions[path_nr].copy())
                paths[path_nr]['rewards'].append(rewards[active_path_nr].copy())
                paths[path_nr]['next_observations'].append(next_obs[active_path_nr].copy())
                # TODO: Calculate next_task_indicators once we work in dynamic envs
                paths[path_nr]['next_task_indicators'].append(task_indicators[active_path_nr].copy())
                paths[path_nr]['terminals'].append(terminals[path_nr].copy())
                paths[path_nr]['agent_infos'].append(agent_infos[active_path_nr])
                paths[path_nr]['env_infos'].append(env_infos[active_path_nr])
                paths[path_nr]['true_tasks'].append(true_tasks[active_path_nr])

            path_length += 1

            # Stop in case all paths are terminated
            if np.array(new_terminals, dtype=np.bool).all():
                break

            # Update obs but only take elements that are not terminated!
            obs = next_obs[np.array(new_terminals, dtype=np.bool) == False]

        for path in paths:
            # Transform lists to np arrays, expand the two arrays that need extra dim in replay buffer
            results.append({key : (np.expand_dims(np.array(path[key]), axis=1) if key in ['terminals', 'true_tasks'] else np.array(path[key])) for key in path.keys()})
            n_steps_total += len(path['observations'])
            n_trajs += 1

        return results, n_steps_total, n_trajs


    def step_workers(self, actions, terminals, animated, save_frames):

        if self.use_multiprocessing:
            out_ = ray.get([worker.step_envs.remote(actions[tasks] if actions is not None else None, terminals[tasks] if terminals is not None else None, animated, save_frames) for tasks, worker in zip(self.index_table, self.workers) if len(tasks) > 0])
        else:
            out_ = [self.workers[0].step_envs(actions, terminals, animated, save_frames)]

        # Flatten list according to how tasks were initialized
        return [el for t in out_ for el in t]

    def collect_replay_data(self, tasks, num_samples_per_task=np.inf):
        num_env_steps = 0
        results = self.collect_data(tasks, 'train', deterministic=False, num_samples_per_task=num_samples_per_task, animated=False)
        for t_ind, path in enumerate(results):
            self.replay_buffer.add_episode(path, task_nr=tasks[t_ind])
            num_env_steps += len(path['observations'])
            
        return num_env_steps

    def evaluate(self, train_test, tasks, num_eval_trajectories, deterministic=True, animated=False, save_frames=False):
        results = self.collect_data(tasks, train_test, deterministic=deterministic, num_trajs_per_task=num_eval_trajectories, animated=animated, save_frames=save_frames)

        if save_frames:
            if 'frame' in results[0]['env_infos'][0].keys():
                images = []
                for p_ind, path in enumerate(results):
                    images_ = []
                    for s_ind, info in enumerate(path['env_infos']):
                        images_.append({
                            'base_task' : path['true_tasks'].squeeze(axis=1)[s_ind]['base_task'],
                            'specification' : path['true_tasks'].squeeze(axis=1)[s_ind]['specification'],
                            'image' : info['frame'],
                            'action' : path['actions'][s_ind],
                            'reward' : path['rewards'][s_ind],
                            'z' : path['task_indicators'][s_ind]
                        })
                    images.append(images_)
                return images

            else:
                raise AssertionError('Saving frames was requested, but no frames were returned from rendering!')

        eval_statistics = OrderedDict()

        # Begin extracting data
        deterministic_string = '_deterministic' if deterministic else '_non_deterministic'
        per_path_rewards = np.array([np.sum(path['rewards']) for path in results])

        eval_average_reward = per_path_rewards.mean()
        eval_std_reward = per_path_rewards.std()
        eval_max_reward = per_path_rewards.max()
        eval_min_reward = per_path_rewards.min()

        eval_statistics[train_test + '_eval_avg_reward' + deterministic_string] = eval_average_reward
        eval_statistics[train_test + '_eval_std_reward' + deterministic_string] = eval_std_reward
        eval_statistics[train_test + '_eval_max_reward' + deterministic_string] = eval_max_reward
        eval_statistics[train_test + '_eval_min_reward' + deterministic_string] = eval_min_reward
        eval_statistics[train_test + '_eval_success_rate'] = -1. # Write out -1. success for non-meta world tasks

        # success rates for meta world
        if 'success' in results[0]['env_infos'][0]:
            success_values = np.array([np.sum([timestep['success'] for timestep in path['env_infos']]) for path in results])
            success_rate = np.sum((success_values > 0).astype(int)) / success_values.shape[0]
            eval_statistics[train_test + '_eval_success_rate'] = success_rate

            # Get success rate per base task
            base_tasks = np.array([path['true_tasks'][0, 0]['base_task'] for path in results])
            success_rates_per_base_task = {base_task : success_values[base_tasks == base_task].mean() for base_task in np.unique(base_tasks)}
            eval_statistics[train_test + '_eval_success_rates_per_base_task'] = success_rates_per_base_task

        if int(os.environ['DEBUG']) == 1:
            print(train_test + ':')
            print('Mean reward: ' + str(eval_average_reward))
            print('Std reward: ' + str(eval_std_reward))
            print('Max reward: ' + str(eval_max_reward))
            print('Min reward: ' + str(eval_min_reward))

        return eval_average_reward, eval_std_reward, eval_max_reward, eval_min_reward, eval_statistics

    def update_context(self, o, a, r, next_o):
        if self.use_data_normalization and self.replay_buffer.stats_dict is not None:
            stats_dict = self.replay_buffer.stats_dict
            o = torch.Tensor((o - stats_dict['observations']['mean']) / (stats_dict['observations']['std'] + 1e-9))
            a = torch.Tensor((a - stats_dict['actions']['mean']) / (stats_dict['actions']['std'] + 1e-9))
            r = torch.Tensor((r - stats_dict['rewards']['mean']) / (stats_dict['rewards']['std'] + 1e-9))
            next_o = torch.Tensor((next_o - stats_dict['next_observations']['mean']) / (stats_dict['next_observations']['std'] + 1e-9))
        else:
            o = torch.Tensor(o)
            a = torch.Tensor(a)
            r = torch.Tensor(r)
            next_o = torch.Tensor(next_o)
        data = torch.cat([o, a, r, next_o], dim=-1).unsqueeze(dim=1)
        context = torch.cat([self.contexts, data], dim=1)
        self.contexts = context[:, -self.time_steps:]

    def build_agent_input(self):
        encoder_input = self.contexts.detach().clone()

        if self.permute_samples:
            perm = torch.LongTensor(torch.randperm(encoder_input.shape[1]))
            encoder_input = encoder_input[:, perm]

        # Trajectory encoding
        if self.encoding_mode == 'trajectory':
            encoder_input = encoder_input.view(encoder_input.shape[0], -1)
        if self.encoding_mode == 'transitionSharedY' or self.encoding_mode == 'transitionIndividualY':
            pass

        return encoder_input.to(ptu.device)

class RolloutWorker:
    def __init__(self,
                 env
                 ):

        self.active_task_list = []
        self.env = env


    def reset_buffer(self, active_task_list):

        self.active_task_list = active_task_list

        # Make tasks unique by adding floating point numbers (workaround to allow different simulation states for same task in the environment buffer)
        task_duplicate_dict = {}
        num_divisor = 10 ** len(str(len(self.active_task_list)))
        for ind, t in enumerate(self.active_task_list):
            if t in task_duplicate_dict:
                task_duplicate_dict[t] += 1
                # Complicated but gives nicer decimals
                self.active_task_list[ind] = t + int(num_divisor * (1. / num_divisor * task_duplicate_dict[t])) / num_divisor
            else:
                task_duplicate_dict[t] = 0

        self.env.clear_buffer()
        for task_nr in self.active_task_list:
            self.env.reset_task(task_nr, keep_buffered=True)

        if len(self.active_task_list) > 0: print(f'\tPreparing env for tasks {self.active_task_list}')
        return len(self.active_task_list)


    def step_envs(self, actions, terminals, animated=False, save_frames=False):
        output = []
        # Only step environments where terminals is False or all if terminals is None
        for i, task_nr in enumerate([self.active_task_list[t_] for t_ in np.where(terminals==False)[0]] if terminals is not None else self.active_task_list):

            self.env.set_task(task_nr)

            if actions is None:
                output.append(self.env.reset())
            else:
                next_o, r, d, env_info = self.env.step(actions[i])

                if animated:
                    self.env.render(mode='human')
                if save_frames:
                    image = Image.fromarray(np.flipud(
                        self.env.get_image(width=640, height=480)))  # make even higher for better quality
                    env_info['frame'] = image

                output.append((next_o, r, d, env_info))

        return output


@ray.remote
class RemoteRolloutWorker(RolloutWorker):
    def __init__(self, env):
        super().__init__(env)
