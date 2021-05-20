import numpy as np


def rollout(env, agent, max_path_length=np.inf, accum_context=True, animated=False, save_frames=False, plotting=False, online=0, buffer_size=1):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout
    :return:
    """
    observations = []
    task_indicators = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []

    if plotting:
        zs = np.zeros((1, agent.latent_dim*2))

    o = env.reset()
    next_o = None
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        out = agent.get_action(o)
        a, agent_info = out[0]
        task_indicator = out[1]
        next_o, r, d, env_info = env.step(a)
        # update the agent's current context
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        if online == 1:
            # Variant 1: context is last N elements
            cont = agent.get_last_n_context_elements(buffer_size)
            agent.infer_posterior(cont)
        elif online == 2:
            # Variant 2: task assignment check
            agent.do_task_assignment(buffer_size)
        else:
            pass
        if plotting:
            zs = np.append(zs, np.concatenate((agent.z_means.detach().cpu().numpy(), agent.z_vars.detach().cpu().numpy()), axis=1), axis=0)
        observations.append(o)
        task_indicators.append(task_indicator)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image(width=640, height=480)))
            env_info['frame'] = image
        env_infos.append(env_info)

    next_task_indicator = agent.get_action(next_o)[1]
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    task_indicators = np.array(task_indicators)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        task_indicators. np.expand_dim(task_indicators, 1)
        next_o = np.array([next_o])
        next_task_indicator = np.array([next_task_indicator])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    next_task_indicators = np.vstack(
        (
            task_indicators[1:, :],
            np.expand_dims(next_task_indicator, 0)
        )
    )

    if plotting:
        import matplotlib.pyplot as plt
        plt.figure()
        zs= zs[1:-1,:]
        for i in range(zs.shape[1]):
            plt.plot(list(range(zs.shape[0])), zs[:,i], label="z"+str(i))

        plt.plot(list(range(len(rewards))), rewards, label="reward")
        plt.legend()
        #plt.savefig("a.png", dpi=300, format="png")
        plt.show()

        if online == 2:
            #other stuff
            plt.figure()
            plt.plot(list(range(len(agent.best_matching_tasks))), agent.best_matching_tasks, label="best_matching_tasks")
            plt.plot(list(range(len(agent.task_numbers))), agent.task_numbers, label="#tasks")
            plt.legend()
            #plt.savefig("b.png", dpi=300, format="png")
            plt.show()


    return dict(
        observations=observations,
        task_indicators=task_indicators,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        next_task_indicators=next_task_indicators,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
