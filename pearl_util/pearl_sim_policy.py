import os, shutil
import os.path as osp
import cv2
import json
import numpy as np
import click
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv, CameraWrapper
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.agent import PEARLAgent
from configs.pearl_default import default_config
from pearl_launch_experiment import deep_update_dict
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.util import rollout


def sim_policy(variant, path_to_exp, num_trajs=1, deterministic=True, save_video=False):
    '''
    simulate a trained policy adapting to a new task
    optionally save videos of the trajectories - requires ffmpeg
    :variant: experiment configuration dict
    :path_to_exp: path to exp folder
    :num_trajs: number of trajectories to simulate per task (default 1)
    :deterministic: if the policy is deterministic (default stochastic)
    :save_video: whether to generate and save a video (default False)
    '''

    # create multi-task environment and sample tasks
    env = CameraWrapper(NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params'])), '')#variant['util_params']['gpu_id']
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    eval_tasks=list(tasks[-variant['n_eval_tasks']:])
    print('testing on {} test tasks, {} trajectories each'.format(len(eval_tasks), num_trajs))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=obs_dim + action_dim + reward_dim,
        output_size=context_encoder,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    # deterministic eval
    if deterministic:
        agent = MakeDeterministic(agent)

    # load trained weights (otherwise simulate random policy)
    context_encoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'context_encoder.pth')))#, map_location='cpu'))
    policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'policy.pth')))#, map_location='cpu'))

    if save_video:
        os.makedirs(os.path.join(path_to_exp, 'videos'), exist_ok=True)

    # loop through tasks collecting rollouts
    all_rets = []
    number2name = {el : key for key, el in env.name2number.items()} if env.name2number is not None else None
    for p_nr, idx in enumerate(eval_tasks):
        env.reset_task(idx)
        agent.clear_z()
        paths = []
        for n in range(num_trajs):
            path = rollout(env, agent, max_path_length=variant['algo_params']['max_path_length'], accum_context=True, save_frames=save_video)
            # Only save last trajectory
            if n == num_trajs - 1 and save_video:

                paths.append(path)

                dir_ = os.path.join(path_to_exp, 'videos',
                                    f'task_{path["env_infos"][0]["true_task"]["base_task"] if number2name is None else number2name[path["env_infos"][0]["true_task"]["base_task"]]}')
                os.makedirs(dir_, exist_ok=True)

                out = cv2.VideoWriter(os.path.join(dir_, f'path_{p_nr}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))

                for im_nr, info in enumerate(path['env_infos']):
                    open_cv_image = np.array(info['frame'])
                    # Convert RGB to BGR
                    open_cv_image = open_cv_image[:, :, ::-1].copy()

                    cv2.putText(open_cv_image,
                                f'{info["true_task"]["base_task"] if number2name is None else number2name[info["true_task"]["base_task"]]}'.upper() + f' | SPEC: {int(info["true_task"]["specification"] * 1000) / 1000 if type(info["true_task"]["specification"]) is np.float64 else [int(k * 1000) / 1000 for k in info["true_task"]["specification"]]}',
                                (0, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                    cv2.putText(open_cv_image, 'reward: ' + str(path["rewards"][im_nr][0]), (0, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.3,
                                (0, 0, 255))
                    # cv2.putText(open_cv_image, 'z: ' + str(path["task_indicators"][im_nr]), (0, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.3,
                    #             (0, 0, 255))

                    # write the flipped frame
                    out.write(open_cv_image)

                # Release everything if job is finished
                out.release()

                # remove frames from memory
                for temp_i in range(len(path['env_infos'])): path['env_infos'][temp_i]['frame'] = None

            if n >= variant['algo_params']['num_exp_traj_eval']:
                agent.infer_posterior(agent.context)
        all_rets.append([sum(p['rewards']) for p in paths])
        print(f'Finished rollout {p_nr+1}/{len(eval_tasks)}')

    # compute average returns across tasks
    n = min([len(a) for a in all_rets])
    rets = [a[:n] for a in all_rets]
    rets = np.mean(np.stack(rets), axis=0)
    for i, ret in enumerate(rets):
        print('trajectory {}, avg return: {} \n'.format(i, ret))


@click.command()
@click.argument('config', default=None)
@click.argument('path', default=None)
@click.option('--num_trajs', default=3)
@click.option('--deterministic', is_flag=True, default=True)
@click.option('--video', is_flag=True, default=True)
def main(config, path, num_trajs, deterministic, video):
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    sim_policy(variant, path, num_trajs, deterministic, video)


if __name__ == "__main__":
    main()
