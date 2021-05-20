# Task Inference based meta-rl algorithm using Gaussian mixture models and gated Recurrent units (TIGR)

import os
import numpy as np
import click
import json
import torch
import copy

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import Mlp, FlattenMlp
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config

from tigr.task_inference.prediction_networks import DecoderMDP
from tigr.sac import PolicyTrainer
from tigr.stacked_replay_buffer import StackedReplayBuffer
from tigr.rollout_worker import RolloutCoordinator
from tigr.agent_module import Agent, ScriptedPolicyAgent
from tigr.training_algorithm import TrainingAlgorithm
from tigr.task_inference.true_gmm_inference import DecoupledEncoder
from tigr.trainer.true_gmm_trainer import AugmentedTrainer

from torch.utils.tensorboard import SummaryWriter
import vis_utils.tb_logging as TB


def experiment(variant):
    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    torch.set_num_threads(1)

    # Important: Gru and Conv only work with trajectory encoding
    if variant['algo_params']['encoder_type'] in ['gru'] and variant['algo_params']['encoding_mode'] != 'trajectory':
        print(f'\nInformation: Setting encoding mode to trajectory since encoder type '
              f'"{variant["algo_params"]["encoder_type"]}" doesn\'t work with '
              f'"{variant["algo_params"]["encoding_mode"]}"!\n')
        variant['algo_params']['encoding_mode'] = 'trajectory'
    elif variant['algo_params']['encoder_type'] in ['transformer', 'conv'] and variant['algo_params']['encoding_mode'] != 'transitionSharedY':
        print(f'\nInformation: Setting encoding mode to trajectory since encoder type '
              f'"{variant["algo_params"]["encoder_type"]}" doesn\'t work with '
              f'"{variant["algo_params"]["encoding_mode"]}"!\n')
        variant['algo_params']['encoding_mode'] = 'transitionSharedY'

    # Seeding
    if(variant['algo_params']['use_fixed_seeding']):
        torch.manual_seed(variant['algo_params']['seed'])
        np.random.seed(variant['algo_params']['seed'])

    # create logging directory
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=variant['util_params']['exp_name'],
                                      base_log_dir=variant['util_params']['base_log_dir'], snapshot_mode='gap',
                                      snapshot_gap=variant['algo_params']['snapshot_gap'])

    # Create tensorboard writer and reset values
    TB.TENSORBOARD_LOGGER = SummaryWriter(log_dir=os.path.join(experiment_log_dir, 'tensorboard'))
    TB.LOG_INTERVAL = variant['util_params']['tb_log_interval']
    TB.TRAINING_LOG_STEP = 0
    TB.AUGMENTATION_LOG_STEP = 0
    TB.TI_LOG_STEP = 0
    TB.DEBUG_LOG_STEP = 0

    # create multi-task environment and sample tasks
    env = ENVS[variant['env_name']](**variant['env_params'])
    if variant['env_params']['use_normalized_env']:
        env = NormalizedBoxEnv(env)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    tasks = list(range(len(env.tasks)))
    train_tasks = list(range(len(env.train_tasks)))
    test_tasks = tasks[-variant['env_params']['n_eval_tasks']:]

    # Dump task dict as json
    name2number = None
    if hasattr(env, 'name2number'):
        name2number = env.name2number
        with open(os.path.join(experiment_log_dir, 'task_dict.json'), 'w') as f:
            json.dump(name2number, f)

    # instantiate networks
    net_complex_enc_dec = variant['reconstruction_params']['net_complex_enc_dec']
    latent_dim = variant['algo_params']['latent_size']
    time_steps = variant['algo_params']['time_steps']
    num_classes = variant['reconstruction_params']['num_classes']

    # encoder used: single transitions or trajectories
    if variant['algo_params']['encoding_mode'] == 'transitionSharedY':
        encoder_input_dim = obs_dim + action_dim + reward_dim + obs_dim
        shared_dim = int(encoder_input_dim * net_complex_enc_dec)  # dimension of shared encoder output
    elif variant['algo_params']['encoding_mode'] == 'trajectory':
        encoder_input_dim = time_steps * (obs_dim + action_dim + reward_dim + obs_dim)
        shared_dim = int(encoder_input_dim / time_steps * net_complex_enc_dec)  # dimension of shared encoder output
    else:
        raise NotImplementedError

    encoder = DecoupledEncoder(
        shared_dim,
        encoder_input_dim,
        latent_dim,
        num_classes,
        time_steps,
        encoding_mode=variant['algo_params']['encoding_mode'],
        timestep_combination=variant['algo_params']['timestep_combination'],
        encoder_type=variant['algo_params']['encoder_type']
    )

    decoder = DecoderMDP(
        action_dim,
        obs_dim,
        reward_dim,
        latent_dim,
        net_complex_enc_dec,
        variant['env_params']['state_reconstruction_clip'],
    )

    M = variant['algo_params']['sac_layer_size']
    qf1 = FlattenMlp(
        input_size=(obs_dim + latent_dim) + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    qf2 = FlattenMlp(
        input_size=(obs_dim + latent_dim) + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=(obs_dim + latent_dim) + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=(obs_dim + latent_dim) + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=(obs_dim + latent_dim),
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_sizes=[M, M, M],
    )

    alpha_net = Mlp(
        hidden_sizes=[latent_dim * 10],
        input_size=latent_dim,
        output_size=1
    )

    networks = {'encoder': encoder,
                'decoder': decoder,
                'qf1': qf1,
                'qf2': qf2,
                'target_qf1': target_qf1,
                'target_qf2': target_qf2,
                'policy': policy,
                'alpha_net': alpha_net}

    replay_buffer = StackedReplayBuffer(
        variant['algo_params']['max_replay_buffer_size'],
        time_steps,
        obs_dim,
        action_dim,
        latent_dim,
        variant['algo_params']['permute_samples'],
        variant['algo_params']['encoding_mode'],
        variant['algo_params']['sampling_mode']
    )
    replay_buffer_augmented = StackedReplayBuffer(
        variant['algo_params']['max_replay_buffer_size'],
        time_steps,
        obs_dim,
        action_dim,
        latent_dim,
        variant['algo_params']['permute_samples'],
        variant['algo_params']['encoding_mode'],
        variant['algo_params']['sampling_mode']
    )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        itr = variant['showcase_itr']
        path = variant['path_to_weights']
        for name, net in networks.items():
            net.load_state_dict(torch.load(os.path.join(path, name + '_itr_' + str(itr) + '.pth'), map_location='cpu'))
        print(f'Loaded weights "{variant["path_to_weights"]}"')
        if os.path.exists(os.path.join(variant['path_to_weights'], 'stats_dict.json')):
            with open(os.path.join(variant['path_to_weights'], 'stats_dict.json'), 'r') as f:
                # Copy so not both changed during updates
                d = npify_dict(json.load(f))
                replay_buffer.stats_dict = d
                replay_buffer_augmented.stats_dict = copy.deepcopy(d)
        else:
            if variant['algo_params']['use_data_normalization']:
                raise ValueError('WARNING: No stats dict for replay buffer was found. '
                                 'Stats dict is required for the algorithm to work properly!')

    #Agent
    agent_class = ScriptedPolicyAgent if variant['env_params']['scripted_policy'] else Agent
    agent = agent_class(
        encoder,
        policy
    )

    # Rollout Coordinator
    rollout_coordinator = RolloutCoordinator(
        env,
        variant['env_name'],
        variant['env_params'],
        variant['train_or_showcase'],
        agent,
        replay_buffer,

        variant['algo_params']['batch_size_rollout'],
        time_steps,

        variant['algo_params']['max_path_length'],
        variant['algo_params']['permute_samples'],
        variant['algo_params']['encoding_mode'],
        variant['util_params']['use_multiprocessing'],
        variant['algo_params']['use_data_normalization'],
        variant['util_params']['num_workers'],
        variant['util_params']['gpu_id'],
        variant['env_params']['scripted_policy']
        )

    reconstruction_trainer = AugmentedTrainer(
        encoder,
        decoder,
        replay_buffer,
        None,
        variant['algo_params']['batch_size_reconstruction'],
        num_classes,
        latent_dim,
        time_steps,
        variant['reconstruction_params']['lr_decoder'],
        variant['reconstruction_params']['lr_encoder'],
        variant['reconstruction_params']['alpha_kl_z'],
        variant['reconstruction_params']['beta_euclid'],
        variant['reconstruction_params']['gamma_sparsity'],
        variant['reconstruction_params']['regularization_lambda'],
        variant['reconstruction_params']['use_state_diff'],
        variant['env_params']['state_reconstruction_clip'],
        variant['algo_params']['use_data_normalization'],

        variant['reconstruction_params']['train_val_percent'],
        variant['reconstruction_params']['eval_interval'],
        variant['reconstruction_params']['early_stopping_threshold'],
        experiment_log_dir,

        variant['reconstruction_params']['use_regularization_loss'],

        use_PCGrad = variant['PCGrad_params']['use_PCGrad'],
        PCGrad_option = variant['PCGrad_params']['PCGrad_option'],
        optimizer_class = torch.optim.Adam,
    )


    # PolicyTrainer
    policy_trainer = PolicyTrainer(
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        alpha_net,
        encoder,

        replay_buffer,
        replay_buffer_augmented,
        variant['algo_params']['batch_size_policy'],
        action_dim,
        'tree_sampling',
        variant['algo_params']['use_data_normalization'],

        use_automatic_entropy_tuning=variant['algo_params']['automatic_entropy_tuning'],
        target_entropy_factor=variant['algo_params']['target_entropy_factor'],
        alpha=variant['algo_params']['sac_alpha'],

        use_PCGrad=variant['PCGrad_params']['use_PCGrad'],
        PCGrad_option=variant['PCGrad_params']['PCGrad_option']
    )

    algorithm = TrainingAlgorithm(
        replay_buffer,
        replay_buffer_augmented,
        rollout_coordinator,
        reconstruction_trainer,
        policy_trainer,
        agent,
        networks,

        train_tasks,
        test_tasks,
        variant['task_distribution'],

        latent_dim,
        num_classes,
        variant['algo_params']['use_data_normalization'],

        variant['algo_params']['num_train_epochs'],
        variant['showcase_itr'] if variant['path_to_weights'] is not None else 0,
        variant['algo_params']['num_training_steps_reconstruction'],
        variant['algo_params']['num_training_steps_policy'],
        variant['algo_params']['num_train_tasks_per_episode'],
        variant['algo_params']['num_transitions_per_episode'],

        variant['algo_params']['augmented_start_percentage'],
        variant['algo_params']['augmented_every'],
        variant['algo_params']['augmented_rollout_length'],
        variant['algo_params']['augmented_rollout_batch_size'],

        variant['algo_params']['num_eval_trajectories'],
        variant['algo_params']['test_evaluation_every'],
        variant['algo_params']['num_showcase'],

        experiment_log_dir,
        name2number
        )

    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    PLOT = variant['util_params']['plot']
    os.environ['DEBUG'] = str(int(DEBUG))
    os.environ['PLOT'] = str(int(PLOT))

    # create temp folder
    if not os.path.exists(variant['reconstruction_params']['temp_folder']):
        os.makedirs(variant['reconstruction_params']['temp_folder'])

    # run the algorithm
    if variant['train_or_showcase'] == 'train':
        algorithm.train()
        algorithm.showcase_task_inference()
    elif variant['train_or_showcase'] == 'showcase_all':
        algorithm.showcase_all()
    elif variant['train_or_showcase'] == 'showcase_task_inference':
        algorithm.showcase_task_inference()
    elif variant['train_or_showcase'] == 'showcase_non_stationary_env':
        algorithm.showcase_non_stationary_env()


def npify_dict(d: dict):
    for k, v in d.items():
        if type(v) is dict:
            d[k] = npify_dict(v)
        else:
            d[k] = np.asarray(v)
    return d


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


@click.command()
@click.argument('config', default=None)
@click.option('--name', default='')
@click.option('--ti_option', default='')
@click.option('--gpu', default=None)
@click.option('--num_workers', default=None)
@click.option('--use_mp', is_flag=True, default=None)
def click_main(config, name, ti_option, gpu, use_mp, num_workers):
    main(config, name, ti_option, gpu, use_mp, num_workers)


def main(config=None, name='', ti_option='', gpu=None, use_mp=None, num_workers=None):
    variant = default_config

    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    # Only set values from input if they are actually inputted
    variant['inference_option'] = variant['inference_option'] if ti_option == '' else ti_option
    variant['util_params']['exp_name'] = f'{os.path.splitext(os.path.split(config)[1])[0].replace("-", "_") if config is not None else "default"}_' + variant['inference_option'] + (f'_{name}' if name != '' else f'')

    variant['util_params']['use_gpu'] = variant['util_params']['use_gpu'] if gpu != '' else False
    variant['util_params']['gpu_id'] = variant['util_params']['gpu_id'] if gpu is None else gpu
    variant['util_params']['use_multiprocessing'] = variant['util_params']['use_multiprocessing'] if use_mp is None else use_mp
    variant['util_params']['num_workers'] = variant['util_params']['num_workers'] if num_workers is None else int(num_workers)

    experiment(variant)


if __name__ == "__main__":
    click_main()
