# default experiment settings
# all experiments should modify these settings only as needed
default_config = dict(
    env_name='cheetah-multi-task',
    env_params=dict(
        n_train_tasks=80,  # number train tasks
        n_eval_tasks=40,  # number evaluation tasks tasks
        use_normalized_env=True,  # if normalized env should be used
        scripted_policy=False,
        state_reconstruction_clip=None,
        task_variants=["velocity_forward", "velocity_backward",
                       "stand_back", "stand_front",
                       "goal_forward", "goal_backward",
                       "flip_forward", "jump"
        ]
    ),
    path_to_weights=None, # path to pre-trained weights to load into networks
    train_or_showcase='train',  # 'train' for train new policy, 'showcase_all' to load trained policy and showcase all test tasks, showcase_task_inference to calculate roll-outs from test-tasks and store the paths
    showcase_itr=10,  # training epoch from which to use weights of policy to showcase
    util_params=dict(
        base_log_dir='output',  # name of output directory
        use_gpu=True,  # set True if GPU available and should be used
        use_multiprocessing=False,  # set True if data collection should be parallelized across CPUs
        num_workers=2,  # number of CPU workers for data collection
        gpu_id=0,  # number of GPU if machine with multiple GPUs
        debug=False,  # debugging triggers printing and writes logs to debug directory
        plot=False,  # plot figures of progress for reconstruction and policy training
        tb_log_interval=100 # interval in which training progress is logged
    ),

    inference_option=['true_gmm'][0],
    task_distribution=[None, 'worst', 'best'][0],

    algo_params=dict(
        use_data_normalization=True,  # normalize data from experience
        sampling_mode=[None, 'linear'][0], # can be one of None, 'linear'
        use_fixed_seeding=True,  # seeding, make comparison more robust
        seed=0,  # seeding, make comparison more robust

        encoding_mode=['transitionSharedY', 'trajectory'][0],  # encoding mode, will be set automatically if wrong is chosen
        encoder_type=['gru', 'mlp', 'conv', 'transformer'][0],
        timestep_combination=['multiplication', 'network'][0],  # method of combining gaussian for timesteps when shared y is chosen, one of 'multiplication' or 'mlp'

        batch_size_rollout=512, # nr of paths processed in 'parallel'
        batch_size_policy=256,  # batch size trainer
        batch_size_reconstruction=4096,  # batch size trainer

        time_steps=64,  # timesteps before current to be considered for determine current task
        latent_size=8,  # dimension of the latent context vector z

        sac_layer_size=300,  # layer size for SAC networks, value 300 taken from PEARL
        max_replay_buffer_size=10000000,  # write as integer!

        permute_samples=False,  # if order of samples from previous timesteps should be permuted (avoid learning by heart)
        num_train_epochs=501,  # number of overall training epochs

        num_training_steps_policy=2048,  # number of policy training steps per training epoch
        num_training_steps_reconstruction=128,  # number of reconstruction training steps per training epoch
        num_train_tasks_per_episode=100,  # number of training tasks from which data is collected per training epoch
        num_transitions_per_episode=200,  # number of overall transitions per task while each epoch's data collection
        max_path_length=200,  # maximum number of transitions per episode in the environment

        # Augmented roll-out: Number of transitions will be calculated to match actual roll-out transitions
        augmented_start_percentage=1/5,  # starting point of augmentation relative to number of training epochs
        augmented_every=0,  # interval between training epochs in which augmented data is generated and policy is trained with it
        augmented_rollout_length=5,  # number of steps generated during augmented roll-out in sequence
        augmented_rollout_batch_size=1024,  # num samples generated in parallel

        num_eval_trajectories=1,  # number evaluation trajectories per test task
        test_evaluation_every=1,  # interval between epochs in which evaluation is performed

        num_showcase=16,  # how often trained policy is showcased

        policy_nets_lr=3e-4,  # Learning rate for policy and qf networks
        target_entropy_factor=1.0,  # target entropy from SAC
        automatic_entropy_tuning=False,  # use automatic entropy tuning in SAC
        sac_alpha=0.2,  # fixed alpha value in SAC when not using automatic entropy tuning

        snapshot_gap=10
    ),

    reconstruction_params=dict(
        use_state_diff=False,  # determines if decoder uses state or state difference as target

        num_classes=8,  # number of base classes in the class encoder

        lr_encoder=3e-4,  # learning rate decoder (ADAM) 3e-4 when combine with combination trainer,
        lr_decoder=3e-4,  # learning rate decoder (ADAM) 3e-4 when combine with combination trainer,

        alpha_kl_z=0.001,  # weighting factor KL loss of z distribution
        beta_euclid=0.0005,  # weighting factor euclid loss of z means
        gamma_sparsity=0.001,  # weighting factor sparsity of latent space

        use_regularization_loss=True,  # classification regularization
        regularization_lambda=0.1,  # weighting factor classification regularization

        net_complex_enc_dec=5.0,  # determines overall net complextity in encoder and decoder

        train_val_percent=0.8,  # percentage of train samples vs. validation samples
        eval_interval=50,  # interval for evaluation with validation data and possible early stopping
        early_stopping_threshold=500,  # minimal epochs before early stopping after new minimum was found
        temp_folder='.temp_cemrl'  # helper folder for storing encoder and decoder weights while training
    ),
    PCGrad_params = dict(
        use_PCGrad=False,
        PCGrad_option=['true_task',
                       'most_likely_task',
                       'random_prob_task'][0]
    )

)
