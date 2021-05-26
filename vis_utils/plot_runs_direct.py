import sys, os, re, time
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as InterFun
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Define folder path for csvs
FOLDER_PATH_RUNS = os.path.join('output', 'cheetah-multi-task', '2021_04_26_8_task')
FOLDER_PATH_FIG = os.path.join('log', 'figures')

CONCAT_RUNS = False
SMOOTHING = 0.1

# Setup:
# List of run names that should be plotted
RUNS_TO_PLOT = [
    # 'MLP_5T',
    # 'GRU_5T',
    # 'CONV_5T',
    # 'TRANSFORMER_5T',

    # 'MLP_5T',
    # 'MLP_5T_PCGRAD',
    # 'MLP_10T',
    # 'MLP_10T_PCGRAD',

    # 'MLP_1T',
    # 'MLP_2T',
    # 'MLP_3T',
    # 'MLP_4T',
    # 'MLP_5T',
    # 'MLP_10T',
    # 'MLP_20T',

    # 'MLP_5T_LD1',
    # 'MLP_5T_LD2',
    # 'MLP_5T_LD3',
    # 'MLP_5T_LD4',

    # 'MLP_AT0S',
    # 'MLP_AT1S',
    # 'MLP_AT5S',
    # 'MLP_AT10S',
    # 'MLP_AT25S',

    # 'MLP_P_A0001_R01',
    # 'MLP_P_A0001_R0',
    # 'MLP_P_A001_R01',
    # 'MLP_P_A01_R01',

    # 'MLP_5_PRIOR_GMM',
    # 'MLP_5_TRUE_GMM',
    # 'MLP_5_COMB._ACTIV.',
    # 'MLP_5_DIRECT_ACTIV.',

    # 'AKZ0.001_BE0.01_GS0.01',
    # 'AKZ0.001_BE0.01_GS0.1',
    # 'AKZ0.001_BE0.1_GS0.01',
    # 'AKZ0.01_BE0.1_GS0.01',
    # 'AKZ0.01_BE0.1_GS0.1',
    # 'AKZ0.1_BE0.01_GS0.1',
    # 'AKZ0.1_BE0.1_GS0.01',
    # 'AKZ0.1_BE0.1_GS0.1'

    # 'SM_NONE',
    # 'SM_LINEAR',

    '8_TASK_GRU_64'
]

# Setup:
# DICT = {Title: regex, ...}
RUN_REGEX_DICT = {
    'MLP_1T': '.*cheetah_multi_task_io=prior_gmm_et=mlp_ts=1_ls=2_prior_gmm',
    'MLP_2T': '2021_02_27_20_07_39_prior_gmm_mlp_2',
    'MLP_3T': '2021_02_27_20_07_25_prior_gmm_mlp_3',
    'MLP_4T': '2021_02_27_20_07_12_prior_gmm_mlp_4',
    'MLP_5T': '.*cheetah_multi_task_io=prior_gmm_et=mlp_ts=5_ls=2_prior_gmm',
    'MLP_10T': '.*cheetah_multi_task_io=prior_gmm_et=mlp_ts=10_ls=2_prior_gmm',
    'MLP_20T': '2021_02_24_16_35_15_prior_gmm_mlp_20',

    'MLP_5T_LD1': '2021_02_27_20_05_41_prior_gmm_mlp_5_ld1',
    'MLP_5T_LD2': '.*cheetah_multi_task_io=prior_gmm_et=mlp_ts=5_ls=2_prior_gmm',
    'MLP_5T_LD3': '2021_02_27_20_05_51_prior_gmm_mlp_5_ld3',
    'MLP_5T_LD4': '.*cheetah_multi_task_io=prior_gmm_et=mlp_ts=5_ls=4_prior_gmm',

    'MLP_AT0S': '2021_02_25_17_05_02_prior_gmm_mlp_5',
    'MLP_AT1S': '2021_03_02_07_22_39_prior_gmm_mlp_at1',
    'MLP_AT5S': '2021_03_01_18_12_38_prior_gmm_mlp_at5',
    'MLP_AT10S': '2021_03_01_18_13_10_prior_gmm_mlp_at10',
    'MLP_AT25S': '2021_03_02_07_23_06_prior_gmm_mlp_at25',

    'MLP_P_A0001_R01': '2021_02_25_17_05_02_prior_gmm_mlp_5',
    'MLP_P_A0001_R0': '2021_03_01_03_25_34_prior_gmm_a_0001_r_0',
    'MLP_P_A001_R01': '2021_03_01_03_25_53_prior_gmm_a_001_r_01',
    'MLP_P_A01_R01': '2021_03_01_03_26_12_prior_gmm_a_01_r_01',

    'MLP_5_PRIOR_GMM'       : '.*cheetah_multi_task_io=prior_gmm_et=mlp_ts=5_ls=2_prior_gmm',
    'MLP_5_TRUE_GMM'        : '.*cheetah_multi_task_io=true_gmm_et=mlp_ts=5_ls=2_true_gmm',
    'MLP_5_COMB._ACTIV.'    : '.*cheetah_multi_task_io=comb_et=mlp_ts=5_ls=2_activation_combination',
    'MLP_5_DIRECT_ACTIV.'   : '.*cheetah_multi_task_io=direct_et=mlp_ts=5_ls=2_direct_activation',

    'GRU_5T': '.*cheetah_multi_task_io=prior_et=gru_ts=5_ls=2_prior_gmm',
    'GRU_10T': '2021_02_25_17_05_58_prior_gmm_gru_10',

    'CONV_5T': '.*cheetah_multi_task_io=prior_gmm_et=conv_ts=5_ls=2_prior_gmm',
    'CONV_10T': '2021_02_25_17_05_23_prior_gmm_conv_10',

    'TRANSFORMER_5T': '.*cheetah_multi_task_io=prior_et=transformer_ts=5_ls=2_prior_gmm',
    'TRANSFORMER_10T': '2021_02_26_15_39_57_prior_gmm_transformer_10',

    'MLP_5T_PCGRAD': '2021_03_01_03_15_43_prior_gmm_mlp_5_pcgrad',
    'MLP_10T_PCGRAD': '2021_02_26_16_42_03_prior_gmm_mlp_10_pcgrad',
    #'TIBIAMRL': 'PLACEHOLDER',

    'AKZ0.001_BE0.01_GS0.01': '.*cheetah_multi_task_akz~0.001_be~0.01_gs~0.01_prior_gmm',
    'AKZ0.001_BE0.01_GS0.1': '.*cheetah_multi_task_akz~0.001_be~0.01_gs~0.1_prior_gmm',
    'AKZ0.001_BE0.1_GS0.01': '.*cheetah_multi_task_akz~0.001_be~0.1_gs~0.01_prior_gmm',
    'AKZ0.01_BE0.1_GS0.01': '.*cheetah_multi_task_akz~0.01_be~0.1_gs~0.01_prior_gmm',
    'AKZ0.01_BE0.1_GS0.1': '.*cheetah_multi_task_akz~0.01_be~0.1_gs~0.1_prior_gmm',
    'AKZ0.1_BE0.01_GS0.1': '.*cheetah_multi_task_akz~0.1_be~0.01_gs~0.1_prior_gmm',
    'AKZ0.1_BE0.1_GS0.01': '.*cheetah_multi_task_akz~0.1_be~0.1_gs~0.01_prior_gmm',
    'AKZ0.1_BE0.1_GS0.1': '.*cheetah_multi_task_akz~0.1_be~0.1_gs~0.1_prior_gmm',

    'GRU_T10': '.*cheetah_multi_task_et~gru_ts~10_prior_gmm',
    'TRANSFORMER_T1': '.*cheetah_multi_task_et~transformer_ts~1_prior_gmm',
    'TRANSFORMER_T5': '.*cheetah_multi_task_et~transformer_ts~5_prior_gmm',

    'T_MULTIPLICATION': '.*cheetah_multi_task_tc~multiplication_prior_gmm',

    'SM_NONE': '.*cheetah_multi_task_td~None_sm~None_prior_gmm',
    'SM_LINEAR': '.*cheetah_multi_task_td~None_sm~linear_prior_gmm',

    'TD_NONE_SMNONE': '.*cheetah_multi_task_td~None_sm~None_prior_gmm',
    'TD_NONE_SMLINEAR': '.*cheetah_multi_task_td~None_sm~linear_prior_gmm',

    'TD_WORST_SMNONE': '.*cheetah_multi_task_td~worst_sm~None_prior_gmm',

    '8_TASK_GRU_64': '.*cheetah_multi_task_ts~64_true_gmm',

}

# Setup:
# DICT = {run name: [(Title, tag), ...], ...}
RUN_TAGS_DICT = {
    'default': [
        ('Evaluation Test ND Average Reward', 'evaluation/nd_test/average_reward'),
        ('Evaluation Test ND Max Reward', 'evaluation/nd_test/max_reward'),
        ('Evaluation Test ND Min Reward', 'evaluation/nd_test/min_reward'),
        ('Evaluation Test ND Std Reward', 'evaluation/nd_test/std_reward'),
        ('Evaluation Test ND Success Rate', 'evaluation/nd_test/success_rate'),

        ('Evaluation Test Average Reward', 'evaluation/test/average_reward'),
        ('Evaluation Test Max Reward', 'evaluation/test/max_reward'),
        ('Evaluation Test Min Reward', 'evaluation/test/min_reward'),
        ('Evaluation Test Std Reward', 'evaluation/test/std_reward'),
        ('Evaluation Test Success Rate', 'evaluation/test/success_rate'),

        ('Evaluation Training Average Reward', 'evaluation/train/average_reward'),
        ('Evaluation Training Max Reward', 'evaluation/train/max_reward'),
        ('Evaluation Training Min Reward', 'evaluation/train/min_reward'),
        ('Evaluation Training Std Reward', 'evaluation/train/std_reward'),
        ('Evaluation Training Success Rate', 'evaluation/train/success_rate'),

        ('Policy Training Alpha Loss', 'rl/alpha'),
        ('Policy Training Policy Loss', 'rl/policy_loss'),
        ('Policy Training QF1 Loss', 'rl/qf1_loss'),
        ('Policy Training QF2 Loss', 'rl/qf2_loss'),

        ('Task Inference Training Mixture Model Combined Loss', 'training/ti_mixture_loss'),
        ('Task Inference Training Mixture Model Elbo Loss', 'training/ti_mixture_elbo_loss'),
        ('Task Inference Training Mixture Model State Loss', 'training/ti_mixture_state_losses'),
        ('Task Inference Training Mixture Model Reward Loss', 'training/ti_mixture_reward_losses'),
        ('Task Inference Training Mixture Model Regularization Loss', 'training/ti_mixture_regularization_loss'),
        ('Task Inference Training Mixture Model Class Activation Accuracy', 'training/ti_classification_acc'),
        ('Task Inference Training Mixture Model Clustering Loss', 'training/ti_mixture_clustering_losses')

    ],
}


def main(run_name=None, interpolation_type='scipy', smooth=True, format_='pdf', plot_std=True, save_=True,
         summary_pref='', fit_plt=False):

    global RUN_REGEX_DICT
    global FOLDER_PATH_RUNS
    global RUNS_TO_PLOT

    if run_name is not None:
        run_name = run_name if run_name[-1] != '/' else run_name[:-1]
        head, tail = os.path.split(run_name)
        if len(head) > 0:
            FOLDER_PATH_RUNS = head
            RUN_REGEX_DICT = {
                'TIBIAMRL': tail,
            }
        else:
            RUN_REGEX_DICT = {
                'TIBIAMRL': run_name,
            }
        RUNS_TO_PLOT = ['TIBIAMRL']


    # Prepare data
    data_dict = {}
    # Get all folders in folder
    folders = sorted([d for d in os.listdir(FOLDER_PATH_RUNS) if os.path.isdir(os.path.join(FOLDER_PATH_RUNS, d))])
    for run_name in RUNS_TO_PLOT:
        for folder in folders:
            if re.match(RUN_REGEX_DICT[run_name], folder) is not None:
                (dirpath, subfolders, subfiles) = next(os.walk(os.path.join(FOLDER_PATH_RUNS, folder, 'tensorboard')))
                #(dirpath, _, subsubfiles) = next(os.walk(os.path.join(dirpath, subfolders[0])))
                # Add tf events from first subfolder
                print(f'Reading in events of {[file for file in subfiles if "events.out" in file][0]} [{folder}]')
                acc = EventAccumulator(os.path.join(dirpath, [file for file in subfiles if 'events.out' in file][0])).Reload()
                # Gather all info for given tags
                for title, tag in RUN_TAGS_DICT[run_name if run_name in RUN_TAGS_DICT.keys() else 'default']:
                    try:
                        list_of_events = acc.Scalars(summary_pref + tag)
                    except Exception as e:
                        print(f'\tAcquiring data for tag "{summary_pref + tag}" went wrong! ({e})')
                        continue

                    _, steps, values = list(zip(*map(lambda x: x._asdict().values(), list_of_events)))

                    df = pd.DataFrame(data=np.array([np.array(steps), np.array(values)]).T, columns=['Step', 'Value'])
                    df.drop_duplicates(subset='Step', keep='last', inplace=True)

                    # Add dfs to data_dict
                    if title in data_dict.keys():
                        if not CONCAT_RUNS:
                            if run_name in data_dict[title].keys():
                                data_dict[title][run_name].append(df)
                            else:
                                data_dict[title][run_name] = [df]
                        else:
                            last_step = data_dict[title][run_name][0]['Step'].to_numpy()[-1]
                            df['Step'] += last_step
                            data_dict[title][run_name][0] = data_dict[title][run_name][0].append(df)
                    else:
                        data_dict[title] = {run_name: [df]}

    print(f'Using {["own", "InterpolatedUnivariateSpline (scipy)"][int(interpolation_type == "scipy")]} interpolation method to patch missing data in some plots')

    # Find min length for plotting only valid data and transform pd frames in numpy arrays
    for title in data_dict.keys():

        # Find corresponding values and interpolate
        for run_name in list(data_dict[title].keys()):

            # Only interpolate in case we have multiple runs that need to be averaged
            min_steps = data_dict[title][run_name][0]['Step'].to_numpy()
            if len(data_dict[title][run_name]) > 1:

                temp_l = np.array([df['Step'].to_numpy()[-1] for df in data_dict[title][run_name]])
                min_steps = data_dict[title][run_name][temp_l.argmin()]['Step'].to_numpy()

                if interpolation_type == 'scipy':
                    for ind, df in enumerate(data_dict[title][run_name]):
                        interpolation_function = InterFun(df['Step'].to_numpy(), df['Value'].to_numpy())
                        data_dict[title][run_name][ind] = interpolation_function(min_steps)
                elif interpolation_type == 'own':
                    for ind, df in enumerate(data_dict[title][run_name]):
                        steps, values = df['Step'].to_numpy(), df['Value'].to_numpy()
                        bigger_array = np.zeros_like(min_steps, dtype=np.float)
                        for arr_ind, step in enumerate(min_steps):
                            bigger_array[arr_ind] = values[np.where(steps >= step)[0][0]] if np.sum(steps >= step) > 0 else values[-1]
                        data_dict[title][run_name][ind] = bigger_array

            else:
                data_dict[title][run_name][0] = data_dict[title][run_name][0]['Value'].to_numpy()

            data_dict[title][run_name + '_steps'] = min_steps

    # Start plotting
    print(f'Plotting ...')
    # Use Latex text
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # Make folder in case not yet existing
    file_name = "_".join([RUN_REGEX_DICT[run_name] for run_name in RUNS_TO_PLOT])
    fig_folder = os.path.join(FOLDER_PATH_FIG, f'{time.strftime("%Y-%m-%d-%H_%M_%S")}_{file_name if len(RUNS_TO_PLOT) < 2 else "comparison"}_smoothing{SMOOTHING}')
    if not os.path.isdir(fig_folder) and save_:
        os.mkdir(fig_folder)

    for title in data_dict.keys():
        plot_title = ('Comparison ' if len(data_dict[title]) > 2 else '') + title
        plt.ioff()
        plt.title(plot_title)
        max_mean, min_mean = -np.inf, np.inf
        for run_name in data_dict[title].keys():
            if '_steps' in run_name:
                continue
            data_arr = np.array(data_dict[title][run_name])
            steps = data_dict[title][run_name + '_steps']
            mean = data_arr.mean(axis=0) if not smooth else smooth_values(data_arr.mean(axis=0))
            std = np.sqrt(data_arr.var(axis=0))
            plt.plot(steps, mean)
            if plot_std: plt.fill_between(steps, mean + std, mean - std, alpha=0.3)

            max_mean = mean.max() if max_mean < mean.max() else max_mean
            min_mean = mean.min() if min_mean > mean.min() else min_mean

        if fit_plt: plt.ylim([min_mean, max_mean])
        plt.legend([f'{el}_[{len(data_dict[title][el])}]' for el in data_dict[title].keys() if '_steps' not in el],
                   bbox_to_anchor=(1, 1), loc='upper left')
        plt.xlabel('Steps')
        plt.ylabel(title)

        # Always show 0
        # y_min, y_max = plt.gca().get_ylim()
        # if y_min > 0 and not fit_plt:
        #     plt.ylim([0, y_max])

        # Save or show
        if save_:
            plt.savefig(os.path.join(fig_folder, plot_title + '.' + format_), format=format_, dpi=100,
                        bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def smooth_values(scalars, weight=None):                        # Scalars as np.array, weight between 0 and 1
    if weight is None: weight = SMOOTHING

    last = scalars[0]                                           # First value in the plot (first timestep)
    smoothed = np.zeros_like(scalars)
    for idx, point in enumerate(scalars):
        smoothed_val = last * weight + (1 - weight) * point     # Calculate smoothed value
        smoothed[idx] = smoothed_val                            # Save it
        last = smoothed_val                                     # Anchor the last smoothed value

    return np.array(smoothed)

if __name__ == '__main__':
    if len(sys.argv) > 0:
        main(*sys.argv[1:])
    else:
        main()
