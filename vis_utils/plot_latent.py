import sys, os, re, time, platform
import json, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from matplotlib.lines import Line2D
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


DATA_DIR = os.path.join('output', 'comparisons', 'cheetah-multi-task', 'ours', '2021_05_10_17_35_02_cheetah_multi_task_ts~64_true_gmm', 'tensorboard')

# DATA_DIR = os.path.join('output', 'clustering', '2021_05_18_16_41_35_cheetah_multi_task_ts~64_no_cl_true_gmm', 'tensorboard')
# DATA_DIR = os.path.join('output', 'clustering', '2021_05_18_16_41_40_cheetah_multi_task_ts~64_no_spars_true_gmm', 'tensorboard')
# DATA_DIR = os.path.join('output', 'clustering', '2021_05_18_16_41_45_cheetah_multi_task_ts~64_no_euclid_true_gmm', 'tensorboard')

FIG_DIR = os.path.join('log', 'encodings')

PLOT_LIST = []

MARKERS = ['.', '^', 's', 'p', '*', 'X', 'h', 'd', '+', 'P']
# COLORS = ['tab:blue', , 'tab:gray', 'tab:olive', 'tab:cyan']

def main(run_name=None, save=True, show_='last', use_tsne=True, DIM_RED=2):
    global DATA_DIR
    if run_name is not None:
        head, tail = os.path.split(run_name)
        if len(head) > 0:
            DATA_DIR = os.path.join(run_name, 'tensorboard')
        else:
            DATA_DIR = os.path.join('output', 'cheetah-multi-task', run_name, 'tensorboard')

    fig_folder = os.path.join(FIG_DIR, f'{time.strftime("%Y-%m-%d-%H_%M_%S")}_{os.path.split(DATA_DIR)[-1]}')
    if not os.path.isdir(fig_folder) and save:
        os.mkdir(fig_folder)

    epoch_dict = {}

    folders_ = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    maxlen = max([len(f) for f in folders_])
    folders_ = sorted([f.zfill(maxlen) for f in folders_])
    for folder in (folders_ if show_ == 'all' else [folders_[-1]]):
        if re.match('[0-9]+', folder):

            step_ = re.findall('([0-9]+)', folder)[0]

            with open(os.path.join(DATA_DIR, folder, 'default', 'metadata.tsv'), newline='') as f:
                r = csv.reader(f, delimiter='\t')
                metadata = [row_ for row_ in r]
            with open(os.path.join(DATA_DIR, folder, 'default', 'tensors.tsv'), newline='') as f:
                r = csv.reader(f, delimiter='\t')
                data = [row_ for row_ in r]

            # Convert to np
            metadata = np.array(metadata)
            data = np.array(data, dtype=np.float)

            # Bring data to 3D
            pcad = ''
            if data.shape[1] < 3:
                temp = np.zeros([data.shape[0], 3])
                temp[:, 0:data.shape[1]] = data
                data = temp

            elif data.shape[1] > 3:
                if use_tsne:
                    print(f'Performing T-SNE from {data.shape[1]} DIM to {DIM_RED}')
                    pcad = f' (Using T-SNE From {data.shape[1]} To {DIM_RED} Dimensions)'
                    true_tasks = np.array([s[0].split('[')[0].strip() for s in metadata])
                    unique_tasks = np.sort(np.unique(true_tasks))
                    data = TSNE(n_components=DIM_RED, init='pca').fit_transform(data)
                else:
                    print(f'Performing PCA from {data.shape[1]} DIM to {DIM_RED}')
                    pcad = f' (Using PCA From {data.shape[1]} To {DIM_RED} Dimensions)'
                    data = perform_pca(data)

            epoch_dict[step_] = [metadata, data, pcad]

    m_l = max([len(k) for k in epoch_dict.keys()])

    # Map
    class_map = {'flip_forward': 'Front flip', 'jump': 'Jump',
                 'goal_backward': 'Goal in back', 'goal_forward': 'Goal in front',
                 'stand_back': 'Back stand', 'stand_front': 'Front stand',
                 'velocity_backward': 'Run forward', 'velocity_forward': 'Run backward'}

    # Plotting
    # Use Latex text
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    plt.style.use('seaborn')

    size_ = 31
    plt.rc('font', size=size_)  # controls default text sizes
    plt.rc('figure', titlesize=size_)  # fontsize of the figure title
    plt.rc('axes', titlesize=size_)  # fontsize of the axes title
    plt.rc('axes', labelsize=size_)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size_*0.8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size_*0.8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=size_*0.8)  # legend fontsize

    for step_ in sorted(epoch_dict.keys()):

        metadata, values, pcad = epoch_dict[step_]

        true_tasks = np.array([s[0].split('[')[0].strip() for s in metadata])
        unique_tasks = np.sort(np.unique(true_tasks))
        max_len_task = max([len(t) for t in unique_tasks])
        specs = np.array([float(re.findall('[0-9]+[.]*[0-9]*', s[0])[0]) for s in metadata])
        predicted_tasks = np.array([s[0].split('->')[1].strip() for s in metadata])

        fig = plt.figure()

        if values.shape[1] == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_aspect('auto')
            # ax.set_title(f'Latent Encodings{pcad} For GMM Training Step {int(step_)}', y=1.08)
        else:
            ax = plt.gca()
            # ax.set_aspect('auto')
            ax.set_title(f'Half-Cheetah 8-Task')

        legend_elements = []
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors += ['#db8233', '#8c564b', '#e377c2']
        for i, target_class in enumerate(unique_tasks):
            for j, pred_class in enumerate(unique_tasks):
                match_values = values[true_tasks == target_class][predicted_tasks[true_tasks == target_class] == pred_class]
                acc = (predicted_tasks[true_tasks == target_class] == pred_class).mean() * 100

                if match_values.shape[1] == 3:
                    el = ax.scatter(match_values[:, 0], match_values[:, 1], match_values[:, 2],
                                    label=class_map[str(target_class)],
                                    marker=MARKERS[j],
                                    color=colors[i % len(colors)])
                else:
                    el = ax.scatter(match_values[:, 0], match_values[:, 1],
                                    label=class_map[str(target_class)],
                                    marker=MARKERS[j],
                                    color=colors[i % len(colors)])

                if target_class == pred_class:
                    legend_elements.append(el)


        if values.shape[1] == 3:
            ax.set_xlabel('Latent Dim 1')
            ax.set_ylabel('Latent Dim 2')
            ax.set_zlabel('Latent Dim 3')
            # Equal scaling
            x_lim_min, x_lim_max = ax.get_xlim()
            y_lim_min, y_lim_max = ax.get_ylim()
            z_lim_min, z_lim_max = ax.get_zlim()
            max_ = np.array([(x_lim_max - x_lim_min), (y_lim_max - y_lim_min), (z_lim_max - z_lim_min)]).max()
            ax.set_xlim(x_lim_min - (max_ - (x_lim_max - x_lim_min)) / 2, x_lim_max + (max_ - (x_lim_max - x_lim_min)) / 2)
            ax.set_ylim(y_lim_min - (max_ - (y_lim_max - y_lim_min)) / 2, y_lim_max + (max_ - (y_lim_max - y_lim_min)) / 2)
            ax.set_zlim(z_lim_min - (max_ - (z_lim_max - z_lim_min)) / 2, z_lim_max + (max_ - (z_lim_max - z_lim_min)) / 2)
        else:
            ax.set_xlabel('Latent Dim 1')
            ax.set_ylabel('Latent Dim 2')

            ts = ax.get_xticks()
            if len(ts) < 5:
                ax.set_xticks(np.linspace(min(ts), max(ts), 5))
            ts = ax.get_yticks()
            if len(ts) < 5:
                ax.set_yticks(np.linspace(min(ts), max(ts), 5))

        # ax.view_init(-90, 90)

        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.9, 0.5), markerscale=1.5)

        fig.set_size_inches(10., 7.1) #8 task

        if save:
            plt.savefig(os.path.join(fig_folder, f'encodings.pdf'), format='pdf', dpi=100, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

        print(f'Created plot for {DATA_DIR}')


def perform_pca(values):

    # sample points equally for all gaussians
    x = np.copy(values)

    # centering the data
    x -= np.mean(x, axis=0)

    cov = np.cov(x, rowvar=False)

    evals, evecs = LA.eigh(cov)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx[:3]]

    return np.dot(x, evecs)


if __name__ == '__main__':
    main(*sys.argv[1:])
