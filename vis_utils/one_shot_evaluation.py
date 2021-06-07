import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

LINE_STYLES = ['-', ':']#, '--', '-.', '-', ':', '--', '-.']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def main(path_to_folder, name=None, save_=True):

    fig_folder = os.path.join('log', 'figures', f'{time.strftime("%Y-%m-%d-%H_%M_%S")}_one_shot_evaluation{f"_{name}" if name is not None else ""}')
    if not os.path.isdir(fig_folder) and save_:
        os.mkdir(fig_folder)
        os.mkdir(os.path.join(fig_folder, 'png'))

    with open(os.path.join(path_to_folder, 'non_stationary_results.json'), 'r') as f:
        # Copy so not both changed during updates
        paths = npify_dict(json.load(f))

    task_dict = None
    if os.path.exists(os.path.join(path_to_folder, 'task_dict.json')):
        with open(os.path.join(path_to_folder, 'task_dict.json'), 'r') as f:
            # Copy so not both changed during updates
            task_dict = json.load(f)
            task_dict = {el: key for key, el in task_dict.items()} if task_dict is not None else None

    task_map = {'velocity_forward': 8, 'velocity_backward': 8,
                'goal_forward': 17, 'goal_backward': 17,
                'flip_forward': 9, 'jump': 10,
                'stand_front': 1, 'stand_back': 1}

    target_map = {'velocity_forward': 'Velocity', 'velocity_backward': 'Velocity',
                'goal_forward': 'Distance', 'goal_backward': 'Distance',
                'flip_forward': 'Velocity', 'jump': 'Velocity',
                'stand_front': 'Angle', 'stand_back': 'Angle'}

    class_map = {'flip_forward': 'Front flip', 'jump': 'Jump',
                 'goal_backward': 'Goal in back', 'goal_forward': 'Goal in front',
                 'stand_back': 'Back stand', 'stand_front': 'Front stand',
                 'velocity_backward': 'Run backward', 'velocity_forward': 'Run forward'}

    plotting_data = {}
    for path in paths:
        plotting_data[len(plotting_data.keys())] = {0: {
            'base_task': np.array([el['base_task'] for el in path]),
            'specification': np.array([
                                          f"{int(el['specification'] * 100) / 100 if type(el['specification']) is float or type(el['specification']) is np.float64 else [int(k * 100) / 100 for k in el['specification']]}"
                                          for el in path]),
            'reward': np.array([el['reward'] for el in path]),
            'value': np.array([el['observation'][task_map[task_dict[el['base_task']]]] for el in path])
        }}

    # Start plotting
    print(f'Plotting ...')

    plt.style.use('seaborn')

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    size_ = 22

    # plt.rc('font', size=size_)  # controls default text sizes
    plt.rc('figure', titlesize=size_)  # fontsize of the figure title
    plt.rc('axes', titlesize=size_)  # fontsize of the axes title
    plt.rc('axes', labelsize=size_)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size_*0.8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size_*0.8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=size_*0.8)  # legend fontsize

    title = 'Non-Stationary Environment'

    for fig_nr, (base_task, spec_dict) in enumerate(plotting_data.items()):
        plt.ioff()
        s_nr = 0
        spec = list(plotting_data[base_task].keys())[0]

        ubt = np.unique(plotting_data[base_task][spec]["base_task"])
        ubt = [ell[0] for ell in [np.argwhere(plotting_data[base_task][spec]["base_task"] == el) for el in ubt] if len(ell) > 0]
        usp = np.unique(plotting_data[base_task][spec]["specification"])
        usp = [ell[0] for ell in [np.argwhere(plotting_data[base_task][spec]["specification"] == el) for el in usp] if len(ell) > 0]
        split_points = np.unique(ubt + usp)
        ncol = len(split_points)

        fig, axs = plt.subplots(1, ncol)
        axs = axs if type(axs) is np.ndarray else [axs]

        for l_nr, i in enumerate(range(len(split_points))):
            bt = plotting_data[base_task][spec]["base_task"][split_points[i]]
            x_min = max([split_points[i] - 1, 0])
            x_min = x_min + 1 if x_min > 0 else x_min
            x_max = len(plotting_data[base_task][spec]['value']) if len(split_points) - 2 < i else split_points[i + 1]
            spec_ = plotting_data[base_task][spec]["specification"][split_points[i]]
            axs[l_nr].plot(range(x_min, x_max),
                          plotting_data[base_task][spec]['value'][x_min:x_max],
                          #label=f'{" ".join([s.capitalize() for s in str(task_dict[bt] if task_dict is not None else bt).split("_")])} ({spec_})',
                          linestyle=LINE_STYLES[s_nr % len(LINE_STYLES)],
                          color=COLORS[l_nr % len(COLORS)],
                          linewidth=size_ * 0.2)
            target = axs[l_nr].plot([x_min, x_max], [float(spec_), float(spec_)],
                                    linestyle='--', color=COLORS[l_nr % len(COLORS)], label=f'Target {target_map[task_dict[bt]]}', linewidth=size_ * 0.2)
            axs[l_nr].set_title(f'{class_map[task_dict[bt]]}')# ({spec_})
            # axs[l_nr].set_yticklabels([])
            axs[l_nr].set_ylabel(target_map[task_dict[bt]])
            axs[l_nr].set_xticks(np.linspace(x_min, x_max, 3).astype(np.int))

            axs[l_nr].legend(handles=target)#, loc='upper right', bbox_to_anchor=(1.07, [1.07, 0.2][np.argmin(np.abs(float(spec_) - np.array(axs[l_nr].get_ylim())))]))

        # fig.suptitle(title, y=1.08)
        fig.text(0.5, -0.1, 'Time Step $\it{t}$', ha='center', fontsize=size_)

        fig.set_size_inches(45, 4)

        if save_:
            plt.savefig(os.path.join(fig_folder, title.replace(' ', '_') + f'_{fig_nr}.pdf'), format='pdf', dpi=100,
                        bbox_inches='tight')
            plt.savefig(os.path.join(fig_folder, 'png', title.replace(' ', '_') + f'_{fig_nr}.png'), format='png', dpi=100,
                        bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    pass


def npify_dict(d: dict):
    if type(d) is dict:
        for k, v in d.items():
            if type(v) is dict:
                d[k] = npify_dict(v)
            else:
                d[k] = np.asarray(v)
    elif type(d) is list:
        for i, el in enumerate(d):
            if type(el) is dict:
                d[i] = npify_dict(el)
            else:
                d[i] = np.asarray(el)
    else:
        raise ValueError(d)
    return d


if __name__ == '__main__':
    main(*sys.argv[1:])