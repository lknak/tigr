import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def main(path_to_folder, use_plots=None, plot_type=0, name=None, save_=True):

    use_plots = None if use_plots in [None, ''] else [int(n) for n in use_plots.split(';')]
    plot_type = int(plot_type)

    fig_folder = os.path.join('log', 'figures', f'{time.strftime("%Y-%m-%d-%H_%M_%S")}_ti_one_plot{f"_{name}" if name is not None else ""}')
    if not os.path.isdir(fig_folder) and save_:
        os.mkdir(fig_folder)
        os.mkdir(os.path.join(fig_folder, 'png'))

    with open(os.path.join(path_to_folder, 'roll_out_results.json'), 'r') as f:
        # Copy so not both changed during updates
        paths = npify_dict(json.load(f))

    task_dict = None
    if os.path.exists(os.path.join(path_to_folder, 'task_dict.json')):
        with open(os.path.join(path_to_folder, 'task_dict.json'), 'r') as f:
            # Copy so not both changed during updates
            task_dict = json.load(f)
            task_dict = {el: key for key, el in task_dict.items()} if task_dict is not None else None

    plotting_data = {}
    for path in paths:
        base_task = path['true_tasks'][0, 0]['base_task']
        spec = path['true_tasks'][0, 0]['specification']
        spec = f"{int(spec * 100) / 100 if type(spec) is float or type(spec) is np.float64 else [int(k * 100) / 100 for k in spec]}"

        task_map = {'velocity_forward': 8, 'velocity_backward': 8,
                    'goal_forward': 17, 'goal_backward': 17,
                    'flip_forward': 9, 'jump': 10,
                    'stand_front': 1, 'stand_back': 1}

        rewards = path['observations'][:, task_map[task_dict[base_task]]]

        if base_task in plotting_data.keys():
            plotting_data[base_task][spec] = rewards
        else:
            plotting_data[base_task] = {spec: rewards}


    target_map = {'velocity_forward': 'Velocity', 'velocity_backward': 'Velocity',
                'goal_forward': 'Distance', 'goal_backward': 'Distance',
                'flip_forward': 'Velocity', 'jump': 'Velocity',
                'stand_front': 'Angle', 'stand_back': 'Angle'}

    class_map = {'flip_forward': 'Front flip', 'jump': 'Jump',
                 'goal_backward': 'Goal in back', 'goal_forward': 'Goal in front',
                 'stand_back': 'Back stand', 'stand_front': 'Front stand',
                 'velocity_backward': 'Run backward', 'velocity_forward': 'Run forward'}

    # Start plotting
    print(f'Plotting ...')

    plt.style.use('seaborn')

    # Use Latex text
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    #matplotlib.rcParams['font.size'] = 24

    if plot_type == 0:
        size_ = 32
    elif plot_type == 1:
        size_ = 38
    else:
        size_ = 12
    plt.rc('figure', titlesize=size_*1.0)  # fontsize of the figure title
    plt.rc('axes', titlesize=size_*1.0)  # fontsize of the axes title
    plt.rc('axes', labelsize=size_*1.0)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size_*0.8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size_*0.8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=size_*0.8)  # legend fontsize


    n_plots = len(plotting_data.keys()) if use_plots is None else len(use_plots)
    fig, axs = plt.subplots(int(np.ceil(n_plots / 4)), min([n_plots, 4]))
    axs = axs.reshape(-1) if type(axs) is np.ndarray else [axs]

    for fig_nr, base_task in enumerate(plotting_data.keys()) if use_plots is None else enumerate(use_plots):
        title = task_dict[base_task] if task_dict is not None and base_task in task_dict else f'Base Task {base_task}'

        y_min, y_max = np.inf, -np.inf
        spec_max, spec_min = None, None
        x_max = 0
        for s_nr, spec in enumerate(sorted([float(k) for k in plotting_data[base_task].keys()])):
            spec = str(spec)
            p_len = 100 if title in ['velocity_forward', 'velocity_backward'] else 50 if title in ['jump'] else len(plotting_data[base_task][spec])
            x_max = max([x_max, p_len])
            axs[fig_nr].plot(range(p_len), plotting_data[base_task][spec][range(p_len)], color=COLORS[s_nr], linewidth=size_ * 0.2)
            axs[fig_nr].plot([0, p_len], [float(spec), float(spec)], linestyle='--', color=COLORS[s_nr], label='target', linewidth=size_ * 0.2)

            y_min = min([y_min, plotting_data[base_task][spec].min()])
            y_max = max([y_max, plotting_data[base_task][spec].max()])
            spec_max = float(spec)
            if s_nr == 0:
                spec_min = float(spec)

        y_min = min([y_min, spec_min])
        y_max = max([y_max, spec_max])

        el = axs[fig_nr].plot([0], [0], linestyle='--', color='k', label=f'Target {target_map[title]}', linewidth=size_ * 0.2)
        axs[fig_nr].legend(handles=el, loc='upper right', bbox_to_anchor=(1.0, 1. if spec_max <= 0 else 0.15))

        axs[fig_nr].set_title(class_map[title])
        axs[fig_nr].set_xlabel('Time Step $\it{t}$')
        axs[fig_nr].set_ylabel(target_map[title])

        axs[fig_nr].set_ylim([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)])
        axs[fig_nr].set_xlim([-1., x_max + 1])

        axs[fig_nr].set_xticks((np.linspace(axs[fig_nr].get_xlim()[0] + 1, axs[fig_nr].get_xlim()[1] - 1, 5) * 10.).astype(np.int) / 10.)
        axs[fig_nr].set_yticks((np.linspace(0. if spec_min > 0. else spec_min, 0. if spec_max < 0. else spec_max, 5) * 10.).astype(np.int) / 10.)

    if plot_type == 0:
        plt.subplots_adjust(hspace=0.4)
        fig.set_size_inches(45, 14.5)
    elif plot_type == 1:
        fig.set_size_inches(42, 6.0)
    else:
        fig.set_size_inches(35, 6.0)

    if save_:
        plt.savefig(os.path.join(fig_folder, 'ti_one_plot.pdf'), format='pdf', dpi=100,
                    bbox_inches='tight')
        plt.savefig(os.path.join(fig_folder, 'png', 'ti_one_plot.png'), format='png', dpi=100,
                    bbox_inches='tight')
    else:
        plt.show()
    plt.close()


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