import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def main(path_to_folder, name=None, save_=True):

    fig_folder = os.path.join('log', 'figures', f'{time.strftime("%Y-%m-%d-%H_%M_%S")}_ti_evaluation{f"_{name}" if name is not None else ""}')
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

    # Start plotting
    print(f'Plotting ...')
    # Use Latex text
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    #matplotlib.rcParams['font.size'] = 24
    plt.style.use('seaborn')

    size_ = 32
    plt.rc('font', size=size_)  # controls default text sizes
    plt.rc('axes', titlesize=size_)  # fontsize of the axes title
    plt.rc('axes', labelsize=size_)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size_)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size_)  # fontsize of the tick labels
    plt.rc('legend', fontsize=size_)  # legend fontsize
    plt.rc('figure', titlesize=size_)  # fontsize of the figure title

    for fig_nr, (base_task, spec_dict) in enumerate(plotting_data.items()):
        title = task_dict[base_task] if task_dict is not None else f'Base Task {base_task}'
        plt.ioff()

        y_min, y_max = np.inf, -np.inf
        spec_max, spec_min = None, None
        x_max = 0
        for s_nr, spec in enumerate(sorted([float(k) for k in plotting_data[base_task].keys()])):
            spec = str(spec)
            p_len = 100 if title in ['velocity_forward', 'velocity_backward'] else len(plotting_data[base_task][spec])
            x_max = max([x_max, p_len])
            plt.plot(range(p_len), plotting_data[base_task][spec][range(p_len)], color=COLORS[s_nr])
            plt.plot([0, p_len], [float(spec), float(spec)], linestyle='--', color=COLORS[s_nr], label='target')

            y_min = min([y_min, plotting_data[base_task][spec].min()])
            y_max = max([y_max, plotting_data[base_task][spec].max()])
            spec_max = float(spec)
            if s_nr == 0:
                spec_min = float(spec)

        y_min = min([y_min, spec_min])
        y_max = max([y_max, spec_max])

        el = plt.plot([0], [0], linestyle='--', color='k', label=f'Target {target_map[title]}')
        plt.legend(handles=el, loc='upper right', bbox_to_anchor=(1.05, 1. if spec_max <= 0 else 0.2))

        # plt.title(title.replace('_', ' ').capitalize())
        plt.xlabel('Time Step $\it{t}$')
        plt.ylabel(target_map[title])
        plt.ylim([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)])
        plt.xlim([-1., x_max + 1])

        plt.gca().set_xticks((np.linspace(plt.xlim()[0] + 1, plt.xlim()[1] - 1, 5) * 10.).astype(np.int) / 10.)
        plt.gca().set_yticks((np.linspace(0. if spec_min > 0. else spec_min, 0. if spec_max < 0. else spec_max, 5) * 10.).astype(np.int) / 10.)

        # matplotlib.pyplot.gcf().set_size_inches(10, 5)

        if save_:
            plt.savefig(os.path.join(fig_folder, title.replace(' ', '_') + f'.pdf'), format='pdf', dpi=100,
                        bbox_inches='tight')
            plt.savefig(os.path.join(fig_folder, 'png', title.replace(' ', '_') + f'.png'), format='png', dpi=100,
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