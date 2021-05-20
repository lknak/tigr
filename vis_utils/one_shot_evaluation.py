import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

LINE_STYLES = ['-', ':', '--', '-.', '-', ':', '--', '-.']
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

    plotting_data = {}
    for path in paths:
        plotting_data[len(plotting_data.keys())] = {0: {
            'base_task': np.array([el['base_task'] for el in path]),
            'specification': np.array([
                                          f"{int(el['specification'] * 100) / 100 if type(el['specification']) is float or type(el['specification']) is np.float64 else [int(k * 100) / 100 for k in el['specification']]}"
                                          for el in path]),
            'reward': np.array([el['reward'] for el in path])
        }}


    # Start plotting
    print(f'Plotting ...')
    # Use Latex text
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    #matplotlib.rcParams['font.size'] = 24
    plt.style.use('seaborn')

    plt.rc('font', size=22)  # controls default text sizes
    plt.rc('axes', titlesize=22)  # fontsize of the axes title
    plt.rc('axes', labelsize=22)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=22)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=22)  # fontsize of the tick labels
    plt.rc('legend', fontsize=22)  # legend fontsize
    plt.rc('figure', titlesize=22)  # fontsize of the figure title

    for fig_nr, (base_task, spec_dict) in enumerate(plotting_data.items()):
        plt.ioff()

        ncol = 1
        x_max = 0
        legend_elements = []
        for s_nr, spec in enumerate(sorted(plotting_data[base_task].keys())):
            x_max = len(plotting_data[base_task][spec]['reward'])
            ubt = np.unique(plotting_data[base_task][spec]["base_task"])
            ubt = [ell[0] for ell in [np.argwhere(plotting_data[base_task][spec]["base_task"] == el) for el in ubt] if len(ell) > 0]
            usp = np.unique(plotting_data[base_task][spec]["specification"])
            usp = [ell[0] for ell in [np.argwhere(plotting_data[base_task][spec]["specification"] == el) for el in usp] if len(ell) > 0]
            split_points = np.unique(ubt + usp)
            ncol = len(split_points)
            for l_nr, i in enumerate(range(len(split_points))):
                bt = plotting_data[base_task][spec]["base_task"][split_points[i]]
                el = plt.plot(range(max([split_points[i] - 1, 0]), len(plotting_data[base_task][spec]['reward']) if len(split_points) - 2 < i else split_points[i + 1]),
                              plotting_data[base_task][spec]['reward'][max([split_points[i] - 1, 0]):(len(plotting_data[base_task][spec]['reward']) if len(split_points) - 2 < i else split_points[i + 1])],
                              label=f'{" ".join([s.capitalize() for s in str(task_dict[bt] if task_dict is not None else bt).split("_")])} ({plotting_data[base_task][spec]["specification"][split_points[i]]})',
                              linestyle=LINE_STYLES[l_nr],
                              color=COLORS[s_nr])
                legend_elements.append(el)

        plt.legend(loc='upper left', bbox_to_anchor=(0.01, 1., 0.98, 0.), ncol=ncol, mode="expand")

        title = 'Non-Stationary Environment'
        plt.xlabel('Time Step $\it{t}$')
        plt.ylabel('Reward $\it{r}$')
        plt.ylim([-1.5, 0.5])
        plt.xlim([-1., x_max + 1])
        matplotlib.pyplot.gcf().set_size_inches(30, 5)

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