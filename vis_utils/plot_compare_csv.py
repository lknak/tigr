import sys, os
import numpy as np
import matplotlib.pyplot as plt
import json, csv
import glob, time
import matplotlib
from scipy.interpolate import InterpolatedUnivariateSpline as InterFun


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def unique(l):
    return list(set(l))


def load_params(params_json_path):
    with open(params_json_path, 'r') as f:
        data = json.loads(f.read())
        if "args_data" in data:
            del data["args_data"]
        if "exp_name" not in data:
            data["exp_name"] = params_json_path.split("/")[-2]
    return data


def flatten_dict(d):
    flat_params = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            v = flatten_dict(v)
            for subk, subv in flatten_dict(v).items():
                flat_params[k + "." + subk] = subv
        else:
            flat_params[k] = v
    return flat_params


def load_progress(progress_csv_path):
    print("Reading %s" % progress_csv_path)
    entries = dict()
    if progress_csv_path.split('.')[-1] == "csv":
        delimiter = ','
    else:
        delimiter = '\t'
    with open(progress_csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            for k, v in row.items():
                if k not in entries:
                    entries[k] = []
                try:
                    entries[k].append(float(v))
                except:
                    entries[k].append(0.)
    entries = dict([(k, np.array(v)) for k, v in entries.items()])
    return entries


def load_exps_data(
        exp_folder_paths,
        data_filename='progress.csv',
        params_filename='params.json',
        disable_variant=False,
):
    # exps = []
    # for exp_folder_path in exp_folder_paths:
    #     exps += [x[0] for x in os.walk()]
    exps_data = []
    for exp in glob.iglob(exp_folder_paths):
        try:
            exp_path = exp
            params_json_path = os.path.join(exp_path, params_filename)
            variant_json_path = os.path.join(exp_path, "variant.json")
            progress_csv_path = os.path.join(exp_path, data_filename)
            if os.stat(progress_csv_path).st_size == 0:
                progress_csv_path = os.path.join(exp_path, "log.txt")
            progress = load_progress(progress_csv_path)
            if os.path.exists(os.path.join(exp_path, data_filename.split(".")[0] + "_cont." + data_filename.split(".")[1])):
                prog_cont = load_progress(os.path.join(exp_path, data_filename.split(".")[0] + "_cont." + data_filename.split(".")[1]))
                for key in prog_cont:
                    if key in ['Number of env steps total', 'Total Train Time (s)', 'Number of train steps total', 'Epoch']:
                        prog_cont[key] += progress[key][-1]
                    progress[key] = np.concatenate([progress[key], prog_cont[key]])
            params = {}
            # if disable_variant:
            #     params = load_params(params_json_path)
            # else:
            #     try:
            #         params = load_params(variant_json_path)
            #     except IOError:
            #         params = load_params(params_json_path)
            exps_data.append(AttrDict(
                progress=progress,
                params=params,
                flat_params=flatten_dict(params)))
        except IOError as e:
            print(e)
    return exps_data


def flatten(l):
    return [item for sublist in l for item in sublist]


def reload_data(path):
    exps_data = load_exps_data(
        path
    )
    plottable_keys = list(
        set(flatten(list(exp.progress.keys()) for exp in exps_data)))
    plottable_keys = sorted([k for k in plottable_keys if k is not None])

    return exps_data, plottable_keys


def main(paths, names=None, title_='Cheetah 8-Task', top_limit=None, x_limits=None, plot_type=0, save_=True):
    subplots = paths.split("||")
    paths_list = [s.split(";") for s in subplots]
    names_list = [n.split(";") for n in [names] * len(paths_list)] if names is not None else None
    top_limit = None if top_limit is None or top_limit == "" else [float(el) for el in top_limit.split("||")]
    x_limits = None if x_limits is None or x_limits == "" else [float(el) for el in x_limits.split("||")]
    plot_type = int(plot_type)

    fig_folder = os.path.join('log', 'figures', f'{time.strftime("%Y-%m-%d-%H_%M_%S")}_pearl_our_comparison')
    if not os.path.isdir(fig_folder) and save_:
        os.mkdir(fig_folder)
        os.mkdir(os.path.join(fig_folder, 'png'))

    dd_list = []
    for paths, names in zip(paths_list, names_list) if names is not None else zip(paths_list, paths_list):
        data_dict = {}
        for path, name in zip(paths, names):
            data_dict[name] = reload_data(path)
        dd_list.append(data_dict)

    pl_list = []
    for data_dict in dd_list:
        plotting_data = {}
        for name, (values, keys) in data_dict.items():
            print(f'Structuring data from {name}')
            for (s, t) in [('AverageReturn_all_test_tasks', 'Average Reward Test'),
                           ('AverageReturn_all_train_tasks', 'Average Reward Training'),
                           ('test_eval_avg_reward_deterministic', 'Average Reward Test'),
                           ('train_eval_avg_reward_deterministic', 'Average Reward Training'),
                           ('Step_1-AverageReturn', 'Average Reward Test'),
                           ('train-AverageReturn', 'Average Reward Test')]:
                if s in keys:
                    for v in values:
                        if 'Number of env steps total' in keys:
                            steps = v['progress']['Number of env steps total']
                        elif 'n_env_steps_total' in keys:
                            steps = v['progress']['n_env_steps_total']
                        elif 'n_timesteps' in keys:
                            steps = v['progress']['n_timesteps']
                        else:
                            raise ValueError('No steps found for data')

                        if t in plotting_data.keys():
                            if name in plotting_data[t].keys():
                                plotting_data[t][name].append([steps, v['progress'][s]])
                            else:
                                plotting_data[t][name] = [[steps, v['progress'][s]]]
                        else:
                            plotting_data[t] = {name: [[steps, v['progress'][s]]]}
        pl_list.append(plotting_data)

    # Start plotting
    print(f'Plotting ...')

    plt.style.use('seaborn')

    # Use Latex text
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    if plot_type == 0:
        size_ = 64 #8 task and encoder ablation
    elif plot_type == 1:
        size_ = 26 # veldir
    elif plot_type == 2:
        size_ = 34 # veldir
    else:
        size_ = 60

    # plt.rc('font', size=size_)  # controls default text sizes
    plt.rc('figure', titlesize=size_)  # fontsize of the figure title
    plt.rc('axes', titlesize=size_)  # fontsize of the axes title
    plt.rc('axes', labelsize=size_)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size_*0.8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size_*0.8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=size_*0.8)  # legend fontsize

    for title, values in pl_list[0].items():
        plt.ioff()

        fig, axs = plt.subplots(int(np.ceil(len(pl_list) / 4)), min([len(pl_list), 4]))
        axs = axs if type(axs) is np.ndarray else [axs]

        for i, plotting_data in enumerate(pl_list):
            # TODO: Limit for display in paper
            min_steps_total = np.max([np.min([np.max(s[0]) for s in plotting_data[title][method_name]]) for method_name in plotting_data[title].keys()])

            max_mean, min_mean = -np.inf, np.inf
            for method_name in plotting_data[title].keys():
                steps = [s[0] for s in plotting_data[title][method_name]]
                steps = [s[:np.argmax(s)] for s in steps]
                min_len = np.argmin([len(s) for s in steps])

                d_list = []
                for j, a in enumerate(plotting_data[title][method_name]):
                    interpolation_function = InterFun(steps[j], a[1][:len(steps[j])])
                    d_list.append(interpolation_function(steps[min_len]))

                data_arr = np.array(d_list)

                mean = data_arr.mean(axis=0)
                std = np.sqrt(data_arr.var(axis=0))
                p = axs[i].plot(steps[min_len], mean, label=str(method_name), linewidth=size_ * 0.15)# , color='red'
                axs[i].hlines(mean.max(), 0., min_steps_total, linestyles='--', colors=p[0].get_color(), linewidth=size_ * 0.15)
                axs[i].fill_between(steps[min_len], mean + std, mean - std, alpha=0.3)

                max_mean = mean.max() if max_mean < mean.max() else max_mean
                min_mean = mean.min() if min_mean > mean.min() else min_mean

            if top_limit is not None and top_limit[i % len(top_limit)] != -1: axs[i].set_ylim(top=top_limit[i % len(top_limit)])
            if title_.split("||")[i] == 'Clustering Losses': axs[i].set_ylim(bottom=-220)
            axs[i].set_title(title_.split("||")[i])

            axs[i].set_xlabel('Training Transition $\it{n}$')
            if i == 0: axs[i].set_ylabel('Average Return $\it{R}$')
            axs[i].semilogx()

            if x_limits is not None and x_limits[i % len(x_limits)] > 0: axs[i].set_xlim(right=x_limits[i % len(x_limits)])

            if axs[i].get_xlim()[0] > 10**4: axs[i].set_xlim(left=10**4)



        # TODO: ncol for paper
        # fig.legend(handles=axs[-1].get_legend_handles_labels()[0], labels=[f'{el}' for el in pl_list[-1][title].keys()],
        #            bbox_to_anchor=(0.5, -0.08), loc='upper center', ncol=len(pl_list[-1][title].keys()) if len(pl_list) > 1 else 3)
        # 5.0 veldir, 12.0 8 task
        # fig.set_size_inches(max([8. * min([len(pl_list), 4]), len(pl_list[0][title].keys()) * 5.]), 5. * int(np.ceil(len(pl_list) / 4)))

        if plot_type == 0:
            # 8 task and encoder ablation
            plt.plot([], [], label='final', c='black', linestyle='--', linewidth=size_ * 0.15)
            leg = fig.legend(handles=axs[-1].get_legend_handles_labels()[0], labels=axs[-1].get_legend_handles_labels()[1],
                             bbox_to_anchor=(0.9, 0.5), loc='center left', ncol=1, handlelength=1)
            for line in leg.get_lines():
                line.set_linewidth(3.0)
            fig.set_size_inches(24., 15.)
        elif plot_type == 1:
            # ant 3 and single
            plt.plot([], [], label='final performance', c='black', linestyle='--', linewidth=size_ * 0.15)
            leg = fig.legend(handles=axs[-1].get_legend_handles_labels()[0], labels=axs[-1].get_legend_handles_labels()[1],
                       bbox_to_anchor=(0.5, -0.08), loc='upper center', ncol=len(pl_list[-1][title].keys()) + 1 if len(pl_list) > 1 else 3, handlelength=1.5)
            for line in leg.get_lines():
                line.set_linewidth(3.0)
            fig.set_size_inches(24., 4.5)
        elif plot_type == 2:
            # ant 3 and single
            plt.plot([], [], label='final performance', c='black', linestyle='--', linewidth=size_ * 0.15)
            leg = fig.legend(handles=axs[-1].get_legend_handles_labels()[0], labels=axs[-1].get_legend_handles_labels()[1],
                       bbox_to_anchor=(0.5, -0.08), loc='upper center', ncol=len(pl_list[-1][title].keys()) + 1 if len(pl_list) > 1 else 3, handlelength=1.5)
            for line in leg.get_lines():
                line.set_linewidth(3.0)
            fig.set_size_inches(32., 4.5)
        else:
            leg = fig.legend(handles=axs[-1].get_legend_handles_labels()[0], labels=axs[-1].get_legend_handles_labels()[1],
                       bbox_to_anchor=(0.5, -0.08), loc='upper center', ncol=len(pl_list[-1][title].keys()) if len(pl_list) > 1 else 3, handlelength=1)
            for line in leg.get_lines():
                line.set_linewidth(3.0)
            fig.set_size_inches(16., 12.)


        if save_:
            plt.savefig(os.path.join(fig_folder, title.replace(' ', '_') + '.pdf'), format='pdf', dpi=100,
                        bbox_inches='tight')
            plt.savefig(os.path.join(fig_folder, 'png', title.replace(' ', '_') + '.png'), format='png', dpi=100,
                        bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    pass


if __name__ == '__main__':
    main(*sys.argv[1:])