import sys, os, re, time, platform
import json, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
import matplotlib
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D


DATA_DIR = os.path.join('output', 'cheetah-multi-task', '2021_04_30_parametrized', '2021_04_30_cheetah_8_task_true_gmm', 'tensorboard')
FIG_DIR = os.path.join('log', 'latent')

PLOT_LIST = []

MARKERS = ['.', '^', 's', 'p', '*', 'X', 'h', 'd', '+', 'P']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def main(run_name=None, save=True, show_='last'):
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

			true_tasks = np.array([s[0].split('[')[0].strip() for s in metadata])
			unique_tasks = np.sort(np.unique(true_tasks))
			specs = np.array([float(re.findall('[-]*[0-9]+[.]*[0-9]*', s[0])[0]) for s in metadata])

			data = np.array(data, dtype=np.float)

			d_list, t_list, s_list = [], [], []
			for true_task in unique_tasks:

				# Bring data to 3D
				pcad = ''
				if data.shape[1] < 3:
					d = data[true_tasks == true_task]
					temp = np.zeros([d.shape[0], 3])
					temp[:, 0:d.shape[1]] = d
					d = temp

				elif data.shape[1] > 3:
					d = data[true_tasks == true_task]
					print(f'Performing PCA from {d.shape[1]} DIM to 3')
					pcad = f' (Using PCA From {d.shape[1]} To 3 Dimensions)'
					d = perform_pca(d)
				d_list.append(d)
				t_list.append(true_task)
				s_list.append(specs[true_tasks == true_task])

			epoch_dict[step_] = [metadata, data, pcad, d_list, t_list, s_list]

	m_l = max([len(k) for k in epoch_dict.keys()])

	# Plotting
	# Use Latex text
	matplotlib.rcParams['mathtext.fontset'] = 'stix'
	matplotlib.rcParams['font.family'] = 'STIXGeneral'

	plt.style.use('seaborn')

	size_ = 12# 20 veldir, 36 8 task
	plt.rc('font', size=size_)  # controls default text sizes
	plt.rc('axes', labelsize=size_)  # fontsize of the x and y labels
	plt.rc('axes', titlesize=size_)  # fontsize of the axes title
	plt.rc('xtick', labelsize=size_)  # fontsize of the tick labels
	plt.rc('ytick', labelsize=size_)  # fontsize of the tick labels
	plt.rc('legend', fontsize=size_)  # legend fontsize
	plt.rc('figure', titlesize=size_)  # fontsize of the figure title

	CMAP_NAME = 'viridis'

	for step_ in sorted(epoch_dict.keys()):

		metadata, values, pcad, d_list, t_list, s_list = epoch_dict[step_]

		fig = plt.figure()

		for i, d in enumerate(d_list):
			axs = fig.add_subplot(int(np.ceil(len(d_list) / 4)), min([len(d_list), 4]), i + 1, projection='3d')

			axs.set_aspect('auto')
			axs.set_title(f'{" ".join([el.capitalize() for el in t_list[i].split("_")])}', y=1.08)

			el = axs.scatter(d[:, 0], d[:, 1], d[:, 2],
							 c=(s_list[i] - s_list[i].min()) / (s_list[i].max() - s_list[i].min()),
							 cmap=plt.get_cmap(CMAP_NAME))

			axs.tick_params(labelsize=0)
			# axs.set_xlabel('Latent Dim 1')
			# axs.set_ylabel('Latent Dim 2')
			# axs.set_zlabel('Latent Dim 3')

		cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=plt.get_cmap(CMAP_NAME)), ax=fig.axes,
					 orientation='vertical',
					 anchor=(1., 0.5),
					 shrink=0.5,
					 ticks=[0, 0.5, 1])
		cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])

		fig.set_size_inches(3. * min([len(d_list), 4]), 2.8 * int(np.ceil(len(d_list) / 4)))

		if save:
			plt.savefig(os.path.join(fig_folder, f'encodings_step_{str(step_).zfill(m_l)}.png'), format='png', dpi=100, bbox_inches='tight')
		else:
			plt.show()

		plt.close()

		print(f'Created plot for {DATA_DIR}')


def perform_pca(values, dims=3):

	# sample points equally for all gaussians
	x = np.copy(values)

	# centering the data
	x -= np.mean(x, axis=0)

	cov = np.cov(x, rowvar=False)

	evals, evecs = LA.eigh(cov)

	idx = np.argsort(evals)[::-1]
	evecs = evecs[:, idx[:dims]]

	return np.dot(x, evecs)


if __name__ == '__main__':
	main(*sys.argv[1:])
