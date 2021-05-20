# Masterthesis: Task Inference Based Meta-Reinforcement Learning for Robotics Environments

by Lukas Knak (TU Munich)

This is the reference implementation of the algorithm, we give the working title `Task Inference based meta-rl algorithm using Gaussian mixture models and gated Recurrent units (TIGR)`.
The code is currently under review and will be cleaned up.
We will also give more details on configuration options.
For now just run the starting example to get a general idea.
This repository is based on [rlkit](https://github.com/vitchyr/rlkit), [PEARL](https://github.com/katerakelly/oyster) and [CEMRL](https://github.com/LerchD/cemrl.git).

--------------------------------------

### Instructions

Deprecated, but currently still necessary instructions:
- Clone this repo with `git clone --recurse-submodules`. If not working install the `rand_params_envs` submodule [rand_param_envs](https://github.com/dennisl88/rand_param_envs) and `meta_rand_envs` submodule [meta_rand_envs](https://github.com/lknak/meta_rand_envs).

#### Installation.
- To install locally, you will need to first install [MuJoCo](https://www.roboti.us/index.html).
For the task distributions in which the reward function varies (Cheetah, Ant, Humanoid), install MuJoCo200.
(following is deprecated and will be removed) For the task distributions where different tasks correspond to different model parameters (Walker and Hopper), MuJoCo131 is required.
Simply install it the same way as MuJoCo200.
- Set `LD_LIBRARY_PATH` to point to both the MuJoCo binaries (`/$HOME/.mujoco/mujoco200/bin`) as well as the gpu drivers (something like `/usr/lib/nvidia-390`, you can find your version by running `nvidia-smi`).
- For the remaining dependencies, we recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html). Use
Use the latest `yml` file (tigr.yml) to set a conda virtual machine.
Make sure the correct GPU driver is installed and you use a matching version of CUDA toolkit and torch.
We use torch 1.7.0 with cuda version 11 for our evaluations (`pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html`)

This installation has been tested only on 64-bit Ubuntu 18.04.

##### Environments:
- We created versions of the standard Mujoco environments in the repository [meta_rand_envs](https://github.com/lknak/meta_rand_envs).
To perform experiments, both submodules [rand_param_envs](https://github.com/dennisl88/rand_param_envs) and [meta_rand_envs](https://github.com/lknak/meta_rand_envs) must be installed in dev mode on the created conda env.
For installation, perform the following steps for meta_rand_envs (rand_param_envs analogously):
```
cd [submodules/meta_rand_envs]
conda activate tigr
pip install -e .
```

- MetaWorld Experiments (currently under review): Clone the package [metaworld](https://github.com/rlworkgroup/metaworld) and install in dev mode on the created conda env (see above).
As their repo does not provide official releases and compatibility over versions and commits is not guaranteed, the wrapper build for this repository is tested and used with the metaworld commit from 7/23/2020 @ 12:11 AM.


### Things to notice
- Most relevant code for `tigr` is in the folder `./tigr`.
- We use `ray` for data collection parallelization.
Make sure to configure a suitable amount of `num_workers` to not crash the program
- Experiments are configured via `json` configuration files located in `./configs`. To reproduce an experiment, run:
`python runner.py ./configs/[EXP].json`
A working starting example is `python runner.py ./configs/cheetah-multi-task.json`.
Further environments are currently under review and might not work.

- The default config file that is overwritten by the individual experiment config files is `./configs/default.py`.
- Environment wrappers will be located in `rlkit/envs`.
- There is lots of unused code from PEARL and rlkit, just ignore it.
- By default the code will use the GPU - to use CPU instead, set `use_gpu=False` in the appropriate config file.

- Output files will be written to `./output/[ENV]/[EXP NAME]` where the experiment name is uniquely generated based on the date.
The file `progress.csv` contains statistics logged over the course of training.
We also provide online evaluation via tensorboard located in the `/tensorboard` folder. Use `tensorboard --logdir=./output/[ENV]/[EXP NAME]/tensorboard` for visualizations of learning curves and the current embeddings.

### Troubleshoot

Just contact me via slack.

