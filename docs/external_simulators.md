# External simulators

The code base does not vendor any simulator. Robomimic / robosuite come with the conda
environment; LIBERO and LIBERO-Plus are plain git checkouts.

## Robomimic / robosuite

* robosuite 1.4.1, robomimic 0.3.0, MuJoCo 2.3.x (`pip` wheels, installed by `conda_environment.yaml`).
* Rendering is off-screen through EGL. `MUJOCO_EGL_DEVICE_ID` selects the GPU used for rendering
  (default `0`); set it to the index of the GPU you run on when using `CUDA_VISIBLE_DEVICES`.
* `diffusion_policy/env/robomimic/xml_files/` contains the task MJCF files used when image observations
  are rendered through robomimic's own pipeline (`raw_obs_images=false`). The evaluation in this
  repository uses `raw_obs_images=true` (raw robosuite renders, matching the training data) and does not
  need them.

## LIBERO (original)

Needed for (a) training-time rollouts of LIBERO policies (checkpoint selection) and (b)
`scripts/convert_libero_to_abs_actions.py`.

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
pip install -r LIBERO/requirements.txt    # most packages are already in the conda environment
export LIBERO_ROOT=$PWD/LIBERO             # optional; ./LIBERO is the default location
```

`diffusion_policy/env_runner/libero_paths.py` adds the checkout to `sys.path` when needed; LIBERO's own
`~/.libero/config.yaml` must point at the checkout (it is written on first import). LIBERO demonstration
files store BDDL paths of the original data-collection layout; `diffusion_policy/env_runner/libero_bddl_mapping.py`
maps them to the files of the checkout.

## LIBERO-Plus

Needed for `eval_libero_plus.py` (the robot-initial-state and other perturbation protocols).

```bash
git clone https://github.com/sylvestf/LIBERO-plus.git LIBERO-plus
cd LIBERO-plus && git apply ../third_party/libero_plus_local.patch && cd ..
export LIBERO_PLUS_ROOT=$PWD/LIBERO-plus   # optional; ./LIBERO-plus is the default location
```

The patch (against commit `4976dc3`) only touches `libero/libero/benchmark/__init__.py`: the perturbed
initial states are loaded with `torch.load(..., weights_only=False)` (required by recent PyTorch versions), and
every perturbation variant of a task receives the instruction of the original LIBERO task.

LIBERO and LIBERO-Plus both read their path configuration from `$LIBERO_CONFIG_PATH/config.yaml` (default
`~/.libero`). `eval_libero_plus.py` points LIBERO-Plus to `~/.libero_plus` unless `LIBERO_CONFIG_PATH` is set,
so the two configurations do not overwrite each other. The file is written on first import; answer `N` to the
prompt to use the paths of the checkout.

`eval_libero_plus.py` inserts the LIBERO-Plus checkout at the front of `sys.path` so that its `libero`
package shadows the original one; do **not** put the original LIBERO on `PYTHONPATH` in the same shell.
Conversely, the training-time LIBERO rollouts and the data-conversion script put the original LIBERO
checkout first, so an editable `pip install -e LIBERO-plus` in the same environment is harmless.
The instruction given to the policy is the LIBERO-10 task name without the scene prefix (for example
`turn on the stove and put the moka pot on it`), identical for all methods.

## Verifying the setup

```bash
python -c "import robomimic, robosuite; print('robomimic ok')"
LIBERO_ROOT=$PWD/LIBERO python -c "from diffusion_policy.env_runner.libero_bddl_mapping import bddl_file_name_dict as d; print(len(d), 'bddl files')"
LIBERO_PLUS_ROOT=$PWD/LIBERO-plus python -c "from diffusion_policy.env_runner.libero_paths import add_libero_plus_to_path; add_libero_plus_to_path(); from libero.libero import benchmark; print(benchmark.get_benchmark_dict()['libero_10']().n_tasks, 'LIBERO-Plus libero_10 variants')"
```
