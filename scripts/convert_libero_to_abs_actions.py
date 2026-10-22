"""Convert LIBERO demonstration files from delta actions to absolute end-effector actions.

The recorded delta actions are replayed through the OSC controller of the
LIBERO (robosuite) environment to recover absolute pose targets; the original
delta actions are kept under ``delta_actions``.  Requires the original LIBERO
repository (see README, "External simulators").

Usage::

    python scripts/convert_libero_to_abs_actions.py \
        --input_dir  <LIBERO datasets>/libero_10 \
        --output_dir data/libero_10_abs/libero_10
"""
import argparse
import glob
import os
import shutil
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')
os.environ['PYOPENGL_PLATFORM'] = 'egl'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffusion_policy.env_runner.libero_paths import add_libero_to_path  # noqa: E402

add_libero_to_path()

import robomimic.utils.env_utils as EnvUtils  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402
from robomimic.config import config_factory  # noqa: E402
from diffusion_policy.env_runner.libero_bddl_mapping import resolve_bddl_file  # noqa: E402


def convert_file(input_path: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy(input_path, output_path)

    ObsUtils.initialize_obs_utils_with_config(config_factory(algo_name='bc'))
    env_meta = FileUtils.get_env_metadata_from_dataset(input_path)
    env_meta["bddl_file"] = resolve_bddl_file(env_meta["bddl_file"])
    env_meta["env_kwargs"]["bddl_file_name"] = env_meta["bddl_file"]
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)

    with h5py.File(input_path, 'r') as f:
        n_demos = len([k for k in f['data'].keys() if k.startswith('demo_')])
    print(f"{os.path.basename(input_path)}: {n_demos} demos")

    abs_actions_all = {}
    for demo_idx in tqdm(range(n_demos), desc="converting"):
        with h5py.File(input_path, 'r') as f:
            states = f[f'data/demo_{demo_idx}/states'][:]
            actions = f[f'data/demo_{demo_idx}/actions'][:]
        goal_pos = np.zeros((len(states), 3), dtype=np.float32)
        goal_ori = np.zeros((len(states), 3), dtype=np.float32)
        for i in range(len(states)):
            env.reset_to({'states': states[i]})
            robot = env.env.robots[0]
            robot.control(actions[i], policy_step=True)
            goal_pos[i] = robot.controller.goal_pos
            goal_ori[i] = Rotation.from_matrix(robot.controller.goal_ori).as_rotvec()
        abs_actions_all[demo_idx] = np.concatenate(
            [goal_pos, goal_ori, actions[:, [-1]].astype(np.float32)], axis=-1)

    with h5py.File(output_path, 'r+') as f:
        for demo_idx, abs_actions in abs_actions_all.items():
            demo = f[f'data/demo_{demo_idx}']
            delta = demo['actions'][:]
            del demo['actions']
            demo.create_dataset('actions', data=abs_actions)
            if 'delta_actions' not in demo:
                demo.create_dataset('delta_actions', data=delta)
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_dir', required=True, help='directory with LIBERO *_demo.hdf5 files')
    parser.add_argument('--output_dir', required=True, help='output directory')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, '*_demo.hdf5')))
    if not files:
        raise SystemExit(f"no *_demo.hdf5 files in {args.input_dir}")
    for path in files:
        convert_file(path, os.path.join(args.output_dir, os.path.basename(path)))
    for pattern in ('*.zarr.zip', '*.zarr.zip.lock'):
        for cache in glob.glob(os.path.join(args.output_dir, pattern)):
            os.remove(cache)


if __name__ == '__main__':
    main()
