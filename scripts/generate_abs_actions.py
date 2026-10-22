"""Add absolute end-effector actions (``abs_actions``) to a Robomimic HDF5 demonstration file.

Both the diffusion policy and the world model use absolute-pose action targets.
For every demonstration step the recorded delta action is replayed through the
robosuite OSC controller to recover the absolute goal pose.  Works for single-arm
tasks (Square, Tool-Hang) and the dual-arm Transport task.

Usage::

    python scripts/generate_abs_actions.py data/square/data/expert_demonstration/square_ph_demo.hdf5
"""
import argparse
import json
import os
import re

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')

import robosuite


def fix_xml_paths(xml_string: str) -> str:
    local_assets = os.path.join(os.path.dirname(robosuite.__file__), "models", "assets")
    return re.sub(r'file="[^"]*robosuite[^"]*/models/assets/', f'file="{local_assets}/', xml_string)


def generate_abs_actions(dataset_path: str) -> None:
    f = h5py.File(dataset_path, 'r')
    env_args = json.loads(f['data'].attrs['env_args'])
    env_kwargs = dict(env_args.get('env_kwargs', {}))
    env_kwargs.update(dict(has_renderer=False, has_offscreen_renderer=False, ignore_done=True,
                           use_object_obs=False, use_camera_obs=False))
    env = robosuite.make(env_args['env_name'], **env_kwargs)
    n_robots = len(env.robots)
    action_per_robot = 7  # pos(3) + axis-angle(3) + gripper(1)

    demos = sorted([k for k in f['data'].keys() if k.startswith('demo_')], key=lambda x: int(x[5:]))
    print(f"{dataset_path}: {env_args['env_name']}, {len(demos)} demos, {n_robots} robot(s)")

    all_abs_actions = {}
    for demo_key in tqdm(demos, desc="converting"):
        demo = f[f'data/{demo_key}']
        states = demo['states'][:]
        actions = demo['actions'][:]
        env.reset()
        env.reset_from_xml_string(fix_xml_paths(demo.attrs['model_file']))
        env.sim.reset()

        abs_actions = np.zeros_like(actions)
        for i in range(len(states)):
            env.sim.set_state_from_flattened(states[i])
            env.sim.forward()
            stacked = actions[i].reshape(n_robots, action_per_robot)
            parts = []
            for idx, robot in enumerate(env.robots):
                robot.control(stacked[idx], policy_step=True)
                controller = robot.controller
                goal_pos = controller.goal_pos.copy()
                goal_ori = Rotation.from_matrix(controller.goal_ori).as_rotvec()
                parts.extend([goal_pos, goal_ori, stacked[idx, -1:]])
            abs_actions[i] = np.concatenate(parts)
        all_abs_actions[demo_key] = abs_actions.astype(np.float32)
    f.close()
    env.close()

    with h5py.File(dataset_path, 'r+') as f_out:
        for demo_key, abs_actions in all_abs_actions.items():
            demo = f_out[f'data/{demo_key}']
            if 'abs_actions' in demo:
                del demo['abs_actions']
            demo.create_dataset('abs_actions', data=abs_actions)

    for cache in (dataset_path + '.zarr.zip', dataset_path + '.zarr.zip.lock'):
        if os.path.exists(cache):
            os.remove(cache)
            print(f"removed stale cache {cache}")
    print(f"wrote abs_actions to {dataset_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', nargs='+', help='Robomimic HDF5 file(s) to convert in place')
    for path in parser.parse_args().dataset:
        generate_abs_actions(path)
