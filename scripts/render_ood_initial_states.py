"""Render the robot-initial-state OOD protocol used for Robomimic (Appendix B, Figs. 7-9).

For a few demonstrations the script shows the in-distribution start next to OOD starts
produced by ``perturb_robot_state`` (Eq. 45): the seven arm joint angles receive Gaussian
noise with standard deviation ``--scale`` (rad) and are clipped to the Panda joint limits;
gripper, object and velocity entries are unchanged.  State ``i`` is drawn with the random
seed ``--test-start-seed + i``, exactly as ``eval_robomimic.py`` does for test episode ``i``.

Usage::

    python scripts/render_ood_initial_states.py data/square/data/expert_demonstration/square_ph_demo.hdf5 \
        --scale 0.1 --n-states 3 --cameras agentview robot0_eye_in_hand --out ood_initial_states.png
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')
import egl_probe

egl_probe.get_available_devices = lambda: [int(os.environ['MUJOCO_EGL_DEVICE_ID'])]

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from diffusion_policy.env_runner.robomimic_image_sequential_runner import perturb_robot_state

N_ARM_JOINTS = 7
ROBOT_STATE_STRIDE = 9  # 7 arm joints + 2 gripper joints per robot in the flattened MuJoCo state


def load_initial_states(dataset_path, demo_indices):
    with h5py.File(dataset_path, 'r') as f:
        return [f[f'data/demo_{i}/states'][0] for i in demo_indices]


def create_env(dataset_path, cameras):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta['env_kwargs']['use_object_obs'] = False
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'rgb': [f'{c}_image' for c in cameras], 'low_dim': ['robot0_eef_pos']})
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True)
    env.env.hard_reset = False
    env.reset()
    robots = env_meta.get('env_kwargs', {}).get('robots', ['Panda'])
    return env, (len(robots) if isinstance(robots, list) else 1)


def render_state(env, state, cameras, size):
    env.reset_to({'states': state})
    return [env.render(mode='rgb_array', height=size, width=size, camera_name=c) for c in cameras]


def arm_joint_deltas_deg(perturbed, base, n_robots):
    deltas = []
    for r in range(n_robots):
        offset = 1 + r * ROBOT_STATE_STRIDE
        deltas.append(np.degrees(perturbed[offset:offset + N_ARM_JOINTS] - base[offset:offset + N_ARM_JOINTS]))
    return np.concatenate(deltas)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help='Robomimic demonstration hdf5 (provides the simulator metadata and initial states)')
    parser.add_argument('--scale', type=float, default=0.1, help='sigma of the joint perturbation in rad (Eq. 45)')
    parser.add_argument('--n-states', type=int, default=3, help='number of OOD starts to render')
    parser.add_argument('--demo-index', type=int, default=0, help='demonstration whose initial state is perturbed')
    parser.add_argument('--test-start-seed', type=int, default=100000, help='seed of the first state (as in the eval configs)')
    parser.add_argument('--cameras', nargs='+', default=['agentview'], help='robosuite camera names, one row each')
    parser.add_argument('--size', type=int, default=256, help='rendered image size in pixels')
    parser.add_argument('--out', default='ood_initial_states.png')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base = load_initial_states(args.dataset, [args.demo_index])[0]
    env, n_robots = create_env(args.dataset, args.cameras)
    states = [('In-distribution', base)]
    for i in range(args.n_states):
        rng = np.random.RandomState(args.test_start_seed + i)
        perturbed = perturb_robot_state(base, args.scale, rng, n_robots=n_robots)
        deltas = arm_joint_deltas_deg(perturbed, base, n_robots)
        print(f"OOD #{i + 1} (seed {args.test_start_seed + i}): joint deltas [deg] = {np.round(deltas, 1)}")
        states.append((f'OOD #{i + 1}  (max |Δq| = {np.abs(deltas).max():.1f}°)', perturbed))

    n_rows, n_cols = len(args.cameras), len(states)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
    for col, (title, state) in enumerate(states):
        images = render_state(env, state, args.cameras, args.size)
        for row, (camera, image) in enumerate(zip(args.cameras, images)):
            axes[row, col].imshow(image)
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(title, fontsize=10)
            if col == 0:
                axes[row, col].text(-0.04, 0.5, camera, transform=axes[row, col].transAxes,
                                    rotation=90, va='center', ha='right', fontsize=10)
    fig.suptitle(f'demo {args.demo_index}, sigma = {args.scale} rad ({np.degrees(args.scale):.1f} deg)', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    env.env.close()  # robosuite MujocoEnv
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
