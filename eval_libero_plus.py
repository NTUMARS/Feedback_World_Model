"""Evaluate a LIBERO-10 single-task policy on the LIBERO-Plus perturbation protocol (Table 1, LIBERO rows).

The script iterates over the LIBERO-Plus variants of one original LIBERO-10 task
(``task=``), filtered by perturbation category (``categories=``; the paper's main
results use ``Robot Initial States``), and runs one episode per variant with the
selected method (``mode=base | guidance | fwm``).

Requires the LIBERO-Plus checkout (see README, "External simulators")::

    python eval_libero_plus.py --config-name=libero_plus_kitchen_s3 mode=fwm \
        policy_checkpoint=... world_model_checkpoint=...

Per-variant outcomes are written to ``<output_dir>/per_task_results.json`` and
the pooled success rate to ``<output_dir>/summary.json``.
"""
import json
import os
import pathlib
import random
import re
import sys
from collections import defaultdict, deque

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')
import egl_probe


def _patched_get_available_devices():
    return [int(os.environ.get('MUJOCO_EGL_DEVICE_ID', '0'))]


egl_probe.get_available_devices = _patched_get_available_devices

from diffusion_policy.env_runner.libero_paths import add_libero_plus_to_path, libero_plus_root

add_libero_plus_to_path()  # LIBERO-Plus must shadow the original LIBERO package

import dill
import hydra
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from libero.libero import benchmark

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.robomimic_image_sequential_runner import save_latent_log
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from fwm import FeedbackGuidedPlanner, PlannerConfig

_ROTATION_TRANSFORMER = RotationTransformer("axis_angle", "rotation_6d")

# LIBERO-Plus appends perturbation suffixes to the original task name.
_PERTURBATION_REGEX = re.compile(
    r'(_language_\d+_view_\d+_\d+_\d+_\d+_\d+_initstate_\d+'
    r'|_view_\d+_\d+_\d+_\d+_\d+_initstate_\d+_noise_\d+'
    r'|_view_\d+_\d+_\d+_\d+_\d+_initstate_\d+'
    r'|_light_\d+|_table_\d+|_tb_\d+|_add_\d+|_level\d+_sample\d+)$'
)


def original_task_name(task_name: str) -> str:
    match = _PERTURBATION_REGEX.search(task_name)
    return task_name[:match.start()] if match else task_name


def task_instruction(task_name: str) -> str:
    """Language instruction used for conditioning: the LIBERO-10 task name without its scene prefix."""
    base = original_task_name(task_name)
    offset = 8 if "SCENE10" in base else 7
    return " ".join(base[base.find("SCENE") + offset:].split("_"))


def load_task_classification(path: str, suite: str):
    with open(path, 'r') as f:
        items = json.load(f)[suite]
    return {item['name']: item['category'] for item in items}


# ----------------------------------------------------------------------- env
def create_env_from_task(task, task_suite, task_id, cfg):
    """Create a LIBERO-Plus environment for one task variant.

    ``_initstate_N`` variants select a robot class with a perturbed initial joint
    configuration.  ``set_init_state`` (needed for the object layout) would reset
    the arm to the nominal pose, so the perturbed joint state is saved and restored.
    """
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = task_suite.get_task_bddl_file_path(task_id)
    init_states = task_suite.get_task_init_states(task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=int(cfg.env_img_res),
        camera_widths=int(cfg.env_img_res),
    )
    env.seed(int(cfg.test_start_seed))
    env.reset()
    init_state = init_states[0] if len(init_states.shape) > 1 else init_states

    if re.search(r'_initstate_[1-9]\d*', task.name):
        n_robot_dof = sum(robot.dof for robot in env.env.robots)
        perturbed_qpos = env.env.sim.data.qpos[:n_robot_dof].copy()
        perturbed_qvel = env.env.sim.data.qvel[:n_robot_dof].copy()
        env.set_init_state(init_state)
        env.env.sim.data.qpos[:n_robot_dof] = perturbed_qpos
        env.env.sim.data.qvel[:n_robot_dof] = perturbed_qvel
        env.env.sim.forward()
        env.env._update_observables(force=True)
        obs = env.env._get_observations()
    else:
        obs = env.set_init_state(init_state)

    # absolute end-effector targets (the policies are trained with abs_action=True)
    for robot in env.env.robots:
        robot.controller.use_delta = False
    return env, obs


def preprocess_image(img):
    """robosuite renders upside down; flip to the orientation of the training data, HWC uint8 -> CHW float."""
    img = img[::-1].copy()
    return np.moveaxis(img, -1, 0).astype(np.float32) / 255.0


def extract_obs(raw_obs, policy_obs_keys):
    obs = {
        'agentview_rgb': preprocess_image(raw_obs['agentview_image']),
        'joint_states': raw_obs['robot0_joint_pos'].astype(np.float32),
        'ee_pos': raw_obs['robot0_eef_pos'].astype(np.float32),
        'ee_ori': raw_obs['robot0_eef_quat'].astype(np.float32),
    }
    if 'eye_in_hand_rgb' in policy_obs_keys and 'robot0_eye_in_hand_image' in raw_obs:
        obs['eye_in_hand_rgb'] = preprocess_image(raw_obs['robot0_eye_in_hand_image'])
    return obs


def stack_history(history, n_obs_steps):
    frames = list(history)
    while len(frames) < n_obs_steps:
        frames.insert(0, frames[0])
    frames = frames[-n_obs_steps:]
    return {k: np.stack([f[k] for f in frames], axis=0) for k in frames[0]}


def undo_transform_action(action):
    """10-d policy action (pos, rot6d, gripper) -> 7-d environment action (pos, axis-angle, gripper)."""
    pos, rot, gripper = action[..., :3], action[..., 3:9], action[..., [-1]]
    rot = _ROTATION_TRANSFORMER.inverse(rot)
    return np.concatenate([pos, rot, gripper], axis=-1)


def run_episode(policy, env, obs, instruction, cfg, device, policy_obs_keys, guided):
    max_steps = int(cfg.max_steps)
    n_action_steps = int(cfg.n_action_steps)
    n_obs_steps = int(cfg.n_obs_steps)

    policy.reset()
    history = deque(maxlen=n_obs_steps)
    history.append(extract_obs(obs, policy_obs_keys))
    rewards, done, t = [], False, 0
    while not done and t < max_steps:
        stacked = stack_history(history, n_obs_steps)
        obs_dict = dict_apply({k: v[np.newaxis] for k, v in stacked.items()},
                              lambda x: torch.from_numpy(x).to(device))
        if guided:
            action_dict = policy.predict_action_guided(obs_dict, language_goal=[instruction])
        else:
            action_dict = policy.predict_action(obs_dict, language_goal=[instruction])
        action = action_dict['action'].detach().cpu().numpy().squeeze(0)
        for a in undo_transform_action(action)[:n_action_steps]:
            obs, reward, _, _ = env.step(a.tolist())
            rewards.append(reward)
            history.append(extract_obs(obs, policy_obs_keys))
            t += 1
            if reward == 1.0:
                done = True
                break
    success = bool(max(rewards) >= 1.0) if rewards else False
    latent_log = policy.planner.pop_latent_log() if guided and policy.planner.records_latents else []
    return success, t, latent_log


# ------------------------------------------------------------------- loading
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_policy(checkpoint: str, device: torch.device, n_action_steps: int):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    OmegaConf.set_struct(cfg, False)
    cfg.n_action_steps = n_action_steps
    cfg.policy.n_action_steps = n_action_steps
    cfg.task.env_runner.n_action_steps = n_action_steps
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    normalizer_path = pathlib.Path(checkpoint).resolve().parent.parent / 'normalizer.pth'
    if normalizer_path.exists():
        policy.normalizer.load_state_dict(torch.load(normalizer_path, map_location='cpu'))
    policy.to(device)
    policy.eval()
    return policy, cfg


def build_planner(cfg, policy_cfg, mode, demo_path, device):
    p = cfg.planner
    planner_cfg = PlannerConfig(
        use_feedback=(mode == 'fwm'),
        feedback_gain=float(p.feedback_gain),
        use_action_aware=(mode == 'fwm') and bool(p.use_action_aware),
        action_aware_beta=float(p.action_aware_beta),
        controllability_num_obs=int(p.controllability_num_obs),
        controllability_num_actions=int(p.controllability_num_actions),
        n_action_steps=int(cfg.n_action_steps),
        control_freq=float(p.control_freq),
        energy_views=list(p.energy_views) if p.get('energy_views', None) else None,
        proprio_in_energy=bool(p.proprio_in_energy),
        proprio_weight=float(p.proprio_weight),
        seed=int(cfg.seed),
        record_latents=bool(cfg.get('record_latents', False)),
    )
    return FeedbackGuidedPlanner.from_checkpoint(
        world_model_ckpt=cfg.world_model_checkpoint,
        expert_dataset_cfg=policy_cfg.task.dataset,
        config=planner_cfg,
        expert_dataset_path=demo_path,
        n_obs_steps=int(cfg.n_obs_steps),
        device=device,
    )


@hydra.main(config_path="configs/eval", config_name="libero_plus_kitchen_s3", version_base=None)
def main(cfg: DictConfig):
    set_global_seed(int(cfg.seed))
    device = torch.device(cfg.device)
    mode = str(cfg.mode)
    assert mode in ('base', 'guidance', 'fwm'), f"unknown mode {mode}"
    output_dir = pathlib.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(output_dir / 'eval_config.yaml'))

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    classification_path = cfg.get('task_classification_path', None) or os.path.join(
        libero_plus_root(), 'libero', 'libero', 'benchmark', 'task_classification.json')
    name_to_category = load_task_classification(classification_path, cfg.task_suite_name)
    categories = list(cfg.categories) if cfg.get('categories', None) else None
    target_task = str(cfg.task)

    task_ids = [
        i for i in range(task_suite.n_tasks)
        if original_task_name(task_suite.get_task(i).name) == target_task
        and (categories is None or name_to_category.get(task_suite.get_task(i).name) in categories)
    ]
    print(f"[{mode}] {len(task_ids)} LIBERO-Plus variants of {target_task} "
          f"(categories={categories or 'all'})")
    if not task_ids:
        raise RuntimeError("no matching LIBERO-Plus tasks; check `task` and `categories`")

    policy, policy_cfg = load_policy(cfg.policy_checkpoint, device, int(cfg.n_action_steps))
    policy_obs_keys = set(policy.normalizer.params_dict.keys())

    if mode != 'base':
        demo_path = cfg.get('demo_dataset_path', None) or os.path.join(
            'data', 'libero_10_single_task', target_task, f'{target_task}_demo.hdf5')
        planner = build_planner(cfg, policy_cfg, mode, demo_path, device)
        policy.set_guidance(
            planner, scale=float(cfg.guidance.scale), steps=int(cfg.guidance.steps),
            threshold=float(cfg.guidance.threshold), max_grad_norm=float(cfg.guidance.max_grad_norm))

    per_category = defaultdict(lambda: {'total': 0, 'success': 0})
    per_task = []
    for task_id in tqdm.tqdm(task_ids, desc=f"{mode}:{target_task[:20]}"):
        task = task_suite.get_task(task_id)
        category = name_to_category.get(task.name, 'unknown')
        instruction = task_instruction(task.name)
        env, obs = create_env_from_task(task, task_suite, task_id, cfg)
        try:
            success, steps, latent_log = run_episode(policy, env, obs, instruction, cfg, device,
                                                     policy_obs_keys, guided=(mode != 'base'))
        finally:
            env.close()
        save_latent_log(latent_log, output_dir / 'latents' / f'task_{task_id:04d}.npz',
                        success=success, task_id=task_id)
        per_category[category]['total'] += 1
        per_category[category]['success'] += int(success)
        per_task.append({'task_id': task_id, 'task_name': task.name, 'category': category,
                         'instruction': instruction, 'success': success, 'steps': steps})
        rate = per_category[category]['success'] / per_category[category]['total']
        tqdm.tqdm.write(f"[{mode}][{category}] task {task_id}: {'success' if success else 'fail'} "
                        f"after {steps} steps | running success rate {rate:.3f}")
        with open(output_dir / 'per_task_results.json', 'w') as f:
            json.dump(per_task, f, indent=2)

    total = sum(v['total'] for v in per_category.values())
    successes = sum(v['success'] for v in per_category.values())
    summary = {
        'mode': mode, 'task': target_task,
        'per_category': {k: dict(v, success_rate=v['success'] / max(v['total'], 1)) for k, v in per_category.items()},
        'overall': {'total': total, 'success': successes, 'success_rate': successes / max(total, 1)},
        'policy_checkpoint': str(cfg.policy_checkpoint),
        'world_model_checkpoint': str(cfg.get('world_model_checkpoint', None)),
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[{mode}] {target_task}: {successes}/{total} = {successes / max(total, 1):.3f}  ->  {output_dir / 'summary.json'}")


if __name__ == '__main__':
    main()
