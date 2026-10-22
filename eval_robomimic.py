"""Evaluate a diffusion policy on Robomimic under robot-initial-state OOD (Table 1, Robomimic rows).

Three modes are supported (``mode=``):

* ``base``      -- the base diffusion policy, no guidance.
* ``guidance``  -- open-loop world-model guidance (``Std. Guid.``).
* ``fwm``       -- feedback world model guidance (``Ours``).

Example::

    python eval_robomimic.py --config-name=robomimic_square mode=fwm \
        policy_checkpoint=... world_model_checkpoint=...

Results are written to ``<output_dir>/eval_results.json`` (``test/mean_score`` is
the success rate).
"""
import json
import os
import pathlib
import random
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import egl_probe


def _patched_get_available_devices():
    return [int(os.environ.get('MUJOCO_EGL_DEVICE_ID', '0'))]


egl_probe.get_available_devices = _patched_get_available_devices

import dill
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from diffusion_policy.env_runner.robomimic_image_sequential_runner import SequentialRobomimicImageRunner
from fwm import FeedbackGuidedPlanner, PlannerConfig


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as e:  # pragma: no cover
        print(f"[seed] deterministic algorithms unavailable: {e}")


def load_policy(checkpoint: str, device: torch.device, n_action_steps: int):
    """Load a trained policy (EMA weights) and the config stored in its checkpoint."""
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


def build_planner(cfg: DictConfig, policy_cfg: DictConfig, mode: str, n_obs_steps: int, device: torch.device):
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
        expert_dataset_path=cfg.get('dataset_path', None),
        n_obs_steps=n_obs_steps,
        device=device,
    )


@hydra.main(config_path="configs/eval", config_name="robomimic_square", version_base=None)
def main(cfg: DictConfig):
    set_global_seed(int(cfg.seed))
    device = torch.device(cfg.device)
    mode = str(cfg.mode)
    assert mode in ('base', 'guidance', 'fwm'), f"unknown mode {mode}"

    output_dir = pathlib.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(output_dir / 'eval_config.yaml'))

    policy, policy_cfg = load_policy(cfg.policy_checkpoint, device, int(cfg.n_action_steps))

    if mode != 'base':
        planner = build_planner(cfg, policy_cfg, mode, n_obs_steps=int(policy_cfg.n_obs_steps), device=device)
        policy.set_guidance(
            planner,
            scale=float(cfg.guidance.scale),
            steps=int(cfg.guidance.steps),
            threshold=float(cfg.guidance.threshold),
            max_grad_norm=float(cfg.guidance.max_grad_norm),
        )

    # --- environment runner (robot-initial-state OOD protocol, Appendix B) ---
    runner_kwargs = OmegaConf.to_container(policy_cfg.task.env_runner, resolve=True)
    runner_kwargs.pop('_target_', None)
    runner_kwargs.pop('n_envs', None)
    if cfg.get('dataset_path', None):
        runner_kwargs['dataset_path'] = cfg.dataset_path
    ds_cfg = policy_cfg.task.dataset
    runner_kwargs.update(dict(
        output_dir=str(output_dir),
        n_train=0, n_train_vis=0,
        n_test=int(cfg.n_test), n_test_vis=int(cfg.n_test_vis),
        test_start_seed=int(cfg.test_start_seed),
        max_steps=int(cfg.max_steps),
        n_action_steps=int(cfg.n_action_steps),
        raw_obs_images=True,
        robot_perturbation_scale=float(cfg.robot_perturbation_scale),
        max_train_episodes=ds_cfg.get('max_train_episodes', None),
        dataset_seed=ds_cfg.get('seed', 42),
        val_ratio=ds_cfg.get('val_ratio', 0.0),
    ))
    runner = SequentialRobomimicImageRunner(**runner_kwargs)
    runner_log = runner.run(policy)

    results = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer, bool)) else str(v))
               for k, v in runner_log.items()}
    results['mode'] = mode
    results['policy_checkpoint'] = str(cfg.policy_checkpoint)
    results['world_model_checkpoint'] = str(cfg.get('world_model_checkpoint', None))
    with open(output_dir / 'eval_results.json', 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"[{mode}] success rate = {runner_log['test/mean_score']:.3f}  ->  {output_dir / 'eval_results.json'}")


if __name__ == '__main__':
    main()
