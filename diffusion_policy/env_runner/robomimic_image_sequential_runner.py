"""Sequential Robomimic evaluation runner with the robot-initial-state OOD protocol (Appendix B).

Episodes are run one at a time (guided sampling keeps a per-episode feedback
state).  When ``robot_perturbation_scale > 0`` every test episode starts from a
demonstration initial state whose arm joint angles are perturbed by Gaussian
noise and clipped to the joint limits (Eq. 45).
"""
import collections
import os
import pathlib
import random

import h5py
import numpy as np
import torch
import tqdm
import wandb

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import downsample_mask, get_val_mask
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

PANDA_JOINT_LIMITS = np.array([
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.0698],
    [-2.8973, 2.8973],
    [-0.0175, 3.7525],
    [-2.8973, 2.8973],
])


def perturb_robot_state(state, noise_scale, rng, joint_limits=PANDA_JOINT_LIMITS, n_robots=1):
    """Perturb the arm joint angles inside a flattened MuJoCo state (Eq. 45).

    Robomimic's flattened state layout is ``[time, qpos..., qvel...]`` with the
    arm joints of robot ``r`` at ``1 + r * 9 : 1 + r * 9 + 7`` (7 arm joints followed
    by 2 gripper joints).  Gripper, object and velocity entries are unchanged.
    """
    perturbed = state.copy()
    n_arm, n_gripper = 7, 2
    for robot_idx in range(n_robots):
        offset = 1 + robot_idx * (n_arm + n_gripper)
        perturbed[offset:offset + n_arm] += rng.normal(0, noise_scale, size=n_arm)
        for i in range(n_arm):
            perturbed[offset + i] = np.clip(perturbed[offset + i], joint_limits[i, 0], joint_limits[i, 1])
    return perturbed


def save_latent_log(log, path, **meta):
    """Store the per-step latent records of one episode (see ``FeedbackGuidedPlanner.pop_latent_log``)."""
    if len(log) == 0:
        return
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.stack([rec[key] for rec in log]) for key in log[0]}
    np.savez(path, **arrays, **{k: np.asarray(v) for k, v in meta.items()})


def create_env(env_meta, shape_meta, enable_render=True, render_obs_key='agentview_image',
               fps=10, crf=22, n_obs_steps=2, n_action_steps=8, max_steps=400, raw_obs_images=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=enable_render, use_image_obs=enable_render)
    env.env.hard_reset = False

    video_recorder = VideoRecorder.create_h264(
        fps=fps, codec='h264', input_pix_fmt='rgb24', crf=crf, thread_type='FRAME', thread_count=1)
    video_wrapper = VideoRecordingWrapper(
        env=RobomimicImageWrapper(
            env=env, shape_meta=shape_meta, init_state=None,
            render_obs_key=render_obs_key, raw_obs_images=raw_obs_images),
        video_recoder=video_recorder, file_path=None, steps_per_render=max(20 // fps, 1))
    return MultiStepWrapper(
        env=video_wrapper, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps, max_episode_steps=max_steps)


class SequentialRobomimicImageRunner(BaseImageRunner):
    """Evaluate a policy episode by episode on Robomimic with optional OOD initial states."""

    def __init__(self,
            output_dir,
            dataset_path,
            shape_meta: dict,
            n_train=0,
            n_train_vis=0,
            train_start_idx=0,
            n_test=50,
            n_test_vis=0,
            test_start_seed=100000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            raw_obs_images=True,
            robot_perturbation_scale=0.0,
            max_train_episodes=None,
            dataset_seed=42,
            val_ratio=0.0,
        ):
        super().__init__(output_dir)
        self.media_dir = pathlib.Path(output_dir).joinpath('media')
        self.media_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = os.path.expanduser(dataset_path)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        robots = env_meta.get('env_kwargs', {}).get('robots', ['Panda'])
        n_robots = len(robots) if isinstance(robots, list) else 1

        # Demonstration initial states that seed the OOD perturbation.  When the
        # policy was trained on a subset of the demonstrations, only that subset
        # is used so that no test episode starts from a held-out demonstration.
        self.env_configs = []
        demo_init_states = []
        with h5py.File(dataset_path, 'r') as f:
            all_demo_keys = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
            for i in range(n_train):
                train_idx = train_start_idx + i
                init_state = f[f'data/demo_{train_idx}/states'][0]
                demo_init_states.append(init_state)
                self.env_configs.append({
                    'prefix': 'train/', 'init_state': init_state, 'seed': None,
                    'enable_render': i < n_train_vis})

            if robot_perturbation_scale > 0 and len(demo_init_states) == 0:
                n_episodes = len(all_demo_keys)
                if max_train_episodes is not None and max_train_episodes < n_episodes:
                    t_mask = ~get_val_mask(n_episodes, val_ratio, seed=dataset_seed)
                    t_mask = downsample_mask(t_mask, max_n=max_train_episodes, seed=dataset_seed)
                    train_indices = np.nonzero(t_mask)[0]
                    for idx in train_indices:
                        demo_init_states.append(f[f'data/demo_{idx}/states'][0])
                    print(f"[OOD] using {len(train_indices)}/{n_episodes} training demonstrations as base states")
                else:
                    for key in all_demo_keys:
                        demo_init_states.append(f[f'data/{key}/states'][0])

        if robot_perturbation_scale > 0:
            print(f"[OOD] robot initial-state perturbation: sigma={robot_perturbation_scale:.3f} rad "
                  f"({np.degrees(robot_perturbation_scale):.1f} deg), n_robots={n_robots}, "
                  f"{len(demo_init_states)} base states")

        for i in range(n_test):
            seed = test_start_seed + i
            if robot_perturbation_scale > 0 and len(demo_init_states) > 0:
                demo_idx = i % len(demo_init_states)
                rng = np.random.RandomState(seed)
                init_state = perturb_robot_state(
                    demo_init_states[demo_idx], robot_perturbation_scale, rng, n_robots=n_robots)
            else:
                init_state = None
            self.env_configs.append({
                'prefix': 'test/', 'init_state': init_state,
                'seed': seed if init_state is None else None,
                'enable_render': i < n_test_vis})

        self.output_dir = output_dir
        self.dataset_path = dataset_path
        self.env_meta = env_meta
        self.shape_meta = shape_meta
        self.render_obs_key = render_obs_key
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.past_action = past_action
        self.abs_action = abs_action
        self.rotation_transformer = rotation_transformer
        self.tqdm_interval_sec = tqdm_interval_sec
        self.raw_obs_images = raw_obs_images
        self.test_start_seed = test_start_seed
        self.robot_perturbation_scale = robot_perturbation_scale

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        guided = getattr(policy, 'planner', None) is not None
        n_inits = len(self.env_configs)
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        success_list = []

        for i, env_cfg in enumerate(self.env_configs):
            prefix, init_state, seed = env_cfg['prefix'], env_cfg['init_state'], env_cfg['seed']
            episode_seed = seed if seed is not None else self.test_start_seed + i
            self._set_episode_seed(episode_seed)

            env = create_env(
                env_meta=self.env_meta, shape_meta=self.shape_meta, enable_render=True,
                render_obs_key=self.render_obs_key, fps=self.fps, crf=self.crf,
                n_obs_steps=self.n_obs_steps, n_action_steps=self.n_action_steps,
                max_steps=self.max_steps, raw_obs_images=self.raw_obs_images)
            self._initialize_env(env, init_state, seed, env_cfg['enable_render'], i)

            obs = env.reset()
            policy.reset()  # also resets the feedback state of an attached planner

            rewards = []
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {self.env_meta['env_name']} {i+1}/{n_inits}",
                             leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            success = False
            while not done:
                np_obs_dict = dict_apply(dict(obs), lambda x: np.expand_dims(x, axis=0))
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

                if guided:
                    action_dict = policy.predict_action_guided(obs_dict)
                else:
                    action_dict = policy.predict_action(obs_dict)
                action = action_dict['action'].detach().cpu().numpy().squeeze(0)
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action")

                env_action = self.undo_transform_action(action) if self.abs_action else action
                obs, reward, done, info = env.step(env_action)
                success = bool(env.env.get_success_label())
                done = np.all(done)
                rewards.append(reward)
                pbar.update(action.shape[0])
                if success:
                    break
            pbar.close()

            success_list.append(success)
            print(f"episode {i+1}/{n_inits}: success={success}  cumulative success rate={np.mean(success_list):.3f}")
            if guided and getattr(policy.planner, 'records_latents', False):
                save_latent_log(policy.planner.pop_latent_log(),
                                pathlib.Path(self.output_dir) / 'latents' / f'episode_{i:03d}.npz',
                                success=success, seed=episode_seed)
            video_path = env.render()
            if isinstance(video_path, list) and len(video_path) > 0:
                video_path = video_path[0]
            all_video_paths[i] = video_path
            all_rewards[i] = rewards
            env.close()
            del env

        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i, env_cfg in enumerate(self.env_configs):
            prefix = env_cfg['prefix']
            reward_array = np.array(all_rewards[i])
            max_reward = float(np.max(reward_array)) if len(reward_array) > 0 else 0.0
            max_rewards[prefix].append(max_reward)
            test_id = env_cfg['seed'] if env_cfg['seed'] is not None else self.test_start_seed + i
            log_data[prefix + f'sim_max_reward_{test_id}'] = max_reward
            log_data[prefix + f'success_{test_id}'] = bool(success_list[i])
            video_path = all_video_paths[i]
            if video_path is not None and os.path.exists(video_path):
                log_data[prefix + f'sim_video_{test_id}'] = wandb.Video(video_path)
        for prefix, value in max_rewards.items():
            log_data[prefix + 'mean_score'] = float(np.mean(value))
        return log_data

    @staticmethod
    def _set_episode_seed(seed: int):
        """Reset all RNGs before each episode so sampling noise is reproducible."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _initialize_env(self, env, init_state, seed, enable_render, i):
        assert isinstance(env.env, VideoRecordingWrapper)
        env.env.video_recoder.stop()
        env.env.file_path = None
        if enable_render:
            env.env.file_path = str(self.media_dir.joinpath(f"eval_video_{i}.mp4"))
        assert isinstance(env.env.env, RobomimicImageWrapper)
        if init_state is not None:
            env.env.env.init_state = init_state
        else:
            env.env.env.init_state = None
            if seed is not None:
                env.seed(seed)

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:  # dual arm
            action = action.reshape(-1, 2, 10)
        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3:3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)
        if raw_shape[-1] == 20:
            uaction = uaction.reshape(*raw_shape[:-1], 14)
        return uaction
