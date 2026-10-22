"""Image-conditioned diffusion policy (CNN/UNet variant) with world-model guidance.

The policy follows Diffusion Policy (Chi et al.).  Two sampling entry points exist:

``predict_action``
    plain DDPM sampling (the base policy).
``predict_action_guided``
    guided sampling (Algorithm 1 of the paper): during the final
    ``guidance_steps`` low-noise denoising steps the score is corrected with the
    gradient of the planner's guidance energy, and after sampling the executed
    action chunk is committed to the planner so the feedback state advances.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.common.language_models import extract_text_features, get_text_model
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import diffusion_policy.model.vision.crop_randomizer as dmvc
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_core as rmbn

LANGUAGE_EMB_DIM = 32


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # parameters passed to scheduler.step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {'low_dim': [], 'rgb': [], 'depth': [], 'scan': []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            if key == 'language':
                continue
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn', hdf5_type='image', task_name='square', dataset_type='ph')
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name, config=config, obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim, device='cpu')
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features))
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape, crop_height=x.crop_height, crop_width=x.crop_width,
                    num_crops=x.num_crops, pos_enc=x.pos_enc))

        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
            if 'language' in shape_meta['obs']:
                global_cond_dim += LANGUAGE_EMB_DIM

        model = ConditionalUnet1D(
            input_dim=input_dim, local_cond_dim=None, global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim, down_dims=down_dims,
            kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale)

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim, obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps, fix_obs_steps=True, action_visible=False)
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # language-conditioned policies (LIBERO) train a small text encoder jointly
        self.text_model, self.tokenizer, self.max_length = None, None, None
        if 'language' in shape_meta['obs']:
            self.text_model, self.tokenizer, self.max_length = get_text_model('libero_10', 'clip')

        # world-model guidance (set with set_guidance)
        self.planner = None
        self.guidance_scale = 0.0
        self.guidance_steps = 0
        self.guidance_threshold = 0.0
        self.max_grad_norm = 0.0

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    # ============================ guidance setup ============================
    def set_guidance(self, planner, scale: float, steps: int, threshold: float = 0.0, max_grad_norm: float = 0.0):
        """Attach a guidance planner (``fwm.FeedbackGuidedPlanner``).

        Args:
            planner: provides ``guidance_energy``, ``current_cost``, ``commit_action`` and ``reset_episode``.
            scale: guidance strength ``lambda`` of Eq. 7.
            steps: ``tau_g``, guidance is applied at diffusion steps ``t < steps`` (the
                last ``steps`` low-noise denoising steps of a ``num_inference_steps`` schedule).
            threshold: guidance is applied only when the current observation's squared
                distance to the expert memory exceeds this value (0 = always).
            max_grad_norm: clip the norm of the guidance gradient (0 = no clipping).
        """
        self.planner = planner
        self.guidance_scale = float(scale)
        self.guidance_steps = int(steps)
        self.guidance_threshold = float(threshold)
        self.max_grad_norm = float(max_grad_norm)
        self.planner.set_policy_action_normalizer(self.normalizer['action'])

    def clear_guidance(self):
        self.planner = None

    def reset(self):
        """Called by the environment runners at the start of every episode."""
        if self.planner is not None:
            self.planner.reset_episode()

    # ============================ conditioning ============================
    def _encode_language(self, language_goal) -> Optional[torch.Tensor]:
        if language_goal is None or self.text_model is None:
            return None
        tokens = self.tokenizer(
            language_goal, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt").to(self.device)
        return extract_text_features(self.text_model, tokens, language_emb_model='clip')

    def _build_conditioning(self, obs_dict: Dict[str, torch.Tensor], text_latents: Optional[torch.Tensor]):
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        T, Da, Do, To = self.horizon, self.action_dim, self.obs_feature_dim, self.n_obs_steps
        device, dtype = self.device, self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            if text_latents is not None:
                global_cond = torch.cat([global_cond, text_latents], dim=-1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs).reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True
        return cond_data, cond_mask, local_cond, global_cond

    def _split_action(self, nsample: torch.Tensor) -> Dict[str, torch.Tensor]:
        naction_pred = nsample[..., :self.action_dim]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {'action': action_pred[:, start:end], 'action_pred': action_pred}

    # ============================ sampling ============================
    def conditional_sample(self, condition_data, condition_mask, local_cond=None, global_cond=None,
                           generator=None, **kwargs):
        model = self.model
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape, dtype=condition_data.dtype,
            device=condition_data.device, generator=generator)
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def guided_conditional_sample(self, condition_data, condition_mask, obs_dict, text_latents=None,
                                  local_cond=None, global_cond=None, generator=None, **kwargs):
        """DDPM sampling with world-model guidance in the final ``guidance_steps`` steps (Algorithm 1, l.5-13)."""
        model = self.model
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape, dtype=condition_data.dtype,
            device=condition_data.device, generator=generator)
        scheduler.set_timesteps(self.num_inference_steps)

        # optional trigger: distance of the current observation to the expert memory
        guidance_enabled = True
        if self.guidance_threshold > 0:
            guidance_enabled = self.planner.current_cost(obs_dict, text_latents) > self.guidance_threshold

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            trajectory = trajectory.detach().requires_grad_(True)
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

            if guidance_enabled and t < self.guidance_steps:
                # clean-sample estimate A^(0) from the current noisy trajectory
                x0 = scheduler.step(model_output, t, trajectory).pred_original_sample
                energy = self.planner.guidance_energy(x0, obs_dict, text_latents)
                grad = torch.autograd.grad(energy, trajectory)[0]
                if self.max_grad_norm > 0:
                    grad_norm = grad.norm()
                    if grad_norm > self.max_grad_norm:
                        grad = grad * (self.max_grad_norm / grad_norm)
                # eps-prediction <=> score: s = -eps / sqrt(1 - alpha_bar_t), so
                # s - lambda * grad E corresponds to eps + lambda * sqrt(1 - alpha_bar_t) * grad E (Eq. 7)
                grad_scale = self.guidance_scale * (1 - scheduler.alphas_cumprod[t]).sqrt()
                model_output = model_output.detach() + grad_scale * grad

            trajectory = scheduler.step(
                model_output, t, trajectory.detach(), generator=generator, **kwargs).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory.detach()

    # ============================ inference ============================
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], language_goal=None) -> Dict[str, torch.Tensor]:
        """Base diffusion policy. ``obs_dict`` values have shape ``(B, To, ...)``."""
        assert 'past_action' not in obs_dict  # not implemented yet
        with torch.no_grad():
            text_latents = self._encode_language(language_goal)
            cond_data, cond_mask, local_cond, global_cond = self._build_conditioning(obs_dict, text_latents)
            nsample = self.conditional_sample(
                cond_data, cond_mask, local_cond=local_cond, global_cond=global_cond, **self.kwargs)
            return self._split_action(nsample)

    def predict_action_guided(self, obs_dict: Dict[str, torch.Tensor], language_goal=None) -> Dict[str, torch.Tensor]:
        """Feedback-guided diffusion policy (requires ``set_guidance``)."""
        assert 'past_action' not in obs_dict  # not implemented yet
        if self.planner is None:
            raise RuntimeError("call set_guidance() before predict_action_guided()")
        with torch.enable_grad():
            with torch.no_grad():
                text_latents = self._encode_language(language_goal)
                cond_data, cond_mask, local_cond, global_cond = self._build_conditioning(obs_dict, text_latents)
            nsample = self.guided_conditional_sample(
                cond_data, cond_mask, obs_dict=obs_dict, text_latents=text_latents,
                local_cond=local_cond, global_cond=global_cond, **self.kwargs)
        result = self._split_action(nsample)
        # Algorithm 1, line 16: advance the feedback state with the executed chunk
        self.planner.commit_action(obs_dict, result['action'], text_latents)
        return result

    # ============================ training ============================
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        text_latents = None
        if 'language' in batch['obs']:
            language_goal = batch['obs']['language']
            del batch['obs']['language']
            text_tokens = {
                "input_ids": language_goal[:, 0].long()[:, 0],
                "attention_mask": language_goal[:, 0].long()[:, 1],
            }
            text_latents = extract_text_features(self.text_model, text_tokens, language_emb_model='clip')

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
            if text_latents is not None:
                global_cond = torch.cat([global_cond, text_latents], dim=-1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=trajectory.device).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        return loss.mean()
