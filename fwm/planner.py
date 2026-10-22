"""Feedback-guided planner: world-model guidance for diffusion-policy sampling.

This module implements the online part of Algorithm 1 in the paper.  Given a
frozen latent world model ``f_theta`` (``dyn_model/``) and the expert latent
memory ``Z^E``, the planner provides three operations that the diffusion policy
calls while sampling an action chunk:

``current_cost(obs)``
    squared distance of the current observation latent to its nearest expert
    latent; the policy uses it as an (optional) trigger for guidance.
``guidance_energy(traj, obs)``
    the differentiable guidance energy ``E_ctrl(A)`` (Eq. 16) of a batch of
    candidate action trajectories, computed from the feedback-corrected
    prediction ``z_fb_{t+1}(A)`` (Eq. 10) and the nearest expert latent (Eq. 5).
``commit_action(obs, action)``
    propagates the feedback state with the executed action chunk (Eq. 11).

Robomimic and LIBERO share one implementation; the only differences are
configuration (which camera views enter the energy, whether the
proprioceptive latent is included, and its weight).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from dyn_model.datasets.img_transforms import get_eval_crop_transform_resnet
from dyn_model.load_model import load_world_model
from fwm.action_aware import (
    estimate_controllability,
    interpolate_weights,
    normalize_weights,
    summarize_weights,
    weighted_squared_error,
)
from fwm.feedback import FeedbackState


@dataclass
class PlannerConfig:
    """Hyper-parameters of the feedback-guided planner.

    Attributes:
        use_feedback: enable the feedback world model (Sec. 4.2).  ``False``
            gives the standard open-loop world-model guidance (``Std. Guid.``).
        feedback_gain: scalar gain ``l`` of Eq. 10 (``L = l I``).  The per-update
            correction is ``dt * l * e_t``.
        use_action_aware: enable the action-aware energy (Sec. 4.3).
        action_aware_beta: interpolation coefficient ``beta`` of Eq. 15.
        controllability_num_obs: ``M`` demonstration states used in Eq. 14.
        controllability_num_actions: ``K`` counterfactual actions per state.
        n_action_steps: number of actions executed per policy call ``T_a``.
        control_freq: environment control frequency in Hz; ``dt = n_action_steps / control_freq``.
        energy_views: camera views whose latents enter the energy and the feedback
            state (``None`` = all views of the world model).
        proprio_in_energy: include the proprioceptive latent in the energy space.
        proprio_weight: scale of the proprioceptive latent inside the energy space
            (visual latents are scaled by 1).
        seed: seed of the counterfactual action sampling in Eq. 14.
        record_latents: keep, for every executed action chunk, the observed latent, the
            open-loop prediction, the feedback-corrected prediction and the observed next
            latent (used by ``scripts/latent_prediction_mse.py`` to validate the world model).
    """

    use_feedback: bool = True
    feedback_gain: float = 0.5
    use_action_aware: bool = True
    action_aware_beta: float = 0.2
    controllability_num_obs: int = 200
    controllability_num_actions: int = 32
    n_action_steps: int = 8
    control_freq: float = 20.0
    energy_views: Optional[List[str]] = None
    proprio_in_energy: bool = False
    proprio_weight: float = 1.0
    seed: int = 0
    record_latents: bool = False

    @property
    def dt(self) -> float:
        return float(self.n_action_steps) / float(self.control_freq)


class FeedbackGuidedPlanner:
    """World-model guidance with online feedback correction (Algorithm 1)."""

    def __init__(
        self,
        world_model: torch.nn.Module,
        world_model_cfg: DictConfig,
        world_model_normalizer: LinearNormalizer,
        expert_dataset: torch.utils.data.Dataset,
        config: PlannerConfig,
        n_obs_steps: int = 2,
        device: Optional[torch.device] = None,
    ):
        self.cfg = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- frozen world model ------------------------------------------------
        self.wm = world_model.to(self.device)
        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad_(False)
        self.wm_cfg = world_model_cfg
        self.wm_normalizer = world_model_normalizer.to(self.device)
        self.policy_action_normalizer: Optional[LinearNormalizer] = None

        self.view_names: List[str] = list(world_model_cfg.view_names)
        self.frameskip = int(world_model_cfg.frameskip)
        self.original_img_size = int(world_model_cfg.original_img_size)
        self.cropped_img_size = int(world_model_cfg.cropped_img_size)
        self.img_transform = (
            get_eval_crop_transform_resnet(self.original_img_size, self.cropped_img_size)
            if bool(world_model_cfg.use_crop)
            else (lambda x: x)
        )
        shape_obs = world_model_cfg.env.shape_obs
        self.proprio_keys: List[str] = [
            k for k, meta in shape_obs.items()
            if meta.get("type", "low_dim") != "rgb" and k != "language"
        ]
        self.uses_language = getattr(self.wm, "language_encoder", None) is not None
        self.n_obs_steps = int(n_obs_steps)

        # --- energy latent space ----------------------------------------------
        emb_dim = int(self.wm.encoder.emb_dim)
        energy_views = list(config.energy_views) if config.energy_views else self.view_names
        unknown = [v for v in energy_views if v not in self.view_names]
        if unknown:
            raise ValueError(f"energy_views {unknown} are not world-model views {self.view_names}")
        visual_index = []
        for v in energy_views:
            start = self.view_names.index(v) * emb_dim
            visual_index.extend(range(start, start + emb_dim))
        self._visual_index = torch.tensor(visual_index, dtype=torch.long, device=self.device)
        self.energy_views = energy_views

        # --- feedback state (Eq. 9-12) ----------------------------------------
        self.feedback = FeedbackState(gain=config.feedback_gain, dt=config.dt)

        # --- optional latent log for world-model validation ------------------
        self.latent_log: List[Dict[str, np.ndarray]] = []
        self._pending_prediction: Optional[Dict[str, np.ndarray]] = None

        # --- expert memory Z^E and action pool A_demo (offline, Algorithm 1 l.1)
        self.expert_dataset = expert_dataset
        self.expert_latents: torch.Tensor  # (N, D) on device
        self.expert_actions: torch.Tensor  # (N, Da) on CPU (raw, un-normalised)
        self._build_expert_memory()

        # --- action-aware weights (Eq. 14-15) ---------------------------------
        self.energy_weights: Optional[torch.Tensor] = None
        if config.use_action_aware:
            self._compute_action_aware_weights()

        print(
            f"[FWM planner] dt={config.dt:.3f}s (n_action_steps={config.n_action_steps}, "
            f"control_freq={config.control_freq}), feedback={'on' if config.use_feedback else 'off'} "
            f"(l={config.feedback_gain}), action-aware={'on' if config.use_action_aware else 'off'} "
            f"(beta={config.action_aware_beta}), energy views={self.energy_views}, "
            f"proprio_in_energy={config.proprio_in_energy} (x{config.proprio_weight}), "
            f"expert memory={tuple(self.expert_latents.shape)}"
        )

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_checkpoint(
        cls,
        world_model_ckpt: str,
        expert_dataset_cfg: DictConfig,
        config: PlannerConfig,
        expert_dataset_path: Optional[str] = None,
        n_obs_steps: int = 2,
        device: Optional[torch.device] = None,
    ) -> "FeedbackGuidedPlanner":
        """Build the planner from a world-model checkpoint and the policy's dataset config.

        The expert memory uses the demonstrations the *policy* was trained on:
        ``expert_dataset_cfg`` is the ``task.dataset`` entry stored in the policy
        checkpoint (including any training-subset selection such as
        ``max_train_episodes``).  Only the sequence parameters are overridden so
        that every demonstration frame becomes one memory entry.
        """
        device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_dir = Path(world_model_ckpt).resolve().parent.parent
        wm_cfg = OmegaConf.load(run_dir / "hydra.yaml")
        world_model = load_world_model(Path(world_model_ckpt), wm_cfg, device=device)
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(torch.load(run_dir / "normalizer.pth", map_location="cpu"))

        ds_cfg = copy.deepcopy(expert_dataset_cfg)
        OmegaConf.set_struct(ds_cfg, False)
        if expert_dataset_path is not None:
            ds_cfg.dataset_path = expert_dataset_path
        if bool(wm_cfg.abs_action) != bool(ds_cfg.get("abs_action", False)):
            raise ValueError("world model and policy dataset disagree on abs_action")
        ds_cfg.horizon = 1
        ds_cfg.n_obs_steps = 1
        ds_cfg.pad_before = 0
        ds_cfg.pad_after = 0
        if "val_ratio" in ds_cfg:
            ds_cfg.val_ratio = 0.0
        expert_dataset = hydra.utils.instantiate(ds_cfg)

        return cls(
            world_model=world_model,
            world_model_cfg=wm_cfg,
            world_model_normalizer=normalizer,
            expert_dataset=expert_dataset,
            config=config,
            n_obs_steps=n_obs_steps,
            device=device,
        )

    def set_policy_action_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Normalizer of the policy's action space (``policy.normalizer['action']``)."""
        self.policy_action_normalizer = normalizer

    def reset_episode(self) -> None:
        """Algorithm 1, line 2: ``z_bar_0 <- z_0`` happens lazily at the first commit."""
        self.feedback.reset()
        self.latent_log = []
        self._pending_prediction = None

    @property
    def records_latents(self) -> bool:
        return bool(self.cfg.record_latents)

    def pop_latent_log(self) -> List[Dict[str, np.ndarray]]:
        """Return the per-step latent records of the current episode and clear them.

        Each record holds ``z_obs`` (observed latent ``z_t``), ``z_pred_base`` (open-loop
        prediction ``f_theta(z_t, A_t^(0))``), ``z_pred_feedback`` (feedback-corrected
        prediction, Eq. 10) and ``z_next`` (observed latent ``z_{t+1}``).  The prediction of
        the last executed chunk has no observed successor and is not included.
        """
        log, self.latent_log, self._pending_prediction = self.latent_log, [], None
        return log

    # ------------------------------------------------------- observation prep
    def _frames_to_wm_obs(self, frames: Dict[str, torch.Tensor], language: Optional[torch.Tensor]) -> Dict:
        """Convert single-frame observations ``{key: (B, ...)}`` to world-model inputs."""
        batch = next(iter(frames.values())).shape[0]
        visual = {}
        for view in self.view_names:
            img = frames[view].to(self.device).float()
            if tuple(img.shape[-2:]) != (self.original_img_size, self.original_img_size):
                raise ValueError(
                    f"view '{view}' has size {tuple(img.shape[-2:])}, but the world model expects "
                    f"{self.original_img_size}x{self.original_img_size} inputs"
                )
            img = self.wm_normalizer[view].normalize(img)
            img = self.img_transform(img.reshape(-1, 3, self.original_img_size, self.original_img_size))
            visual[view] = img.reshape(batch, 1, 3, self.cropped_img_size, self.cropped_img_size)
        proprio = torch.cat([frames[k].to(self.device).float().reshape(batch, -1) for k in self.proprio_keys], dim=-1)
        proprio = self.wm_normalizer["state"].normalize(proprio).reshape(batch, 1, -1)
        obs_wm = {"visual": visual, "proprio": proprio}
        if self.uses_language and language is not None:
            lang = language.to(self.device).float()
            if lang.dim() == 1:
                lang = lang.reshape(1, 1, -1)
            elif lang.dim() == 2:
                lang = lang.reshape(lang.shape[0], 1, -1)
            if lang.shape[0] != batch:
                lang = lang[:1].expand(batch, -1, -1)
            obs_wm["language"] = lang
        return obs_wm

    @staticmethod
    def _last_frame(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """``(B, To, ...) -> (B, ...)`` using the most recent frame of the policy input."""
        return dict_apply(obs_dict, lambda x: x[:, -1] if x.dim() >= 3 else x)

    def _expand_obs(self, obs_wm: Dict, batch: int) -> Dict:
        out = {
            "visual": {k: v.expand(batch, -1, -1, -1, -1) for k, v in obs_wm["visual"].items()},
            "proprio": obs_wm["proprio"].expand(batch, -1, -1),
        }
        if "language" in obs_wm:
            out["language"] = obs_wm["language"].expand(batch, -1, -1)
        return out

    # ----------------------------------------------------------- latent space
    def _energy_latent(self, z_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Project world-model latents ``{'visual', 'proprio'}`` to the energy space ``(B, D)``."""
        z_visual = z_obs["visual"].reshape(z_obs["visual"].shape[0], -1)
        z_visual = z_visual.index_select(-1, self._visual_index)
        if not self.cfg.proprio_in_energy:
            return z_visual
        z_proprio = z_obs["proprio"].reshape(z_obs["proprio"].shape[0], -1) * self.cfg.proprio_weight
        return torch.cat([z_visual, z_proprio], dim=-1)

    def encode(self, obs_wm: Dict) -> torch.Tensor:
        """Observed latent ``z_t = psi(O_t)`` in the energy space, shape ``(B, D)``."""
        return self._energy_latent(self.wm.encode_obs(obs_wm))

    def predict(self, obs_wm: Dict, actions_wm: torch.Tensor) -> torch.Tensor:
        """Open-loop prediction ``f_theta(z_t, A)`` in the energy space.

        Args:
            obs_wm: world-model observation batch of size ``B``.
            actions_wm: world-model-normalised action chunk ``(B, frameskip, Da)``.
        """
        if self.uses_language and "language" not in obs_wm:
            raise ValueError("this world model is language-conditioned; pass language latents")
        act = rearrange(actions_wm, "b f a -> b 1 (f a)")
        z = self.wm.encode(obs_wm, act)
        z_next = self.wm.predict(z)[:, -1:]
        z_obs_pred, _ = self.wm.separate_emb(z_next)
        return self._energy_latent(z_obs_pred)

    def _actions_to_wm(self, actions_raw: torch.Tensor) -> torch.Tensor:
        """Raw (un-normalised) actions ``(B, T, Da)`` -> one world-model chunk ``(B, frameskip, Da)``."""
        n = actions_raw.shape[1]
        if n < self.frameskip:
            pad = actions_raw[:, -1:].expand(-1, self.frameskip - n, -1)
            actions_raw = torch.cat([actions_raw, pad], dim=1)
        actions_raw = actions_raw[:, : self.frameskip]
        return self.wm_normalizer["act"].normalize(actions_raw)

    def nearest_expert(self, z: torch.Tensor, chunk_size: int = 2048):
        """Nearest expert latent (Eq. 5). Returns ``(squared_distance (B,), index (B,))``."""
        best_cost = None
        best_idx = None
        for start in range(0, self.expert_latents.shape[0], chunk_size):
            chunk = self.expert_latents[start : start + chunk_size]
            dist = torch.cdist(z, chunk, p=2)
            cost, idx = dist.min(dim=-1)
            if best_cost is None:
                best_cost, best_idx = cost, idx + start
            else:
                better = cost < best_cost
                best_cost = torch.where(better, cost, best_cost)
                best_idx = torch.where(better, idx + start, best_idx)
        return best_cost ** 2, best_idx

    # --------------------------------------------------------- online methods
    @torch.no_grad()
    def current_cost(self, obs_dict: Dict[str, torch.Tensor], language: Optional[torch.Tensor] = None) -> float:
        """Squared distance of the current latent to the expert memory (guidance trigger)."""
        obs_wm = self._frames_to_wm_obs(self._last_frame(obs_dict), language)
        z_t = self.encode(obs_wm)
        cost, _ = self.nearest_expert(z_t)
        return float(cost[0].item())

    def guidance_energy(
        self,
        trajectory: torch.Tensor,
        obs_dict: Dict[str, torch.Tensor],
        language: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Guidance energy ``E_ctrl(A)`` (Eq. 16) averaged over the batch.

        Args:
            trajectory: predicted clean action trajectories ``(B, T, Da)`` in the
                policy's normalised action space (``x_0`` estimate of the sampler).
            obs_dict: policy observation ``{key: (B, To, ...)}``.
            language: language latents ``(B, 32)`` for language-conditioned models.
        """
        if self.policy_action_normalizer is None:
            raise RuntimeError("call set_policy_action_normalizer() before guidance")
        start = self.n_obs_steps - 1  # first action that will actually be executed
        chunk = trajectory[:, start : start + self.frameskip]
        actions_raw = self.policy_action_normalizer.unnormalize(chunk)
        actions_wm = self._actions_to_wm(actions_raw)
        batch = trajectory.shape[0]

        obs_wm = self._frames_to_wm_obs(self._last_frame(obs_dict), language)
        if obs_wm["proprio"].shape[0] != 1:
            raise ValueError("guided sampling supports one environment at a time (observation batch size 1)")
        with torch.no_grad():
            z_t = self.encode(obs_wm)[0]                               # z_t = psi(O_t)

        z_pred = self.predict(self._expand_obs(obs_wm, batch), actions_wm)  # Eq. 2 / 8, with gradients
        if self.cfg.use_feedback:
            z_pred = self.feedback.correct(z_pred, z_t)                # Eq. 10
        _, idx = self.nearest_expert(z_pred.detach())                  # Eq. 5
        diff = z_pred - self.expert_latents[idx]
        energy = weighted_squared_error(diff, self.energy_weights)     # Eq. 16 (Eq. 6 if uniform)
        return energy.mean()

    @torch.no_grad()
    def commit_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        action: torch.Tensor,
        language: Optional[torch.Tensor] = None,
    ) -> None:
        """Propagate the feedback state with the executed action chunk (Eq. 11).

        Args:
            obs_dict: the observation the action was computed from ``{key: (B, To, ...)}``.
            action: executed (un-normalised) actions ``(B, T_a, Da)``.
        """
        if not (self.cfg.use_feedback or self.cfg.record_latents):
            return
        obs_wm = self._frames_to_wm_obs(self._last_frame(obs_dict), language)
        if obs_wm["proprio"].shape[0] != 1:
            raise ValueError("feedback supports one environment at a time (observation batch size 1)")
        z_t = self.encode(obs_wm)[0]
        actions_wm = self._actions_to_wm(action.to(self.device).float()[:1])
        z_pred = self.predict(obs_wm, actions_wm)[0]

        if self.cfg.record_latents:
            # the observation of this step is the ground truth for the previous prediction
            if self._pending_prediction is not None:
                self.latent_log.append({**self._pending_prediction, "z_next": z_t.cpu().numpy()})
            z_fb = self.feedback.correct(z_pred.unsqueeze(0), z_t)[0] if self.cfg.use_feedback else z_pred
            self._pending_prediction = {
                "z_obs": z_t.cpu().numpy(),
                "z_pred_base": z_pred.cpu().numpy(),
                "z_pred_feedback": z_fb.cpu().numpy(),
            }
        if self.cfg.use_feedback:
            self.feedback.propagate(z_t, z_pred)

    # ------------------------------------------------------------- offline
    def _sample_frames(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Dataset sample ``{'obs': {key: (T=1, ...)}}`` -> single-frame batch of size 1."""
        return {k: torch.as_tensor(v)[-1:].float() for k, v in sample["obs"].items() if k != "language"}

    def _sample_language(self, sample: Dict) -> Optional[torch.Tensor]:
        """Encode the tokenised instruction stored in a LIBERO dataset sample."""
        if not self.uses_language:
            return None
        tokens = torch.as_tensor(sample["obs"]["language"])[-1]  # (2, L): input_ids, attention_mask
        text_tokens = {
            "input_ids": tokens[0].long().reshape(1, -1).to(self.device),
            "attention_mask": tokens[1].long().reshape(1, -1).to(self.device),
        }
        with torch.no_grad():
            return self.wm.language_encoder(text_tokens).reshape(1, 1, -1)

    @torch.no_grad()
    def _build_expert_memory(self) -> None:
        loader = DataLoader(self.expert_dataset, batch_size=64, shuffle=False, num_workers=4)
        latents, actions = [], []
        for batch in loader:
            frames = {k: v[:, -1] for k, v in batch["obs"].items() if k != "language"}
            # encode_obs() does not need the instruction; only predict() does.
            latents.append(self.encode(self._frames_to_wm_obs(frames, language=None)).cpu())
            actions.append(batch["action"][:, 0].float().cpu())
        self.expert_latents = torch.cat(latents, dim=0).to(self.device)
        self.expert_actions = torch.cat(actions, dim=0)
        print(f"[FWM planner] expert memory: {self.expert_latents.shape[0]} frames, dim {self.expert_latents.shape[1]}")

    @torch.no_grad()
    def _compute_action_aware_weights(self) -> None:
        generator = torch.Generator().manual_seed(int(self.cfg.seed))
        K = int(self.cfg.controllability_num_actions)

        def predict_fn(obs_idx: int, actions_raw: torch.Tensor) -> torch.Tensor:
            sample = self.expert_dataset[int(obs_idx)]
            obs_wm = self._frames_to_wm_obs(self._sample_frames(sample), self._sample_language(sample))
            obs_k = self._expand_obs(obs_wm, K)
            actions = actions_raw.to(self.device).float().unsqueeze(1).expand(-1, self.frameskip, -1)
            return self.predict(obs_k, self.wm_normalizer["act"].normalize(actions))

        raw = estimate_controllability(
            predict_fn,
            num_observations_total=len(self.expert_dataset),
            action_pool=self.expert_actions,
            num_observations=self.cfg.controllability_num_obs,
            num_actions=K,
            generator=generator,
        )
        normalized = normalize_weights(raw)                                       # Eq. 15
        self.energy_weights = interpolate_weights(normalized, self.cfg.action_aware_beta).to(self.device)
        print(f"[FWM planner] action-aware weights: {summarize_weights(normalized)}")
