"""End-to-end smoke test of guided sampling on CPU with a tiny policy and a synthetic world model.

Exercises ``FeedbackGuidedPlanner`` (expert memory, controllability weights,
feedback-corrected energy, feedback propagation) through
``DiffusionUnetHybridImagePolicy.predict_action_guided`` without any simulator
or trained checkpoint.  Requires the robomimic package (installed with the
conda environment).
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

pytest.importorskip("robomimic")

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: E402

from diffusion_policy.common.normalize_util import get_image_range_normalizer  # noqa: E402
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer  # noqa: E402
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy  # noqa: E402
from dyn_model.models.proprio import ProprioceptiveEmbedding  # noqa: E402
from dyn_model.models.visual_dyn_model import VisualDynamicsModel  # noqa: E402
from dyn_model.models.vit import ViTPredictor  # noqa: E402
from fwm import FeedbackGuidedPlanner, PlannerConfig  # noqa: E402

VIEWS = ["cam_a", "cam_b"]
IMG, CROP = 84, 76
PROPRIO_DIM, ACTION_DIM, FRAMESKIP = 3, 4, 4
EMB, PROPRIO_EMB, ACTION_EMB = 8, 4, 6


class FakeEncoder(nn.Module):
    """Stands in for the policy's ResNet backbones: one 8-d embedding per view."""

    def __init__(self):
        super().__init__()
        self.emb_dim = EMB
        self.proj = nn.Linear(3, EMB)

    def forward(self, x):
        out = {}
        for view in VIEWS:
            img = x[view]                                  # (B, T, 3, h, w)
            pooled = img.mean(dim=(-2, -1))                # (B, T, 3)
            out[view] = self.proj(pooled).unsqueeze(2)     # (B, T, 1, EMB)
        return out


class FakeExpertDataset(torch.utils.data.Dataset):
    def __init__(self, n=40, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.images = {v: torch.rand(n, 1, 3, IMG, IMG, generator=g) for v in VIEWS}
        self.proprio = torch.randn(n, 1, PROPRIO_DIM, generator=g)
        self.actions = torch.randn(n, 1, ACTION_DIM, generator=g)

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, i):
        obs = {v: self.images[v][i] for v in VIEWS}
        obs["pos"] = self.proprio[i]
        return {"obs": obs, "action": self.actions[i]}


def build_world_model():
    torch.manual_seed(0)
    predictor = ViTPredictor(
        num_patches=1, num_frames=1, dim=EMB * len(VIEWS) + PROPRIO_EMB + ACTION_EMB,
        depth=1, heads=1, mlp_dim=16, pool="mean", dim_head=8,
        visual_dim=EMB * len(VIEWS), proprio_dim=PROPRIO_EMB, action_dim=ACTION_EMB)
    wm = VisualDynamicsModel(
        image_size=CROP, num_hist=1, num_pred=1, encoder=FakeEncoder(),
        proprio_encoder=ProprioceptiveEmbedding(num_frames=1, tubelet_size=1, in_chans=PROPRIO_DIM, emb_dim=PROPRIO_EMB),
        action_encoder=ProprioceptiveEmbedding(num_frames=1, tubelet_size=1, in_chans=FRAMESKIP * ACTION_DIM, emb_dim=ACTION_EMB),
        predictor=predictor, proprio_dim=PROPRIO_EMB, action_dim=ACTION_EMB,
        train_encoder=False, train_predictor=True, view_names=VIEWS, use_layernorm=True)
    normalizer = LinearNormalizer()
    for v in VIEWS:
        normalizer[v] = get_image_range_normalizer()
    normalizer["state"] = SingleFieldLinearNormalizer.create_fit(np.random.randn(100, PROPRIO_DIM))
    normalizer["act"] = SingleFieldLinearNormalizer.create_fit(np.random.randn(100, ACTION_DIM))
    cfg = OmegaConf.create({
        "view_names": VIEWS, "frameskip": FRAMESKIP, "use_crop": True, "abs_action": True,
        "original_img_size": IMG, "cropped_img_size": CROP,
        "env": {"shape_obs": {"pos": {"shape": [PROPRIO_DIM]},
                              **{v: {"shape": [3, IMG, IMG], "type": "rgb"} for v in VIEWS}}},
    })
    return wm, cfg, normalizer


def build_policy():
    torch.manual_seed(0)
    shape_meta = {
        "obs": {**{v: {"shape": [3, IMG, IMG], "type": "rgb"} for v in VIEWS}, "pos": {"shape": [PROPRIO_DIM]}},
        "action": {"shape": [ACTION_DIM]},
    }
    scheduler = DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2",
                              clip_sample=True, prediction_type="epsilon", variance_type="fixed_small")
    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=shape_meta, noise_scheduler=scheduler, horizon=8, n_action_steps=FRAMESKIP, n_obs_steps=2,
        num_inference_steps=10, crop_shape=(CROP, CROP), diffusion_step_embed_dim=16, down_dims=(16, 32),
        kernel_size=3, n_groups=4, obs_encoder_group_norm=True, eval_fixed_crop=True)
    normalizer = LinearNormalizer()
    for v in VIEWS:
        normalizer[v] = get_image_range_normalizer()
    normalizer["pos"] = SingleFieldLinearNormalizer.create_fit(np.random.randn(100, PROPRIO_DIM))
    normalizer["action"] = SingleFieldLinearNormalizer.create_fit(np.random.randn(100, ACTION_DIM))
    policy.set_normalizer(normalizer)
    policy.eval()
    return policy


def make_obs(seed):
    g = torch.Generator().manual_seed(seed)
    obs = {v: torch.rand(1, 2, 3, IMG, IMG, generator=g) for v in VIEWS}
    obs["pos"] = torch.randn(1, 2, PROPRIO_DIM, generator=g)
    return obs


@pytest.fixture(scope="module")
def planner_and_policy():
    wm, cfg, normalizer = build_world_model()
    planner = FeedbackGuidedPlanner(
        world_model=wm, world_model_cfg=cfg, world_model_normalizer=normalizer,
        expert_dataset=FakeExpertDataset(),
        config=PlannerConfig(use_feedback=True, feedback_gain=0.5, use_action_aware=True, action_aware_beta=0.5,
                             controllability_num_obs=5, controllability_num_actions=4,
                             n_action_steps=FRAMESKIP, control_freq=20.0, energy_views=["cam_a"],
                             proprio_in_energy=True, proprio_weight=2.0, seed=0),
        n_obs_steps=2, device=torch.device("cpu"))
    policy = build_policy()
    policy.set_guidance(planner, scale=0.1, steps=5, threshold=0.0, max_grad_norm=1.0)
    return planner, policy


def test_expert_memory_and_weights(planner_and_policy):
    planner, _ = planner_and_policy
    assert planner.expert_latents.shape == (40, EMB + PROPRIO_EMB)   # one view + proprio latent
    assert planner.expert_actions.shape == (40, ACTION_DIM)
    assert planner.energy_weights.shape == (EMB + PROPRIO_EMB,)
    assert torch.isfinite(planner.energy_weights).all()
    assert planner.feedback.dt == pytest.approx(FRAMESKIP / 20.0)


def test_guided_sampling_runs_and_feedback_advances(planner_and_policy):
    planner, policy = planner_and_policy
    policy.reset()
    assert not planner.feedback.initialized

    out = policy.predict_action_guided(make_obs(1))
    assert out["action"].shape == (1, FRAMESKIP, ACTION_DIM)
    assert torch.isfinite(out["action"]).all()
    assert planner.feedback.initialized and planner.feedback.num_updates == 1

    obs2 = make_obs(2)
    residual_before = planner.feedback.residual(planner.encode(planner._frames_to_wm_obs(
        planner._last_frame(obs2), None))[0])
    assert residual_before.norm() > 0            # the belief differs from the new observation
    out2 = policy.predict_action_guided(obs2)
    assert torch.isfinite(out2["action"]).all()
    assert planner.feedback.num_updates == 2

    policy.reset()
    assert not planner.feedback.initialized


def test_energy_gradient_flows_to_actions(planner_and_policy):
    planner, policy = planner_and_policy
    policy.reset()
    traj = torch.randn(3, 8, ACTION_DIM, requires_grad=True)
    energy = planner.guidance_energy(traj, make_obs(3))
    grad = torch.autograd.grad(energy, traj)[0]
    assert energy.ndim == 0 and torch.isfinite(energy)
    start = policy.n_obs_steps - 1
    assert grad[:, start:start + FRAMESKIP].abs().sum() > 0          # executed chunk receives gradient
    assert grad[:, :start].abs().sum() == 0 and grad[:, start + FRAMESKIP:].abs().sum() == 0


def test_disabled_trigger_reproduces_base_sampling(planner_and_policy):
    planner, policy = planner_and_policy
    policy.set_guidance(planner, scale=0.1, steps=5, threshold=1e9, max_grad_norm=1.0)  # never fires
    obs = make_obs(4)
    policy.reset()
    torch.manual_seed(123)
    base = policy.predict_action(obs)["action"]
    policy.reset()
    torch.manual_seed(123)
    guided = policy.predict_action_guided(obs)["action"]
    assert torch.allclose(base, guided, atol=1e-6)
    policy.set_guidance(planner, scale=0.1, steps=5, threshold=0.0, max_grad_norm=1.0)


def test_open_loop_guidance_mode():
    wm, cfg, normalizer = build_world_model()
    planner = FeedbackGuidedPlanner(
        world_model=wm, world_model_cfg=cfg, world_model_normalizer=normalizer, expert_dataset=FakeExpertDataset(),
        config=PlannerConfig(use_feedback=False, use_action_aware=False, n_action_steps=FRAMESKIP, control_freq=20.0),
        n_obs_steps=2, device=torch.device("cpu"))
    policy = build_policy()
    policy.set_guidance(planner, scale=0.1, steps=5)
    policy.reset()
    out = policy.predict_action_guided(make_obs(5))
    assert torch.isfinite(out["action"]).all()
    assert planner.energy_weights is None
    assert not planner.feedback.initialized          # feedback disabled: nothing is committed


def test_latent_recording_and_mse_script(tmp_path):
    import importlib.util
    import pathlib
    from diffusion_policy.env_runner.robomimic_image_sequential_runner import save_latent_log

    wm, cfg, normalizer = build_world_model()
    planner = FeedbackGuidedPlanner(
        world_model=wm, world_model_cfg=cfg, world_model_normalizer=normalizer, expert_dataset=FakeExpertDataset(),
        config=PlannerConfig(use_feedback=True, feedback_gain=0.5, use_action_aware=False,
                             n_action_steps=FRAMESKIP, control_freq=20.0, record_latents=True),
        n_obs_steps=2, device=torch.device("cpu"))
    policy = build_policy()
    policy.set_guidance(planner, scale=0.1, steps=5)
    policy.reset()
    for i in range(4):
        policy.predict_action_guided(make_obs(10 + i))
    log = planner.pop_latent_log()
    assert len(log) == 3                                   # the last prediction has no observed successor
    assert set(log[0]) == {"z_obs", "z_pred_base", "z_pred_feedback", "z_next"}
    assert np.allclose(log[0]["z_pred_base"], log[0]["z_pred_feedback"])   # e_0 = 0 at the first chunk
    assert not np.allclose(log[1]["z_pred_base"], log[1]["z_pred_feedback"])
    for k in range(2):
        assert np.allclose(log[k]["z_next"], log[k + 1]["z_obs"])         # ground truth = next observation
    assert planner.latent_log == [] and planner._pending_prediction is None

    save_latent_log(log, tmp_path / "latents" / "episode_000.npz", success=True, seed=0)
    save_latent_log(log, tmp_path / "latents" / "episode_001.npz", success=False, seed=1)
    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "latent_prediction_mse.py"
    spec = importlib.util.spec_from_file_location("latent_prediction_mse", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    episodes = mod.load_episodes(str(tmp_path))
    assert len(episodes) == 2 and episodes[0]["z_next"].shape == (3, EMB * len(VIEWS))  # both views, no proprio
    reduction = mod.summarize(episodes, per_episode=True)
    assert np.isfinite(reduction)
    mod.plot_episode(episodes[0], str(tmp_path / "trajectory.png"))
    assert (tmp_path / "trajectory.png").exists()
