"""CPU unit tests for the feedback state and the action-aware energy (no simulator required)."""
import math

import pytest
import torch

from fwm.action_aware import (
    estimate_controllability,
    interpolate_weights,
    normalize_weights,
    weighted_squared_error,
)
from fwm.feedback import FeedbackState


def test_feedback_initialisation_and_first_update_has_no_residual():
    fb = FeedbackState(gain=0.5, dt=0.75)
    z0 = torch.randn(8)
    z_pred = torch.randn(8)
    assert not fb.initialized
    assert torch.allclose(fb.residual(z0), torch.zeros(8))
    fb.propagate(z0, z_pred)                      # z_bar_0 = z_0, e_0 = 0
    assert fb.initialized
    assert torch.allclose(fb.z_bar, z_pred)      # z_bar_1 = z_0 + dt * v = f(z_0, A_0)


def test_feedback_correction_matches_eq10_and_propagation_matches_eq11():
    gain, dt = 0.4, 0.5
    fb = FeedbackState(gain=gain, dt=dt)
    z0 = torch.zeros(4)
    fb.propagate(z0, torch.ones(4))               # z_bar_1 = 1
    z1 = torch.full((4,), 1.5)                    # observed latent differs from belief
    e1 = z1 - fb.z_bar
    assert torch.allclose(e1, torch.full((4,), 0.5))

    z_pred = torch.randn(3, 4)                    # three candidate action sequences
    corrected = fb.correct(z_pred, z1)            # Eq. 10
    assert torch.allclose(corrected, z_pred + dt * gain * e1)

    z_pred_exec = torch.full((4,), 2.0)
    fb.propagate(z1, z_pred_exec)                 # Eq. 11
    expected = torch.full((4,), 1.0) + dt * ((z_pred_exec - z1) / dt + gain * e1)
    assert torch.allclose(fb.z_bar, expected)
    assert fb.num_updates == 2


def test_feedback_error_dynamics_contract_for_stable_gain():
    """e_{t+1} = (1 - dt*l) e_t + dt*Delta_t: a constant model residual is absorbed (Appendix A)."""
    gain, dt = 1.0, 0.5
    fb = FeedbackState(gain=gain, dt=dt)
    true_velocity = torch.tensor([1.0, -1.0])
    model_bias = torch.tensor([0.3, 0.3])         # the model is biased by a constant Delta
    z = torch.zeros(2)
    fb.propagate(z, z + dt * (true_velocity + model_bias))
    residual_norms = []
    for _ in range(40):
        z = z + dt * true_velocity                # environment transition
        residual_norms.append(fb.residual(z).norm().item())
        fb.propagate(z, z + dt * (true_velocity + model_bias))
    # converges to the bound ||Delta|| / l (Eq. 13); here ||Delta|| / l = ||model_bias||
    assert residual_norms[-1] == pytest.approx(model_bias.norm().item(), abs=1e-3)
    assert abs(residual_norms[-1] - residual_norms[-2]) < 1e-4   # settled (no oscillation)


def test_feedback_reset_and_gain_warning():
    fb = FeedbackState(gain=0.5, dt=0.5)
    fb.propagate(torch.zeros(2), torch.ones(2))
    fb.reset()
    assert not fb.initialized and fb.num_updates == 0
    with pytest.warns(UserWarning):
        FeedbackState(gain=5.0, dt=0.5)           # dt*l = 2.5 > 2: unstable


def test_weight_normalisation_and_interpolation():
    raw = torch.tensor([1.0, 3.0, 0.0, 4.0])
    w_bar = normalize_weights(raw)
    assert w_bar.mean().item() == pytest.approx(1.0, rel=1e-5)
    assert torch.allclose(interpolate_weights(w_bar, 0.0), torch.ones(4))
    assert torch.allclose(interpolate_weights(w_bar, 1.0), w_bar)
    assert torch.allclose(interpolate_weights(w_bar, 0.3), 1 + 0.3 * (w_bar - 1))


def test_weighted_squared_error_reduces_to_plain_distance():
    diff = torch.randn(5, 7)
    assert torch.allclose(weighted_squared_error(diff), (diff ** 2).sum(-1))
    assert torch.allclose(weighted_squared_error(diff, torch.ones(7)), (diff ** 2).sum(-1))
    w = torch.rand(7)
    assert torch.allclose(weighted_squared_error(diff, w), (w * diff ** 2).sum(-1))


def test_estimate_controllability_recovers_action_sensitive_dimensions():
    """A linear 'world model' whose first dims depend on the action and whose last dims do not."""
    torch.manual_seed(0)
    D, Da = 6, 2
    A = torch.zeros(D, Da)
    A[0, 0], A[1, 1], A[2, 0] = 3.0, 2.0, 1.0     # dims 3..5 are not controllable

    def predict_fn(obs_idx, actions):
        return actions @ A.T + obs_idx * torch.ones(actions.shape[0], D)

    pool = torch.randn(500, Da)
    raw = estimate_controllability(predict_fn, num_observations_total=50, action_pool=pool,
                                   num_observations=10, num_actions=64,
                                   generator=torch.Generator().manual_seed(1), log_every=0)
    assert raw.shape == (D,)
    assert torch.all(raw[:3] > 0.5)
    assert torch.all(raw[3:] < 1e-6)
    assert raw[0] > raw[2]
