"""Action-aware guidance energy (Sec. 4.3 of the paper).

The controllability of the ``j``-th latent dimension is estimated offline by its
counterfactual variance under the learned world model (Eq. 14)::

    w_j = 1/M * sum_i Var_{a ~ A_demo}( [f_theta(z_i, a)]_j )

The weights are normalised to unit mean and softly interpolated with uniform
weighting (Eq. 15)::

    w_bar_j      = w_j / (1/D * sum_k w_k + eps)
    w_j^(beta)   = 1 + beta * (w_bar_j - 1)

and replace the plain squared error in the guidance energy (Eq. 16)::

    E_ctrl(A) = sum_j w_j^(beta) * (z_fb_{t+1,j}(A) - z^E_{i*,j})^2
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch


def normalize_weights(raw_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Unit-mean normalisation ``w_bar = w / (mean(w) + eps)`` (Eq. 15, left)."""
    return raw_weights / (raw_weights.mean() + eps)


def interpolate_weights(normalized_weights: torch.Tensor, beta: float) -> torch.Tensor:
    """Soft interpolation ``w^(beta) = 1 + beta (w_bar - 1)`` (Eq. 15, right).

    ``beta = 0`` recovers the uniform (standard) energy; ``beta = 1`` uses the
    fully controllability-weighted energy.
    """
    return 1.0 + float(beta) * (normalized_weights - 1.0)


def weighted_squared_error(diff: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-sample energy ``sum_j w_j diff_j^2`` for ``diff`` of shape ``(B, D)``.

    With ``weights=None`` this is the standard latent distance of Eq. 6.
    """
    if weights is None:
        return (diff ** 2).sum(dim=-1)
    return (weights.to(diff.device).reshape(1, -1) * diff ** 2).sum(dim=-1)


@torch.no_grad()
def estimate_controllability(
    predict_fn: Callable[[int, torch.Tensor], torch.Tensor],
    num_observations_total: int,
    action_pool: torch.Tensor,
    num_observations: int = 200,
    num_actions: int = 32,
    generator: Optional[torch.Generator] = None,
    log_every: int = 50,
) -> torch.Tensor:
    """Counterfactual-variance controllability estimate (Eq. 14).

    Args:
        predict_fn: ``predict_fn(obs_index, actions)`` returns the predicted latent
            ``f_theta(z_i, a)`` for every action in ``actions`` (shape ``(K, Da)``)
            as a tensor of shape ``(K, D)`` in the energy latent space.
        num_observations_total: number of demonstration frames available
            (observation indices are drawn uniformly spaced in ``[0, N)``).
        action_pool: demonstration actions ``A_demo`` of shape ``(N_a, Da)`` from
            which the ``K`` counterfactual actions are sampled uniformly.
        num_observations: ``M`` (the paper uses 200).
        num_actions: ``K`` (the paper uses 32).
        generator: optional CPU ``torch.Generator`` for reproducible sampling.

    Returns:
        Raw (un-normalised) weights ``w`` of shape ``(D,)`` on CPU.
    """
    if num_observations_total <= 0:
        raise ValueError("expert memory is empty; cannot estimate controllability")
    M = min(int(num_observations), int(num_observations_total))
    K = int(num_actions)
    obs_indices = torch.linspace(0, num_observations_total - 1, M).long()

    variances = []
    for i, obs_idx in enumerate(obs_indices.tolist()):
        action_idx = torch.randint(0, action_pool.shape[0], (K,), generator=generator)
        actions = action_pool[action_idx]
        z_pred = predict_fn(obs_idx, actions)          # (K, D)
        variances.append(z_pred.var(dim=0).detach().cpu())
        if log_every and (i + 1) % log_every == 0:
            print(f"[action-aware] controllability: {i + 1}/{M} observations")

    return torch.stack(variances, dim=0).mean(dim=0)


def summarize_weights(weights: torch.Tensor, top_fraction: float = 0.05) -> dict:
    """Small summary used for logging (CV and mass of the top dimensions)."""
    w = weights.detach().cpu().float()
    sorted_w, _ = torch.sort(w, descending=True)
    n_top = max(1, int(round(top_fraction * w.numel())))
    return {
        "dim": int(w.numel()),
        "cv": float(w.std() / (w.mean() + 1e-8)),
        "max": float(w.max()),
        "min": float(w.min()),
        f"top{int(top_fraction * 100)}pct_mass": float(sorted_w[:n_top].sum() / (w.sum() + 1e-8)),
    }
