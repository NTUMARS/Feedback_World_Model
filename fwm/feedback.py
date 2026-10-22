"""Feedback state of the feedback world model (Sec. 4.2 of the paper).

The world model ``f_theta`` is reinterpreted in velocity form (Eq. 8)::

    v_t(A) = (f_theta(z_t, A) - z_t) / dt

and an auxiliary feedback state ``z_bar_t`` (initialised to ``z_0``) tracks the
model's internal belief about where the system should be after the actions
executed so far.  The residual ``e_t = z_t - z_bar_t`` (Eq. 9) is held fixed
during one denoising process and corrects every candidate prediction (Eq. 10)::

    v_hat_t(A)     = v_t(A) + l * e_t
    z_fb_{t+1}(A)  = z_t + dt * v_hat_t(A) = f_theta(z_t, A) + dt * l * e_t

After the clean action ``A^(0)`` has been executed, the feedback state is
propagated with the same corrected velocity (Eq. 11)::

    z_bar_{t+1} = z_bar_t + dt * v_hat_t(A^(0))

With a scalar gain ``L = l I`` the discrete error dynamics are
``e_{t+1} = (1 - dt*l) e_t + dt*Delta_t`` (Appendix A), so ``0 < dt*l < 2`` is
required for stability and ``0 < dt*l <= 1`` gives a monotone, non-oscillatory
correction.  All paper settings satisfy the latter.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch


class FeedbackState:
    """Scalar-gain latent observer shared by all tasks.

    Args:
        gain: feedback gain ``l`` (the paper uses ``L = l I``).
        dt:   environment update interval in seconds, i.e. the duration of one
              executed action chunk (``n_action_steps / control_freq``).
    """

    def __init__(self, gain: float, dt: float):
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if gain < 0:
            raise ValueError(f"feedback gain must be non-negative, got {gain}")
        self.gain = float(gain)
        self.dt = float(dt)
        self.z_bar: Optional[torch.Tensor] = None
        self.num_updates = 0

        effective = self.gain * self.dt
        if effective >= 2.0:
            warnings.warn(
                f"dt*l = {effective:.3f} >= 2: the feedback error dynamics are unstable "
                f"(Appendix A). Reduce the gain.",
                stacklevel=2,
            )
        elif effective > 1.0:
            warnings.warn(
                f"dt*l = {effective:.3f} > 1: the correction over-shoots (oscillatory). "
                f"The paper uses 0 < dt*l <= 1.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Forget the feedback state at the beginning of a new episode."""
        self.z_bar = None
        self.num_updates = 0

    @property
    def initialized(self) -> bool:
        return self.z_bar is not None

    # ------------------------------------------------------------ Eq. 9 / 10
    def residual(self, z_obs: torch.Tensor) -> torch.Tensor:
        """Feedback residual ``e_t = z_t - z_bar_t`` (Eq. 9).

        Before the first executed action the belief equals the observation,
        so the residual is zero.
        """
        if self.z_bar is None:
            return torch.zeros_like(z_obs)
        return z_obs - self.z_bar.to(z_obs.device)

    def correction(self, z_obs: torch.Tensor) -> torch.Tensor:
        """Constant additive correction ``dt * l * e_t`` applied to predictions."""
        return self.dt * self.gain * self.residual(z_obs)

    def correct(self, z_pred: torch.Tensor, z_obs: torch.Tensor) -> torch.Tensor:
        """Feedback-corrected prediction ``z_fb_{t+1}(A)`` (Eq. 10).

        Args:
            z_pred: open-loop predictions ``f_theta(z_t, A)`` of shape ``(B, D)``
                    (gradients w.r.t. the candidate actions flow through here).
            z_obs:  observed latent ``z_t`` of shape ``(D,)`` or ``(1, D)``.
        """
        corr = self.correction(z_obs.detach()).reshape(1, -1)
        return z_pred + corr

    # ----------------------------------------------------------------- Eq. 11
    @torch.no_grad()
    def propagate(self, z_obs: torch.Tensor, z_pred_executed: torch.Tensor) -> None:
        """Advance the belief with the executed action (Eq. 11).

        Args:
            z_obs:            observed latent ``z_t`` of shape ``(D,)``.
            z_pred_executed:  open-loop prediction ``f_theta(z_t, A^(0))`` for the
                              action chunk that is actually executed, shape ``(D,)``.
        """
        z_obs = z_obs.detach().reshape(-1)
        z_pred_executed = z_pred_executed.detach().reshape(-1)
        if self.z_bar is None:
            # z_bar_0 = z_0: the first update carries no residual.
            self.z_bar = z_obs.clone()
        velocity = (z_pred_executed - z_obs) / self.dt                  # v_t(A^(0))
        corrected_velocity = velocity + self.gain * (z_obs - self.z_bar)  # v_hat_t
        self.z_bar = self.z_bar + self.dt * corrected_velocity           # z_bar_{t+1}
        self.num_updates += 1
