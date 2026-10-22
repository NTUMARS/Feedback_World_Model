"""Feedback world model (FWM) guidance for diffusion policies.

Modules
-------
feedback      -- the latent-space feedback state (Sec. 4.2, Eqs. 9-12)
action_aware  -- controllability weights and the action-aware energy (Sec. 4.3, Eqs. 14-16)
planner       -- ``FeedbackGuidedPlanner``: world-model guidance used inside diffusion sampling
"""

from fwm.feedback import FeedbackState
from fwm.action_aware import (
    estimate_controllability,
    interpolate_weights,
    normalize_weights,
    weighted_squared_error,
)
from fwm.planner import FeedbackGuidedPlanner, PlannerConfig

__all__ = [
    "FeedbackState",
    "FeedbackGuidedPlanner",
    "PlannerConfig",
    "estimate_controllability",
    "interpolate_weights",
    "normalize_weights",
    "weighted_squared_error",
]
