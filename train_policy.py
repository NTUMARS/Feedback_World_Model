"""Train the base diffusion policy (Diffusion Policy, UNet/CNN variant).

Examples (run from the repository root)::

    python train_policy.py --config-name=robomimic_square training.device=cuda:0
    python train_policy.py --config-name=libero10_single_task \
        task_name=KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it

Checkpoints and ``normalizer.pth`` are written to ``hydra.run.dir``.
LIBERO policies evaluate rollouts during training and therefore need the
original LIBERO repository (see README, "External simulators").
"""
import os
import pathlib
import sys

# line-buffered logs when redirected to a file
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

# Off-screen EGL rendering must be configured before robomimic / robosuite are imported.
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '0')
import egl_probe


def _patched_get_available_devices():
    return [int(os.environ.get('MUJOCO_EGL_DEVICE_ID', '0'))]


egl_probe.get_available_devices = _patched_get_available_devices  # skip device probing

import hydra
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('configs', 'policy')),
    config_name='robomimic_square',
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)  # resolve ${now:} once so every use sees the same timestamp
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
