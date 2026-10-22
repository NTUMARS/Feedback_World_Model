"""Reconstruct a trained latent world model from a ``dyn_model/train.py`` run."""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from dyn_model.models.language_encoder import LanguageEncoder
from dyn_model.models.resnet_encoder import ResNetEncoder

LANGUAGE_EMB_DIM = 32


def _uses_language(train_cfg: DictConfig) -> bool:
    return "language" in train_cfg.env.get("shape_obs", {})


def load_world_model(ckpt_path: Path, train_cfg: DictConfig, device) -> torch.nn.Module:
    """Instantiate the world model described by ``train_cfg`` and load ``ckpt_path``.

    ``train_cfg`` is the ``hydra.yaml`` written next to the checkpoints by
    ``dyn_model/train.py``.  The visual encoder is the frozen ResNet backbone of
    the diffusion policy referenced by ``train_cfg.env.policy_ckpt_path``.
    """
    ckpt = torch.load(Path(ckpt_path).open("rb"), map_location=device)

    encoder = ResNetEncoder(policy_ckpt_path=train_cfg.env.policy_ckpt_path, view_names=train_cfg.view_names)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])

    language_encoder = None
    language_emb_dim = 0
    if _uses_language(train_cfg):
        language_encoder = LanguageEncoder(policy_ckpt_path=train_cfg.env.policy_ckpt_path)
        for p in language_encoder.parameters():
            p.requires_grad = False
        language_emb_dim = LANGUAGE_EMB_DIM

    action_dim = int(train_cfg.env.action_dim)
    proprio_dim = int(train_cfg.env.proprio_dim)
    action_encoder = hydra.utils.instantiate(
        train_cfg.action_encoder, in_chans=action_dim * int(train_cfg.frameskip), emb_dim=train_cfg.action_emb_dim
    )
    action_encoder.load_state_dict(ckpt["action_encoder"])
    proprio_encoder = hydra.utils.instantiate(
        train_cfg.proprio_encoder, in_chans=proprio_dim, emb_dim=train_cfg.proprio_emb_dim
    )
    proprio_encoder.load_state_dict(ckpt["proprio_encoder"])

    visual_dim = encoder.emb_dim * len(train_cfg.view_names)
    predictor = hydra.utils.instantiate(
        train_cfg.predictor,
        num_patches=1,
        num_frames=train_cfg.num_hist,
        dim=visual_dim + proprio_encoder.emb_dim + action_encoder.emb_dim + language_emb_dim,
        visual_dim=visual_dim,
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
    )
    predictor.load_state_dict(ckpt["predictor"])

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        view_names=train_cfg.view_names,
        use_layernorm=train_cfg.use_layernorm,
        language_encoder=language_encoder,
    )
    model.to(device)
    model.eval()
    return model
