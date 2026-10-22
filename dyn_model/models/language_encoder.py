"""Frozen text encoder taken from a trained language-conditioned diffusion policy."""
import torch
import hydra
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class LanguageEncoder(torch.nn.Module):
    """Wraps the policy's text model so the world model shares its language embedding."""
    def __init__(self, policy_ckpt_path):
        super().__init__()
        self.policy_ckpt_path = policy_ckpt_path

        with open(self.policy_ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill)
            cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(payload['cfg'], output_dir='debug_obs_encoder')
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy.to(device)
        policy.eval()

        self.text_model = policy.text_model
        self.text_model.eval()  

        del workspace
        del policy
        torch.cuda.empty_cache()

    def forward(self, text_tokens):
        text_latents = self.text_model(text_tokens)  # (batch_size, 32)
        text_latents = text_latents.unsqueeze(1) # dummy patch dim

        return text_latents
