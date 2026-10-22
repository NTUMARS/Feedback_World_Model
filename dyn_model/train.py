"""Train the lightweight latent world model (Appendix F).

The visual encoder is the frozen ResNet backbone of a trained diffusion policy
(``env.policy_ckpt_path``); the proprioception encoder, the action-chunk encoder
and the Transformer predictor are trained with a one-step latent prediction
loss.  Run from the repository root::

    python dyn_model/train.py env=square
    python dyn_model/train.py env=libero10_single_task task=KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
"""

import itertools
import logging
import os
import random
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from einops import rearrange
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from diffusion_policy.common.language_models import extract_text_features
from dyn_model.models.language_encoder import LanguageEncoder
from dyn_model.models.resnet_encoder import ResNetEncoder

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
LANGUAGE_EMB_DIM = 32


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _import_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
        log.info(f"Model saved dir: {cfg['saved_folder']}")
        run_name = f"{Path(cfg['saved_folder']).name}_{cfg.env.name}_f{cfg.frameskip}"

        self.accelerator = Accelerator(log_with="wandb")
        self.device = self.accelerator.device
        self.total_epochs = cfg.training.epochs
        self.epoch = 0
        self.uses_language = "language" in cfg.env.get("shape_obs", {})

        assert cfg.training.batch_size % self.accelerator.num_processes == 0
        OmegaConf.set_struct(cfg, False)
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg.get("wandb_run_id", None)
            self.wandb_run = wandb.init(
                project=cfg.wandb.project,
                mode=cfg.wandb.mode,
                config=OmegaConf.to_container(cfg, resolve=True),
                id=wandb_run_id,
                resume="allow",
                name=run_name,
            )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        # ------------------------------------------------------------ data
        dataset_cls = _import_class(cfg.env.dataset_class)
        dataset_params = dict(
            num_hist=cfg.num_hist,
            num_pred=cfg.num_pred,
            frameskip=cfg.frameskip,
            view_names=cfg.env.view_names,
            abs_action=cfg.abs_action,
            use_crop=cfg.use_crop,
            original_img_size=cfg.env.original_img_size,
            cropped_img_size=cfg.env.cropped_img_size,
            action_dim=cfg.env.action_dim,
            shape_obs=cfg.env.shape_obs,
        )
        if cfg.env.get("data_dirs", None) is not None:
            dataset_params["data_dirs"] = list(cfg.env.data_dirs)
        train_dataset = dataset_cls(zarr_path=cfg.env.train_data_path, train=True, **dataset_params)
        valid_dataset = dataset_cls(zarr_path=cfg.env.val_data_path, train=False, **dataset_params)

        self.cropped_image_size = cfg.env.cropped_img_size
        self.original_img_size = train_dataset.original_img_size
        self.normalizer = train_dataset.get_normalizer().to(self.device)
        self.train_img_transform = train_dataset.transform
        self.valid_img_transform = valid_dataset.transform
        torch.save(self.normalizer.state_dict(), os.path.join(cfg["saved_folder"], "normalizer.pth"))

        self.datasets = {"train": train_dataset, "valid": valid_dataset}
        self.dataloaders = {
            split: torch.utils.data.DataLoader(
                self.datasets[split], batch_size=cfg.gpu_batch_size, shuffle=True,
                num_workers=cfg.training.num_workers, prefetch_factor=8)
            for split in ["train", "valid"]
        }
        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"])

        # ---------------------------------------------------------- models
        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.language_encoder = None
        self._keys_to_save = ["epoch", "predictor", "predictor_optimizer", "action_encoder", "proprio_encoder"]
        self.init_models()
        self.init_optimizers()
        self.epoch_log = OrderedDict()

    # ------------------------------------------------------------------ io
    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt = {}
            for key in self._keys_to_save:
                item = self.__dict__.get(key, None)
                if item is None:
                    continue
                if isinstance(item, torch.nn.Module):
                    item = self.accelerator.unwrap_model(item)
                    ckpt[key] = item.state_dict()
                elif isinstance(item, torch.optim.Optimizer):
                    ckpt[key] = item.state_dict()
                else:
                    ckpt[key] = item
            ckpt["epoch"] = self.epoch
            torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            torch.save(ckpt, "checkpoints/model_latest.pth")
            log.info(f"Saved checkpoint to {os.getcwd()}/checkpoints/model_{self.epoch}.pth")

    def load_ckpt(self, filename):
        ckpt = torch.load(filename, map_location=self.device)
        self.epoch = ckpt.get("epoch", 0)
        for key in self._keys_to_save:
            if key in ckpt and key in self.__dict__ and self.__dict__[key] is not None:
                if hasattr(self.__dict__[key], "load_state_dict"):
                    self.__dict__[key].load_state_dict(ckpt[key])
                else:
                    self.__dict__[key] = ckpt[key]

    def init_models(self):
        cfg = self.cfg
        # frozen visual encoder shared with the diffusion policy
        self.encoder = ResNetEncoder(policy_ckpt_path=cfg.env.policy_ckpt_path, view_names=cfg.env.view_names)
        for p in self.encoder.parameters():
            p.requires_grad = False

        language_emb_dim = 0
        if self.uses_language:
            self.language_encoder = LanguageEncoder(policy_ckpt_path=cfg.env.policy_ckpt_path)
            for p in self.language_encoder.parameters():
                p.requires_grad = False
            self.language_encoder = self.accelerator.prepare(self.language_encoder)
            language_emb_dim = LANGUAGE_EMB_DIM

        self.proprio_encoder = hydra.utils.instantiate(
            cfg.proprio_encoder, in_chans=self.datasets["train"].proprio_dim, emb_dim=cfg.env.proprio_emb_dim)
        self.action_encoder = hydra.utils.instantiate(
            cfg.action_encoder, in_chans=self.datasets["train"].action_dim, emb_dim=cfg.env.action_emb_dim)
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)
        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        visual_dim = self.encoder.emb_dim * len(cfg.env.view_names)
        self.predictor = hydra.utils.instantiate(
            cfg.predictor,
            num_patches=1,
            num_frames=cfg.num_hist,
            dim=visual_dim + cfg.env.proprio_emb_dim + cfg.env.action_emb_dim + language_emb_dim,
            visual_dim=visual_dim,
            proprio_dim=cfg.env.proprio_emb_dim,
            action_dim=cfg.env.action_emb_dim,
        )
        self.encoder, self.predictor = self.accelerator.prepare(self.encoder, self.predictor)
        self.encoder.eval()

        resume_ckpt = Path(cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if resume_ckpt.exists():
            self.load_ckpt(resume_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {resume_ckpt}")

        self.model = hydra.utils.instantiate(
            cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            proprio_dim=cfg.env.proprio_emb_dim,
            action_dim=cfg.env.action_emb_dim,
            view_names=cfg.env.view_names,
            use_layernorm=cfg.use_layernorm,
            language_encoder=self.language_encoder,
        )
        for name, module in [("predictor", self.predictor), ("proprio encoder", self.proprio_encoder),
                             ("action encoder", self.action_encoder)]:
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            log.info(f"{name}: {n_params:,} trainable parameters")

    def init_optimizers(self):
        self.predictor_optimizer = torch.optim.AdamW(
            [p for p in self.predictor.parameters() if p.requires_grad], lr=self.cfg.training.predictor_lr)
        self.action_encoder_optimizer = torch.optim.AdamW(
            itertools.chain(self.action_encoder.parameters(), self.proprio_encoder.parameters()),
            lr=self.cfg.training.action_encoder_lr)
        self.predictor_optimizer, self.action_encoder_optimizer = self.accelerator.prepare(
            self.predictor_optimizer, self.action_encoder_optimizer)

    # ------------------------------------------------------------- loop
    def run(self):
        init_epoch = self.epoch + 1
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            self.val()
            self.logs_flash(step=self.epoch)
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                self.save_ckpt()

    def _prepare_batch(self, data, train: bool):
        obs, act, state = data
        n_frames = self.cfg.num_hist + self.cfg.num_pred
        transform = self.train_img_transform if train else self.valid_img_transform
        for view_name in self.cfg.env.view_names:
            img = self.normalizer[view_name].normalize(obs["visual"][view_name])
            if train:
                img = torch.stack([transform(x) for x in img])  # per-sample random crop
            else:
                img = transform(img.view(-1, 3, self.original_img_size, self.original_img_size))
            obs["visual"][view_name] = img.view(-1, n_frames, 3, self.cropped_image_size, self.cropped_image_size)
        obs["proprio"] = self.normalizer["state"].normalize(obs["proprio"])
        if self.uses_language and obs.get("language", None) is not None:
            tokens = obs["language"]
            text_tokens = {"input_ids": tokens[:, 0].long()[:, 0], "attention_mask": tokens[:, 0].long()[:, 1]}
            obs["language"] = extract_text_features(self.language_encoder, text_tokens, language_emb_model="clip")
        act = self.normalizer["act"].normalize(act)
        act = rearrange(act, "b (n f) d -> b n (f d)", n=n_frames, d=self.datasets["train"].original_action_dim)
        act[:, -1:, :] = 0  # the action after the predicted frame is unknown
        return obs, act

    def train(self):
        for data in tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train"):
            obs, act = self._prepare_batch(data, train=True)
            self.model.train()
            with self.accelerator.autocast():
                loss, loss_components = self.model(obs, act)
            self.predictor_optimizer.zero_grad()
            self.action_encoder_optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.predictor_optimizer.step()
            self.action_encoder_optimizer.step()
            self._log_components(loss_components, prefix="train_")

    def val(self):
        self.model.eval()
        for data in tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid"):
            obs, act = self._prepare_batch(data, train=False)
            with torch.no_grad():
                loss, loss_components = self.model(obs, act)
            self._log_components(loss_components, prefix="val_")

    def _log_components(self, loss_components, prefix):
        gathered = self.accelerator.gather_for_metrics(loss_components)
        for key, value in gathered.items():
            count, total = self.epoch_log.get(prefix + key, (0, 0.0))
            self.epoch_log[prefix + key] = (count + 1, total + value.detach().cpu().mean().item())

    def logs_flash(self, step):
        epoch_log = {key: total / count for key, (count, total) in self.epoch_log.items()}
        epoch_log["epoch"] = step
        log.info(f"Epoch {self.epoch}  train loss {epoch_log.get('train_loss', float('nan')):.4f}  "
                 f"val loss {epoch_log.get('val_loss', float('nan')):.4f}")
        if self.accelerator.is_main_process:
            self.wandb_run.log(epoch_log)
        self.epoch_log = OrderedDict()


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg):
    seed_everything(cfg.training.seed)
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
