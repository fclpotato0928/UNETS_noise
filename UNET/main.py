import hydra
import wandb
from omegaconf import OmegaConf

from code.data import UnetbreastModule
from code.model import UNetTrainer


@hydra.main(version_base=None, config_path="config", config_name="config_transition")
def main(cfg):
    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    data_module = UnetbreastModule(cfg)
    train_loader = data_module.get_loader("train")
    val_loader   = data_module.get_loader("val")

    trainer = UNetTrainer(cfg, train_loader, val_loader)
    trainer.train()

    if cfg.wandb.enable:
        wandb.finish()


if __name__ == "__main__":
    main()
