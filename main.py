import hydra
import wandb
from omegaconf import OmegaConf

from code.data import UnetbreastModule, RetinaModule
from code.model import UNetTrainer


@hydra.main(version_base=None, config_path="config", config_name="full_baseline")
def main(cfg):
    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    if cfg.dataset.name == "retina":
        data_module = RetinaModule(cfg)
    elif cfg.dataset.name == "breast":
        data_module = UnetbreastModule(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    
    train_loader = data_module.get_loader("train")
    val_loader   = data_module.get_loader("val")

    trainer = UNetTrainer(cfg, train_loader, val_loader)
    trainer.train()

    if cfg.wandb.enable:
        wandb.finish()


if __name__ == "__main__":
    main()
