import os
import torch
from torch.cuda.amp import autocast, GradScaler
from monai.networks.nets import UNet
from monai.losses import DiceLoss
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
from ..utils import dice_score


class UNetTrainer:
    def __init__(self, cfg, train_loader, val_loader=None):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            num_res_units=2,
        ).to(self.device)
        if self.cfg.training.mode == "finetune":
            assert self.cfg.training.pretrained_path is not None

            ckpt = torch.load(
                self.cfg.training.pretrained_path,
                map_location="cpu"
            )
            self.model.load_state_dict(
                ckpt["model_state_dict"],
                strict=False
            )
            print("[Trainer] Pretrained weights loaded")

            if getattr(self.cfg.training, "freeze_encoder", False):
                for name, param in self.model.named_parameters():
                    if "down" in name or "encoder" in name:
                        param.requires_grad = False
                print("[Trainer] Encoder frozen")

        self.loss_fn = DiceLoss(sigmoid=True)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.training.lr,
        )
        
        # Mixed Precision을 위한 GradScaler
        self.scaler = GradScaler()

        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # ===== AnnealedGaussianLabeld 찾기 (Dataset transform에서) =====
        self.annealed_transform = None
        if hasattr(train_loader.dataset, 'transform'):
            for t in train_loader.dataset.transform.transforms:
                # isinstance 대신 클래스 이름으로 체크
                if type(t).__name__ == 'AnnealedGaussianLabeld':
                    self.annealed_transform = t
                    print(f"[Trainer] AnnealedGaussianLabeld detected (max_epochs={t.max_epochs}, mode={t.mode})")
                    break
        
        self.best_dice = 0.0
        if cfg.wandb.enable and wandb.run is not None:
            run_name = wandb.run.name
        else:
            run_name = "debug"
        self.ckpt_dir = os.path.join(cfg.training.ckpt_dir, run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train(self):
        for epoch in range(self.cfg.training.epochs):
            # ===== AnnealedGaussianLabeld에 현재 epoch 설정 =====
            if self.annealed_transform is not None:
                self.annealed_transform.set_epoch(epoch)
            
            train_loss = self._train_one_epoch(epoch)

            if self.val_loader is not None:
                val_loss, val_dice = self.validate(epoch)

                # ===== best Dice 기준 저장 =====
                if val_dice > self.best_dice:
                    self.best_dice = val_dice
                    self._save_checkpoint(epoch)

                if self.cfg.wandb.enable:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/dice": val_dice,
                        "best/dice": self.best_dice,
                    })

                print(
                    f"[Epoch {epoch+1}] "
                    f"train: {train_loss:.4f} | "
                    f"val: {val_loss:.4f} | "
                    f"dice: {val_dice:.4f} | "
                    f"best: {self.best_dice:.4f}"
                )
            else:
                if self.cfg.wandb.enable:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                    })

                print(f"[Epoch {epoch+1}] train: {train_loss:.4f}")

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.cfg.training.epochs} [Train]",
            ncols=100
        )

        for batch in pbar:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            
            # Mixed Precision 학습
            with autocast(dtype=torch.bfloat16):  # RTX 5070 Ti는 BF16 최적화됨
                pred = self.model(x)
                loss = self.loss_fn(pred, y)

            # Gradient Scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            
            # 진행률 바에 현재 loss 표시
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (pbar.n + 1):.4f}'
            })

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0

        # Validation 진행률 바
        pbar = tqdm(
            self.val_loader, 
            desc=f"Epoch {epoch+1}/{self.cfg.training.epochs} [Val]",
            ncols=100
        )

        for batch in pbar:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            # Validation도 Mixed Precision 사용 (더 빠름)
            with autocast(dtype=torch.bfloat16):
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                dice = dice_score(pred, y)

            total_loss += loss.item()
            total_dice += dice.item()
            
            # 진행률 바에 현재 dice 표시
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice.item():.4f}'
            })

        return (
            total_loss / len(self.val_loader),
            total_dice / len(self.val_loader),
        )

    def _save_checkpoint(self, epoch):
        ckpt_path = os.path.join(
            self.ckpt_dir,
            "best_model.pth"
        )

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),  # Scaler 상태도 저장
            "best_dice": self.best_dice,
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }, ckpt_path)

        if self.cfg.wandb.enable:
            wandb.save(ckpt_path)