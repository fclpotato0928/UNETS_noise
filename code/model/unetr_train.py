import os
import shutil
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR


class UnetrTrainer:
    def __init__(self, cfg, dataloader, val_loader=None):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = UNETR(
            in_channels=3,
            out_channels=1,
            img_size=cfg.model.img_size,   
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
            spatial_dims=2,
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
            print("pertrained weights loaded")

            
