from monai.data import Dataset, DataLoader
import os
import random
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    Resized,
    Lambdad,
    MapTransform,
)


class LabelSmoothingd(MapTransform):
    """
    Label Smoothing Transform for Segmentation
    0 → epsilon (e.g., 1e-4)
    1 → 1 - epsilon (e.g., 0.9999)
    
    Args:
        keys: 적용할 키 (보통 "label")
        epsilon: smoothing 값 (기본: 1e-4)
    """
    def __init__(self, keys, epsilon=1e-4):
        super().__init__(keys)
        self.epsilon = epsilon
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            # 0 → epsilon, 1 → 1-epsilon
            smoothed = torch.where(
                label == 0,
                torch.full_like(label, self.epsilon),
                torch.full_like(label, 1.0 - self.epsilon)
            )
            d[key] = smoothed
        return d


class AnnealedGaussianLabeld(MapTransform):
    """
    Epoch-dependent label smoothing (Gaussian-like softening → discrete)

    p=1 → soft (Gaussian-like)
    p=0 → hard (0/1)

    Trainer에서 set_epoch(epoch) 반드시 호출해야 함
    """
    def __init__(
        self,
        keys,
        max_epochs,
        eps_start=0.499,
        eps_end=0.0,
        mode="linear",  # linear | cosine
    ):
        super().__init__(keys)
        self.max_epochs = max_epochs
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.mode = mode
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _epsilon(self):
        t = min(self.epoch / self.max_epochs, 1.0)

        if self.mode == "cosine":
            t = 0.5 * (1 + torch.cos(torch.tensor(t * torch.pi)))

        return self.eps_end + (self.eps_start - self.eps_end) * (1 - t)

    def __call__(self, data):
        d = dict(data)
        eps = self._epsilon()

        for key in self.key_iterator(d):
            label = d[key]

            # hard → soft → hard 복귀
            soft = label * (1 - 2 * eps) + eps
            d[key] = soft.clamp(0.0, 1.0)

        return d




class UnetbreastModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_root = cfg.dataset.data_root
        self.num_workers = cfg.training.num_workers
        self.pin_memory = cfg.dataset.pin_memory

        self.valid_classes = ["benign", "malignant"]
        self.split_ratio = 0.8
        self.seed = 42
        self.image_size = (512, 512)

        # Label Smoothing 설정
        self.label_smoothing_epsilon = getattr(
            cfg.dataset, 'label_smoothing_epsilon', 0.0
        )

        self._all_files = self._build_file_list()
        self._train_files, self._val_files = self._split_files()

    def _get_transforms(self, split="train"):
        keys = ["image", "label"]

        if split == "train":
            transforms_list = [
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                # 이미지를 무조건 3채널로 통일
                Lambdad(
                    keys="image", 
                    func=lambda x: (
                        x.repeat(3, 1, 1) if x.shape[0] == 1  # 1채널 -> 3채널 복제
                        else x[:3] if x.shape[0] >= 3          # 4채널(RGBA) -> 3채널(RGB)
                        else x.repeat(3 // x.shape[0] + 1, 1, 1)[:3]  # 2채널 등 예외 처리
                    )
                ),
                # 마스크를 1채널로 통일
                Lambdad(
                    keys="label", 
                    func=lambda x: x[:1] if x.shape[0] >= 1 else x
                ),
                Resized(keys=keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]),
                ScaleIntensityd(keys="image"),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandRotate90d(keys=keys, prob=0.5),
            ]
            
            # ===== Label Smoothing 적용 (Train only) =====
            if self.label_smoothing_epsilon > 0:
                transforms_list.append(
                    LabelSmoothingd(keys="label", epsilon=self.label_smoothing_epsilon)
                )
                print(f"[Dataset] Label Smoothing enabled (epsilon={self.label_smoothing_epsilon})")
            
            # ===== Annealed Gaussian Label Smoothing (Train only) =====
            if self.cfg.training.get("use_annealed_smoothing", False):
                annealed_transform = AnnealedGaussianLabeld(
                    keys=["label"],
                    max_epochs=self.cfg.training.get("annealed_max_epochs", self.cfg.training.epochs),
                    eps_start=self.cfg.training.get("annealed_eps_start", 0.499),
                    eps_end=self.cfg.training.get("annealed_eps_end", 0.0),
                    mode=self.cfg.training.get("annealed_mode", "cosine")
                )
                transforms_list.append(annealed_transform)
                print(f"[Dataset] AnnealedGaussianLabeld enabled "
                      f"(max_epochs={annealed_transform.max_epochs}, "
                      f"eps_start={annealed_transform.eps_start}, "
                      f"eps_end={annealed_transform.eps_end}, "
                      f"mode={annealed_transform.mode})")
            
            transforms_list.append(EnsureTyped(keys=keys))
            
            return Compose(transforms_list)
            
        else:  # validation/test
            # Validation에는 원본 label 사용 (정확한 평가)
            return Compose([
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                # 이미지를 무조건 3채널로 통일
                Lambdad(
                    keys="image", 
                    func=lambda x: (
                        x.repeat(3, 1, 1) if x.shape[0] == 1  # 1채널 -> 3채널 복제
                        else x[:3] if x.shape[0] >= 3          # 4채널(RGBA) -> 3채널(RGB)
                        else x.repeat(3 // x.shape[0] + 1, 1, 1)[:3]  # 2채널 등 예외 처리
                    )
                ),
                # 마스크를 1채널로 통일
                Lambdad(
                    keys="label", 
                    func=lambda x: x[:1] if x.shape[0] >= 1 else x
                ),
                Resized(keys=keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]),
                ScaleIntensityd(keys="image"),
                EnsureTyped(keys=keys),
            ])
            
    def _build_file_list(self):
        files = []

        # breast_ultrasound 폴더 찾기
        breast_ultrasound_dir = os.path.join(self.data_root, "breast_ultrasound")
        
        # breast_ultrasound 폴더가 없으면 data_root 직접 사용
        if os.path.isdir(breast_ultrasound_dir):
            base_dir = breast_ultrasound_dir
        else:
            base_dir = self.data_root

        for cls in self.valid_classes:
            class_dir = os.path.join(base_dir, cls)
            if not os.path.isdir(class_dir):
                print(f"[Warning] Directory not found: {class_dir}")
                continue

            for fname in os.listdir(class_dir):
                if not fname.lower().endswith(".png"):
                    continue
                if "_mask.png" in fname:
                    continue

                image_path = os.path.join(class_dir, fname)
                
                mask_patterns = [
                    fname.replace(".png", "_mask.png"),
                    fname.replace(" (", " (").replace(").png", ")_mask.png"),
                ]
                
                mask_path = None
                for pattern in mask_patterns:
                    potential_mask = os.path.join(class_dir, pattern)
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                        break

                if mask_path is None:
                    print(f"[Warning] Mask not found for: {fname}")
                    continue

                files.append({
                    "image": image_path,
                    "label": mask_path,
                })

        if len(files) == 0:
            raise RuntimeError(
                f"[UnetbreastModule] No image-mask pairs found in {base_dir}"
            )

        print(f"[UnetbreastModule] Found {len(files)} image-mask pairs")
        return files

    def _split_files(self):
        random.seed(self.seed)
        files = self._all_files.copy()
        random.shuffle(files)

        split_idx = int(len(files) * self.split_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        print(f"[UnetbreastModule] Train: {len(train_files)}, Val: {len(val_files)}")
        return train_files, val_files

    def get_loader(self, split="train"):
        assert split in ["train", "val", "test"]

        if split == "train":
            files = self._train_files
        elif split == "val":
            files = self._val_files
        else:
            files = self._val_files

        transforms = self._get_transforms(split)
        dataset = Dataset(files, transform=transforms)

        loader = DataLoader(
            dataset,
            batch_size=(
                self.cfg.training.batch_size
                if split == "train"
                else self.cfg.evaluation.batch_size
            ),
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader