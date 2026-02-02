import os
import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    ScaleIntensityd, 
    Resized, 
    Lambdad, 
    EnsureTyped
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss

def dice_score(pred, target):
    """
    Dice Score 계산 (Robust version)
    """
    # 예측값 이진화 (0.5 기준)
    pred = (pred > 0.5).float()
    
    # 정답값 이진화 (혹시 모를 0~255 값 대응을 위해 0.5보다 크면 1로 간주)
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0
    
    dice = (2.0 * intersection) / union
    return dice.item()

def evaluate_zeroshot(
    checkpoint_path,
    image_folder,
    mask_folder,
    output_csv="evaluation_results.csv",
    output_mask_folder=None,
    batch_size=4,
    image_size=(512, 512),
    device=None,
    threshold=0.5,
    save_masks=False,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation] Using device: {device}")
    
    # 경로 확인
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print("[Evaluation] Loading model...")
    # 모델 구조 (학습 때와 동일해야 함)
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(64, 128, 256, 512),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print(f"[Evaluation] Loaded checkpoint. Best Dice: {ckpt.get('best_dice', 'N/A')}")
    
    # 파일 찾기
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png"))) + \
                  sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {image_folder}")
    
    data_pairs = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        # 마스크 매칭 로직
        mask_patterns = [
            filename,
            filename.replace(".png", "_mask.png"),
            filename.replace(".jpg", "_mask.png"),
        ]
        
        mask_path = None
        for pattern in mask_patterns:
            potential = os.path.join(mask_folder, pattern)
            if os.path.exists(potential):
                mask_path = potential
                break
        
        if mask_path:
            data_pairs.append({"image": img_path, "label": mask_path})
        else:
            print(f"[Warning] Skipping {filename} (No mask found)")

    if not data_pairs:
        raise RuntimeError("No valid pairs found.")
        
    print(f"[Evaluation] Evaluating {len(data_pairs)} pairs.")

    # ==========================================
    # [수정 1] Transforms 수정 (가장 중요)
    # ==========================================
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # 이미지 채널 강제 3채널 (RGB)
        Lambdad(keys="image", func=lambda x: x[:3] if x.shape[0] >= 3 else x.repeat(3, 1, 1)),
        # 마스크 채널 강제 1채널 (Grayscale)
        Lambdad(keys="label", func=lambda x: x[:1]),
        
        Resized(keys=["image", "label"], spatial_size=image_size, mode=["bilinear", "nearest"]),
        
        # ★★★ 핵심 수정: label도 0~1로 스케일링 (0~255 -> 0~1) ★★★
        ScaleIntensityd(keys=["image", "label"]), 
        
        EnsureTyped(keys=["image", "label"]),
    ])
    
    dataset = Dataset(data_pairs, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    if save_masks and output_mask_folder:
        os.makedirs(output_mask_folder, exist_ok=True)
        
    loss_fn = DiceLoss(sigmoid=True)
    results = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            
            # 예측
            logits = model(x)
            loss = loss_fn(logits, y)
            
            probs = torch.sigmoid(logits)
            
            # 배치 내 개별 처리
            for i in range(len(x)):
                # 텐서 준비
                pred_t = probs[i]
                label_t = y[i]
                
                # Dice Score 계산
                dice = dice_score(pred_t, label_t)
                
                # ==========================================
                # [수정 2] 통계 계산 시 이진화(Binarization) 확실하게 처리
                # ==========================================
                # Threshold 적용 -> 0 또는 1로 변환
                pred_mask_np = (pred_t > threshold).float().cpu().numpy().squeeze()
                label_mask_np = (label_t > 0.5).float().cpu().numpy().squeeze() 
                
                # 픽셀 수 계산 (이제 0과 1만 있으므로 sum이 곧 픽셀 개수)
                pred_pixels = np.sum(pred_mask_np)
                label_pixels = np.sum(label_mask_np)
                total_pixels = pred_mask_np.size
                
                img_path = data_pairs[len(results)]["image"]
                filename = os.path.basename(img_path)
                
                results.append({
                    "filename": filename,
                    "dice_score": dice,
                    "dice_loss": loss.item(),
                    "pred_pixels": int(pred_pixels),
                    "label_pixels": int(label_pixels),
                    "pred_ratio": float(pred_pixels / total_pixels),
                    "label_ratio": float(label_pixels / total_pixels) # 이제 정상 수치(0.xx) 나옴
                })
                
                # 마스크 저장
                if save_masks:
                    save_name = os.path.join(output_mask_folder, f"{Path(filename).stem}_pred.png")
                    # 저장할 때는 다시 255를 곱해서 눈에 보이게 만듦
                    Image.fromarray((pred_mask_np * 255).astype(np.uint8)).save(save_name)

    df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print(f"RESULTS SUMMARY (Total: {len(df)})")
    print(f"Avg Dice Score: {df['dice_score'].mean():.4f}")
    print(f"Avg Label Ratio: {df['label_ratio'].mean():.4f} (Should be 0.0 ~ 1.0)")
    print("="*50 + "\n")
    
    df.to_csv(output_csv, index=False)
    return df

if __name__ == "__main__":
    base_path = "/home/terminaterpanda/UNET"
    
    # 설정
    df = evaluate_zeroshot(
        checkpoint_path=os.path.join(base_path, "checkpoints/unet-transition/best_model.pth"),
        image_folder=os.path.join(base_path, "data/Retina/Data/test/image"),
        mask_folder=os.path.join(base_path, "data/Retina/Data/test/mask"),
        output_csv=os.path.join(base_path, "code/utils/results/retina_eval_transition.csv"),
        output_mask_folder=os.path.join(base_path, "code/utils/results/predicted_masks_transition"),
        batch_size=8,
        save_masks=True
    )