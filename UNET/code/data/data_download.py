import kagglehub
import shutil
from pathlib import Path

# 현재 파일 위치: UNET/code/data/data_download.py
current_file = Path(__file__).resolve()

# 프로젝트 루트: UNET
project_root = current_file.parent.parent.parent

# UNET/data 폴더 생성
data_root = project_root / "data"
data_root.mkdir(parents=True, exist_ok=True)

print("Project root:", project_root)
print("Data directory:", data_root)

# 다운로드
src = kagglehub.dataset_download(
    "aryashah2k/breast-ultrasound-images-dataset"
)

sra = kagglehub.dataset_download(
    "abdallahwagih/retina-blood-vessel"
)

# 목적지 경로
dst = data_root / "breast_ultrasound"
dsa = data_root / "Retina"

dst.mkdir(parents=True, exist_ok=True)
dsa.mkdir(parents=True, exist_ok=True)

# 복사
shutil.copytree(src, dst, dirs_exist_ok=True)
shutil.copytree(sra, dsa, dirs_exist_ok=True)

print("Copied to:", dst)
print("Copied to:", dsa)
