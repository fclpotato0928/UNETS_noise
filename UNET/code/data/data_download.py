import kagglehub
import shutil
from pathlib import Path

src = kagglehub.dataset_download(
    "aryashah2k/breast-ultrasound-images-dataset"
)

sra = kagglehub.dataset_download("abdallahwagih/retina-blood-vessel")

dst = Path("/home/terminaterpanda/UNET/data/breast_ultrasound")
dsa = Path("/home/terminaterpanda/UNET/data/Retina")

dst.mkdir(parents=True, exist_ok=True)
dsa.mkdir(parents=True, exist_ok=True)
shutil.copytree(src, dst, dirs_exist_ok=True)
shutil.copytree(sra, dsa, dirs_exist_ok=True)
print("Copied to:", dst)
print("Copied to:", dsa)