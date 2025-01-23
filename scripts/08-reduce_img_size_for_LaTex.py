"""This script was used to reduce the plot size so that it does not take forever to compile on OverLeaf"""
from pathlib import Path
from PIL import Image

from AI_representation_bias_in_farming import utils

megadata = utils.read_megadata()

source_folder = "/Users/skysheng/Library/CloudStorage/OneDrive-UBC/University of British Columbia/Research/PhD Project/AI representation bias/AI_bias_in_farming/results/plots"

source_path = Path(source_folder)
dest_folder = "/Users/skysheng/Downloads/resize"

for source_file in source_path.glob("*.png"):
    img = Image.open(source_file)
    # Convert RGBA to RGB
    if img.mode == "RGBA":
        img = img.convert("RGB")
    new_size = tuple(int(dim * 0.7) for dim in img.size)
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

    dest_file = Path(dest_folder) / source_file.name.replace(".png", ".jpg")
    img_resized.save(dest_file, format="JPEG", quality=100, optimize=True)
