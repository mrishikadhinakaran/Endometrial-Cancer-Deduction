import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Define paths
DATASET_PATH = Path("data/histopathological image dataset for ET")
PROCESSED_PATH = Path("processed_data")

# Ensure processed data folder exists
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# Define target size for resizing
IMG_SIZE = (224, 224)

# Supported image extensions
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Find all image files
image_paths = [p for p in DATASET_PATH.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]

if not image_paths:
    print("⚠️ No images found! Check your dataset path.")
    exit()

print(f"Found {len(image_paths)} images! Processing...")

# Process images
for img_path in tqdm(image_paths, desc="🔄 Processing Images"):
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Skipping corrupted file: {img_path}")
            continue
        
        # Resize image
        img_resized = cv2.resize(img, IMG_SIZE)
        img_normalized = img_resized.astype(np.float32) / 255.0
        rel_path = img_path.relative_to(DATASET_PATH)
        save_path = PROCESSED_PATH / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), (img_normalized * 255).astype(np.uint8))

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

print("Image preprocessing complete! Processed images saved in `processed_data/`")
