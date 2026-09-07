"""
Automated House Rooms Dataset Loader & Downloader
Downloads the high-quality Kaggle House Rooms dataset (5,000+ real images of
Bathroom, Bedroom, Dining Room, Kitchen, and Living Room) and formats it into
train/val split folders for PyTorch fine-tuning.
"""

import os
import shutil
import random
import kagglehub

# Mapping from Kaggle dataset folder names to our project categories
ROOM_MAP = {
 'Bathroom': 'bathroom',
 'Bedroom': 'bedroom',
 'Dinning': 'dining_room',
 'Kitchen': 'kitchen',
 'Livingroom': 'living_room'
}

def setup_and_download():
 print("=== Downloading House Rooms Dataset from Kaggle ===")

 # 1. Download Kaggle Dataset via kagglehub
 try:
 raw_dataset_path = kagglehub.dataset_download("robinreni/house-rooms-image-dataset")
 print(f"Dataset downloaded to: '{raw_dataset_path}'")
 except Exception as e:
 print(f"!! Error downloading dataset via kagglehub: {e}")
 return

 source_dir = os.path.join(raw_dataset_path, "House_Room_Dataset")
 if not os.path.exists(source_dir):
 source_dir = raw_dataset_path

 target_base = "dataset"
 os.makedirs(os.path.join(target_base, "train"), exist_ok=True)
 os.makedirs(os.path.join(target_base, "val"), exist_ok=True)

 print("\n>> Processing and copying images into train/val splits (80% train, 20% val)...")

 total_copied = 0
 for src_folder, target_room in ROOM_MAP.items():
 src_path = os.path.join(source_dir, src_folder)
 if not os.path.exists(src_path):
 continue

 train_dest = os.path.join(target_base, "train", target_room)
 val_dest = os.path.join(target_base, "val", target_room)
 os.makedirs(train_dest, exist_ok=True)
 os.makedirs(val_dest, exist_ok=True)

 images = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
 random.seed(42)
 random.shuffle(images)

 split_idx = int(len(images) * 0.8)
 train_imgs = images[:split_idx]
 val_imgs = images[split_idx:]

 for img in train_imgs:
 shutil.copy(os.path.join(src_path, img), os.path.join(train_dest, img))
 for img in val_imgs:
 shutil.copy(os.path.join(src_path, img), os.path.join(val_dest, img))

 print(f" [+] {target_room.upper():12s}: {len(train_imgs)} train images, {len(val_imgs)} val images")
 total_copied += len(images)

 print(f"\nSuccess! Processed {total_copied} total room images.")
 print("Dataset is formatted in 'dataset/train/' and 'dataset/val/'")
 print("You can now run: python train_room_model.py")


if __name__ == '__main__':
 setup_and_download()
