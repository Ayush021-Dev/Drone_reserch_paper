import os
import shutil
import random
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent / 'Drone_Dataset'
DB1_DIR = BASE_DIR / 'Database1'
TRAIN_LIST = BASE_DIR / 'train.txt'
TEST_LIST = BASE_DIR / 'test.txt'
NAMES_FILE = BASE_DIR / 'obj.names'

# Output structure for YOLOv5
YOLOV5_DIR = BASE_DIR / 'yolov5_data'
IMAGES_DIR = YOLOV5_DIR / 'images'
LABELS_DIR = YOLOV5_DIR / 'labels'
SPLITS = ['train', 'val', 'test']

# Create output directories
for split in SPLITS:
    (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
    (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

# Helper to copy image and label
def copy_pair(img_path, split):
    img_name = os.path.basename(img_path)
    label_name = os.path.splitext(img_name)[0] + '.txt'
    src_img = DB1_DIR / img_name
    src_label = DB1_DIR / label_name
    dst_img = IMAGES_DIR / split / img_name
    dst_label = LABELS_DIR / split / label_name
    if src_img.exists():
        shutil.copy2(src_img, dst_img)
    if src_label.exists():
        shutil.copy2(src_label, dst_label)
    else:
        # Create empty label if missing
        open(dst_label, 'w').close()

# Read image lists
with open(TRAIN_LIST) as f:
    train_imgs = [Path(line.strip()).name for line in f if line.strip()]
with open(TEST_LIST) as f:
    test_imgs = [Path(line.strip()).name for line in f if line.strip()]

# Split train into train/val
random.seed(42)
val_count = max(1, int(0.1 * len(train_imgs)))
val_imgs = set(random.sample(train_imgs, val_count))
train_imgs_final = [img for img in train_imgs if img not in val_imgs]

# Copy train/val/test
for img in train_imgs_final:
    copy_pair(img, 'train')
for img in val_imgs:
    copy_pair(img, 'val')
for img in test_imgs:
    copy_pair(img, 'test')

# Read class names
with open(NAMES_FILE) as f:
    names = [line.strip() for line in f if line.strip()]

# Write data.yaml
with open(YOLOV5_DIR / 'data.yaml', 'w') as f:
    f.write(f"train: {IMAGES_DIR / 'train'}\n")
    f.write(f"val: {IMAGES_DIR / 'val'}\n")
    f.write(f"test: {IMAGES_DIR / 'test'}\n")
    f.write(f"\nnc: {len(names)}\n")
    f.write(f"names: {names}\n")

print('Dataset prepared for YOLOv5 in:', YOLOV5_DIR) 