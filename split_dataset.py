import os
import shutil
import random

# ============================================
# SETTINGS
# ============================================

SOURCE_DIR = "DATASET"

TRAIN_DIR = os.path.join(SOURCE_DIR, "train")
VAL_DIR = os.path.join(SOURCE_DIR, "val")
TEST_DIR = os.path.join(SOURCE_DIR, "test")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

random.seed(42)

# ============================================
# FIND CLASSES
# ============================================

classes = []

for folder in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder)

    if os.path.isdir(folder_path):

        if folder not in ["train", "val", "test"]:
            classes.append(folder)

print("Classes Found:")
print(classes)

# ============================================
# CREATE FOLDERS
# ============================================

for cls in classes:

    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

# ============================================
# SPLIT DATA
# ============================================

for cls in classes:

    print(f"\nProcessing {cls}...")

    source_class = os.path.join(SOURCE_DIR, cls)

    images = [
        f for f in os.listdir(source_class)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Copy Train
    for img in train_images:

        shutil.copy2(
            os.path.join(source_class, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    # Copy Validation
    for img in val_images:

        shutil.copy2(
            os.path.join(source_class, img),
            os.path.join(VAL_DIR, cls, img)
        )

    # Copy Test
    for img in test_images:

        shutil.copy2(
            os.path.join(source_class, img),
            os.path.join(TEST_DIR, cls, img)
        )

    print(f"Total : {total}")
    print(f"Train : {len(train_images)}")
    print(f"Val   : {len(val_images)}")
    print(f"Test  : {len(test_images)}")

print("\n===================================")
print("Dataset Split Completed Successfully!")
print("===================================")