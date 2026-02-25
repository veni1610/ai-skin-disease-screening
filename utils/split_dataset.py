import os
import shutil
import random

raw_dir = "dataset/raw"
processed_dir = "dataset/processed"

split_ratio = (0.7, 0.15, 0.15)

for class_name in os.listdir(raw_dir):

    class_path = os.path.join(raw_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    train_split = int(split_ratio[0] * len(images))
    val_split = int(split_ratio[1] * len(images))

    train_imgs = images[:train_split]
    val_imgs = images[train_split:train_split + val_split]
    test_imgs = images[train_split + val_split:]

    for split, split_imgs in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):
        split_folder = os.path.join(processed_dir, split, class_name)
        os.makedirs(split_folder, exist_ok=True)

        for img in split_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_folder, img)
            shutil.copy(src, dst)

print("Dataset split completed successfully.")