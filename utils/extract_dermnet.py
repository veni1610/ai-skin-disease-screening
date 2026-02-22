import os
import shutil

DERMNET_PATH = r"C:\Users\vysha\Downloads\Dermnet"

DEST_ROOT = "dataset/raw"

classes_to_extract = {
    "Acne and Rosacea Photos": "acne",
    "Psoriasis pictures Lichen Planus and related diseases": "psoriasis",
    "Eczema Photos": "eczema",
    "Melanoma Skin Cancer Nevi and Moles": "melanoma"
}

for split in ["train", "test"]:
    split_path = os.path.join(DERMNET_PATH, split)

    for original_class, new_name in classes_to_extract.items():

        source_class_path = os.path.join(split_path, original_class)

        if os.path.exists(source_class_path):

            for img in os.listdir(source_class_path):

                source_img = os.path.join(source_class_path, img)
                dest_img = os.path.join(DEST_ROOT, new_name, img)

                os.makedirs(os.path.dirname(dest_img), exist_ok=True)
                shutil.copy(source_img, dest_img)

print("DermNet extraction complete.")