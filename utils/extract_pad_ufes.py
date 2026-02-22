import os
import shutil
import pandas as pd

PAD_UFES_PATH = r"C:\Users\vysha\Downloads\pad_ufes"  # Main folder
CSV_PATH = os.path.join(PAD_UFES_PATH, "metadata.csv")

DEST_ROOT = "dataset/raw"

# Map PAD-UFES labels to your folder names
label_map = {
    "MEL": "melanoma",
    "BCC": "bcc",
    "SEK": "seborrheic_keratosis"
}

# Load metadata
df = pd.read_csv(CSV_PATH)
print(df.head())
print(df.columns)

for code, folder_name in label_map.items():

    filtered = df[df["diagnostic"] == code]

    print(f"Extracting {folder_name}: {len(filtered)} images")

    for _, row in filtered.iterrows():

        img_name = row["img_id"] 

        # Search in all 3 folders
        for part in ["imgs_part_1", "imgs_part_2", "imgs_part_3"]:

            part_path = os.path.join(PAD_UFES_PATH, part)

            for root, dirs, files in os.walk(part_path):
                for file in files:

                    # match file name without extension
                    if file.startswith(img_name):

                        source_path = os.path.join(root, file)
                        dest_path = os.path.join(DEST_ROOT, folder_name, file)

                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.copy(source_path, dest_path)
                        break

print("PAD-UFES extraction complete.")