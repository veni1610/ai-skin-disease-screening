import os

train_files = set()
test_files = set()

for root, dirs, files in os.walk("dataset/processed/train"):
    for f in files:
        train_files.add(f)

for root, dirs, files in os.walk("dataset/processed/test"):
    for f in files:
        test_files.add(f)

duplicates = train_files.intersection(test_files)

print("Duplicate files:", len(duplicates))
