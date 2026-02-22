import os

ROOT = "dataset/raw"

print("Class-wise image count:\n")

for class_name in os.listdir(ROOT):
    class_path = os.path.join(ROOT, class_name)
    
    if os.path.isdir(class_path):
        count = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        
        print(f"{class_name}: {count}")