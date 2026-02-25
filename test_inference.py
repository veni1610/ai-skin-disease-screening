import os
from app.inference import predict

test_folder = "dataset/processed/test/acne"

# pick first image automatically
image_name = os.listdir(test_folder)[0]
image_path = os.path.join(test_folder, image_name)

result = predict(image_path)
print(result)