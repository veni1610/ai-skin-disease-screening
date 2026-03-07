import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(model, image_tensor, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer])

    #generate CAM
    grayscale_cam = cam(input_tensor=image_tensor)[0]

    #convert tensor -> numpy image
    img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    #normalize safely
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    #img = (img - img.min()) / (img.max() - img.min())

    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    return cam_image