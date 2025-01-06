import numpy as np
import cv2
from ultralytics import YOLO
import torch


def remove_background(image_path):
    # Explicitly set device to CPU
    torch.device('cpu')

    # Load the pre-trained YOLOv8 segmentation model
    model = YOLO("yolov8n-seg.pt")

    # Force model to CPU
    model.model.to('cpu')

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Perform segmentation on CPU
    results = model(image_path, device='cpu', confidence_threshold=0.5)

    # Check if any objects are detected
    if len(results[0].masks) == 0:
        raise ValueError("No objects detected in the image")

    # Get the first mask (assuming single object detection)
    mask = results[0].masks[0].data.cpu().numpy().squeeze()

    # Create a binary mask
    mask_binary = (mask > 0.5).astype(np.uint8)

    # Ensure mask matches image dimensions
    if mask_binary.shape != image.shape[:2]:
        mask_binary = cv2.resize(mask_binary, (image.shape[1], image.shape[0]))

    # Create background-removed image
    background_removed = np.zeros_like(image)
    background_removed[mask_binary == 1] = image[mask_binary == 1]

    return background_removed


# Usage
try:
    result = remove_background("apple3.jpg")
    cv2.imwrite("image_without_background.jpg", result)
    cv2.imshow("Background Removed", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"Error: {e}")