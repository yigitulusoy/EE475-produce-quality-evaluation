import numpy as np
import cv2
from ultralytics import YOLO
import torch
import os
from pathlib import Path
from tqdm import tqdm

def remove_background_batch(input_folder, output_folder):
    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the pre-trained YOLOv8 segmentation model
    model = YOLO("yolov8n-seg.pt")
    model.to(device)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in Path(input_folder).glob('*') if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to process")

    successful = 0
    failed = 0
    skipped = 0
    
    # Initialize tqdm progress bar
    pbar = tqdm(image_files, desc="Processing images", unit="image", dynamic_ncols=True)

    for img_path in pbar:
        try:
            # Load original image
            image = cv2.imread(str(img_path))
            if image is None:
                failed += 1
                # Use pbar.write to log messages without disrupting the single-line progress bar
                pbar.write(f"Failed to read {img_path.name}")
                pbar.set_postfix(success=successful, skipped=skipped, failed=failed)
                continue

            # Perform inference on the image
            result = model(image, conf=0.3)[0]  # single image result

            # Check if result has masks
            if not hasattr(result, 'masks') or result.masks is None:
                skipped += 1
                pbar.set_postfix(success=successful, skipped=skipped, failed=failed)
                continue

            # Check if any objects are detected
            if len(result.masks) == 0:
                skipped += 1
                pbar.set_postfix(success=successful, skipped=skipped, failed=failed)
                continue

            # Get the first mask (you can modify this logic if you want different behavior)
            mask = result.masks[0].data.cpu().numpy().squeeze()

            # Create a binary mask
            mask_binary = (mask > 0.5).astype(np.uint8)

            # Resize mask to match image dimensions if needed
            if mask_binary.shape != image.shape[:2]:
                mask_binary = cv2.resize(mask_binary, (image.shape[1], image.shape[0]))

            # Create background-removed image
            background_removed = np.zeros_like(image)
            background_removed[mask_binary == 1] = image[mask_binary == 1]

            # Save the result
            output_path = os.path.join(output_folder, f"nobg_{img_path.name}")
            cv2.imwrite(output_path, background_removed)
            successful += 1

        except Exception as e:
            failed += 1
            # Log error without breaking the progress bar
            pbar.write(f"Error processing {img_path.name}: {str(e)}")

        # Update the postfix on the progress bar each iteration
        pbar.set_postfix(success=successful, skipped=skipped, failed=failed)

    # Close the progress bar
    pbar.close()

    # Print final statistics
    print("\nProcessing Complete!")
    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Skipped (no detection): {skipped}")
    print(f"Failed: {failed}")


# Usage
if __name__ == "__main__":
    input_folder = "Apple_Yolo/train/images"
    output_folder = "output_images"

    try:
        remove_background_batch(input_folder, output_folder)
        print("Batch processing completed!")
    except Exception as e:
        print(f"Error during batch processing: {e}")
