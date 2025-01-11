import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def remove_background_batch(input_folder, output_folder, debug_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the pre-trained YOLOv8 segmentation model
    model = YOLO("yolov8n-seg.pt")
    model.to(device)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in Path(input_folder).glob('*') if f.suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} images to process for background removal")

    pbar = tqdm(image_files, desc="Removing background", unit="image", dynamic_ncols=True)
    for img_path in pbar:
        try:
            # Load and preprocess the image
            image = cv2.imread(str(img_path))
            if image is None:
                pbar.write(f"Failed to read {img_path.name}")
                continue

            # Try multiple confidence thresholds
            confidence_thresholds = [0.3, 0.2, 0.1, 0.05]
            mask_binary = None
            
            for conf in confidence_thresholds:
                results = model(image, conf=conf)
                result = results[0]

                if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                    # Get all masks
                    all_masks = []
                    for mask in result.masks:
                        mask_data = mask.data.cpu().numpy().squeeze()
                        if mask_data.shape != image.shape[:2]:
                            mask_data = cv2.resize(mask_data, (image.shape[1], image.shape[0]))
                        all_masks.append((mask_data > 0.5).astype(np.uint8))
                    
                    # If multiple masks found, combine them or select the largest one
                    if all_masks:
                        if len(all_masks) > 1:
                            # Find the largest mask
                            areas = [np.sum(mask) for mask in all_masks]
                            mask_binary = all_masks[np.argmax(areas)]
                        else:
                            mask_binary = all_masks[0]
                        break

            if mask_binary is None:
                pbar.write(f"YOLO failed for {img_path.name} - trying preprocessing")
                # Additional preprocessing as fallback
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                mask_binary = thresh // 255

            # Create background-removed image
            background_removed = image.copy()
            background_removed[mask_binary == 0] = 0

            # Save background removed image
            output_path = os.path.join(output_folder, img_path.name)
            cv2.imwrite(output_path, background_removed)

            # Create debug visualization
            debug_img = image.copy()
            debug_img[mask_binary == 0] = [0, 0, 255]
            debug_path = os.path.join(debug_folder, f"debug_{img_path.name}")
            cv2.imwrite(debug_path, debug_img)

        except Exception as e:
            pbar.write(f"Error processing {img_path.name}: {str(e)}")

    pbar.close()
    print("\nBackground removal completed!")

def load_ground_truth_from_txt(txt_path):
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    pepper_filenames = []
    ground_truth = []
    for line in lines:
        if line.strip():  # Skip empty lines
            try:
                filename, severity = line.strip().split()
                pepper_filenames.append(filename)
                ground_truth.append(int(severity))
            except ValueError as e:
                print(f"Error parsing line: {line.strip()}")
                continue
    
    return pepper_filenames, ground_truth

def evaluate_curvature(image_path, ground_truth=None):
    # Read the image in color
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None

    # Create mask from the background-removed image (where background is red)
    mask = cv2.inRange(img, np.array([1, 1, 1]), np.array([255, 255, 254]))
    
    # Clean up the mask more aggressively
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours with all points
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        print(f"No contours found in: {image_path}")
        return None

    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Smooth the contour
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Further smooth the points using a moving average
    points = approx.reshape(-1, 2)
    smoothed = np.zeros_like(points, dtype=np.float32)
    window = 3
    
    for i in range(len(points)):
        neighbors = [points[(i + j) % len(points)] for j in range(-window, window + 1)]
        smoothed[i] = np.mean(neighbors, axis=0)
    
    smoothed = smoothed.astype(np.int32)
    
    # Draw debug visualization
    debug_img = img.copy()
    cv2.drawContours(debug_img, [smoothed.reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
    debug_path = image_path.replace('.jpg', '_contour.jpg')
    cv2.imwrite(debug_path, debug_img)

    # Get the principal axis (longest direction) of the pepper
    (x, y), (MA, ma), angle = cv2.fitEllipse(smoothed)
    
    # Divide contour into segments
    num_segments = 10
    segment_points = np.array_split(smoothed, num_segments)
    
    # Calculate direction changes between segments
    direction_changes = []
    curved_length = 0
    total_length = cv2.arcLength(smoothed, True)
    
    for i in range(len(segment_points)):
        # Get average direction of current segment
        seg_current = segment_points[i]
        if len(seg_current) < 2:
            continue
            
        dir_current = seg_current[-1] - seg_current[0]
        
        # Get average direction of next segment
        next_idx = (i + 1) % len(segment_points)
        seg_next = segment_points[next_idx]
        if len(seg_next) < 2:
            continue
            
        dir_next = seg_next[-1] - seg_next[0]
        
        # Calculate angle between directions
        cos_angle = np.dot(dir_current, dir_next) / (np.linalg.norm(dir_current) * np.linalg.norm(dir_next))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        direction_changes.append(angle)
        
        # If segment shows significant curvature, add its length to curved_length
        if angle > 0.5:  # increased threshold to ~28.6 degrees
            curved_length += cv2.arcLength(seg_current, False)
    
    # Calculate curvature metrics
    avg_direction_change = np.mean(direction_changes) if direction_changes else 0
    max_direction_change = np.max(direction_changes) if direction_changes else 0
    curve_ratio = curved_length / total_length if total_length > 0 else 0
    
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Avg Direction Change: {avg_direction_change:.3f}")
    print(f"Max Direction Change: {max_direction_change:.3f}")
    print(f"Curve Ratio: {curve_ratio:.3f}")

    # Classification based on overall shape
    if curve_ratio < 0.75 and avg_direction_change < 0.685:  # Mostly straight
        predicted = 1
    elif curve_ratio < 0.83 and avg_direction_change < 0.75:  # Medium curve
        predicted = 2
    else:  # High curve
        predicted = 3
    
    print(f"Predicted class: {predicted}, Ground Truth: {ground_truth}")
    return predicted

def process_dataset_with_txt(image_folder, txt_path):
    pepper_filenames, ground_truth = load_ground_truth_from_txt(txt_path)
    
    if not pepper_filenames:
        print("No filenames loaded from text file")
        return 1.0, []

    predictions = []
    total_images = len(pepper_filenames)
    error_count = 0
    
    # Initialize counters for matches
    matches = {1: 0, 2: 0, 3: 0}  # Correct predictions for each class
    totals = {1: 0, 2: 0, 3: 0}   # Total ground truth for each class

    for i, filename in enumerate(pepper_filenames):
        if not filename.startswith('resized_'):
            filename = 'resized_' + filename
        if not filename.lower().endswith('.jpg'):
            filename = filename + '.jpg'
            
        image_path = os.path.join(image_folder, filename)
        
        # Count total instances of each class
        true_class = ground_truth[i]
        totals[true_class] = totals.get(true_class, 0) + 1
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            predictions.append(None)
            error_count += 1  # Count missing images as errors
            continue

        severity = evaluate_curvature(image_path, ground_truth[i])
        predictions.append(severity)
        
        if severity is None:
            error_count += 1  # Count failed predictions as errors
        elif severity != ground_truth[i]:
            error_count += 1  # Count mismatches as errors
            print(f"Mismatch for {filename}: Predicted {severity}, Actual {ground_truth[i]}")
        else:
            # Count correct predictions for each class
            matches[severity] = matches.get(severity, 0) + 1

    # Calculate error rate
    error_rate = error_count / total_images
    
    # Print detailed results
    print("\nDetailed Results:")
    print(f"Total images: {total_images}")
    print(f"Total errors: {error_count}")
    print(f"Error breakdown:")
    print(f"  - Missing or failed predictions: {predictions.count(None)}")
    print(f"  - Wrong predictions: {error_count - predictions.count(None)}")
    
    if any(p is not None for p in predictions):
        valid_predictions = [p for p in predictions if p is not None]
        print("Prediction distribution:", np.bincount(valid_predictions)[1:])
    
    # Print matching statistics
    print("\nMatching Statistics:")
    for class_num in [1, 2, 3]:
        total = totals[class_num]
        matched = matches[class_num]
        if total > 0:
            accuracy = (matched / total) * 100
            print(f"Class {class_num}: {matched}/{total} correct ({accuracy:.1f}%)")
    
    return error_rate, predictions

def organize_by_prediction(image_folder, predictions, pepper_filenames):
    # Create folders for each class
    class_folders = {
        1: os.path.join(image_folder, "1_straight"),
        2: os.path.join(image_folder, "2_medium"),
        3: os.path.join(image_folder, "3_curved")
    }
    
    # Create the folders if they don't exist
    for folder in class_folders.values():
        os.makedirs(folder, exist_ok=True)

    # Move files to appropriate folders
    for filename, prediction in zip(pepper_filenames, predictions):
        if prediction is None:
            continue
            
        if not filename.startswith('resized_'):
            filename = 'resized_' + filename
        if not filename.lower().endswith('.jpg'):
            filename = filename + '.jpg'
        
        # Source file path
        src_path = os.path.join(image_folder, filename)
        if not os.path.exists(src_path):
            print(f"Source file not found: {src_path}")
            continue
            
        # Destination file path
        dst_path = os.path.join(class_folders[prediction], filename)
        
        try:
            # Copy instead of move to preserve original files
            import shutil
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {filename}: {str(e)}")

# Example usage
input_folder = r"C:\sonaama\peppers"
output_folder = r"C:\sonaama\peppers_no_bg"
debug_folder = r"C:\sonaama\background_removed_peppers"

# First remove backgrounds and create debug images
remove_background_batch(input_folder, output_folder, debug_folder)

# Then process the dataset
image_folder = output_folder
txt_path = r"C:\sonaama\biber_puanlama.txt"

error_rate, predictions = process_dataset_with_txt(image_folder, txt_path)
print(f"\nError rate: {error_rate * 100:.2f}%")
print("Predictions:", predictions)

# Get filenames from ground truth file again
pepper_filenames, _ = load_ground_truth_from_txt(txt_path)

# Organize images into class folders
organize_by_prediction(image_folder, predictions, pepper_filenames)
