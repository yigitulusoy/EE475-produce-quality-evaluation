import os
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np

def remove_background_batch(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load YOLO model
    model = YOLO('yolov8n-seg.pt')
    
    # Get all image files in input folder
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_file in tqdm(image_files):
        # Read image
        img_path = os.path.join(input_folder, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Could not read image: {img_path}")
            continue
            
        # Run inference
        results = model(image, conf=0.25)
        
        # Check if we got any results
        if not results or len(results) == 0:
            print(f"No detection results for {img_file}")
            continue
            
        result = results[0]  # Get first result
        
        # Check if we have any masks
        if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
            # Get the mask for the first detected object
            mask = result.masks[0].data.cpu().numpy()[0]
            
            # Resize mask to match image size
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Create 3-channel mask
            mask_3channel = np.stack([mask] * 3, axis=-1)
            
            # Apply mask to image
            masked_image = (image * mask_3channel).astype(np.uint8)
            
            # Save the masked image
            output_path = os.path.join(output_folder, img_file)
            cv2.imwrite(output_path, masked_image)
        else:
            print(f"No segmentation masks found in {img_file}")

def detect_black_spots(image):
    """
    Detect black spots, bruises, and color variations using spatial and frequency analysis
    """
    try:
        # Original detection code
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Frequency domain analysis
        # Convert to float and apply FFT
        f_transform = np.fft.fft2(gray.astype(float))
        f_shift = np.fft.fftshift(f_transform)
        
        # Create high-pass filter for edge detection
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        mask = np.ones((rows, cols), np.uint8)
        center = 30
        mask[crow-center:crow+center, ccol-center:ccol+center] = 0
        
        # Apply high-pass filter and inverse FFT
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize and threshold the frequency domain result
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, freq_mask = cv2.threshold(img_back, 30, 255, cv2.THRESH_BINARY)
        
        # Original spatial domain detection
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        hsv_mask = cv2.inRange(hsv, lower_black, upper_black)
        _, gray_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(hsv_mask, gray_mask), freq_mask)
        
        # Noise removal and enhancement
        kernel = np.ones((3,3), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marked_image = image.copy()
        
        # Add color variation detection
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        
        # Calculate local standard deviation for color variation
        kernel_size = 15
        l_std = cv2.blur(l_channel, (kernel_size, kernel_size))
        a_std = cv2.blur(a_channel, (kernel_size, kernel_size))
        b_std = cv2.blur(b_channel, (kernel_size, kernel_size))
        
        # Combine color variations
        color_variation = cv2.addWeighted(
            cv2.addWeighted(l_std, 0.5, a_std, 0.25, 0),
            0.5,
            b_std,
            0.25,
            0
        )
        
        # Threshold for significant color variations
        _, variation_mask = cv2.threshold(color_variation, 20, 255, cv2.THRESH_BINARY)
        
        # Find contours in variation mask
        variation_contours, _ = cv2.findContours(
            variation_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Process both black spots and color variations
        min_area = 10
        max_area = 500
        spot_count = 0
        valid_spots = []
        
        # Original black spot detection (kept from previous version)
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if 0.3 < circularity < 1.0:
                    spot_count += 1
                    valid_spots.append(contour)
                    # Draw contour in red for black spots
                    cv2.drawContours(marked_image, [contour], -1, (0, 0, 255), 2)
                    
                    # Add frequency domain detection highlight in blue
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    freq_overlap = cv2.bitwise_and(freq_mask, mask)
                    if np.sum(freq_overlap) > 0:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(marked_image, (cx, cy), 15, (255, 0, 0), 2)
        
        # Add bruise/color variation detection
        for contour in variation_contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Larger minimum area for bruises
                # Draw contour in yellow for bruises/color variations
                cv2.drawContours(marked_image, [contour], -1, (0, 255, 255), 2)
                spot_count += 1
                valid_spots.append(contour)
        
        # If we have either significant color variations or enough black spots
        if spot_count >= 2:
            return True, marked_image
            
        return False, image
        
    except Exception as e:
        print(f"Error in detect_black_spots: {str(e)}")
        return False, image

def detect_greening(image):
    """
    Detect citrus greening disease by checking for significant green coloration
    """
    try:
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define green color range
        lower_green = np.array([23, 5, 3])  # Green hue range
        upper_green = np.array([150, 255, 255])
        
        # Create mask for green pixels
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate percentage of green pixels
        total_pixels = image.shape[0] * image.shape[1]
        green_pixels = np.sum(green_mask > 0)
        green_ratio = green_pixels / total_pixels
        
        # Mark as greening if more than 60% of the orange is green
        has_greening = green_ratio > 0.02
        
        # Mark affected areas on the image if greening is detected
        marked_image = image.copy()
        if has_greening:
            # Create a green overlay for affected areas
            green_areas = cv2.bitwise_and(image, image, mask=green_mask)
            marked_image = cv2.addWeighted(marked_image, 1, green_areas, 0.5, 0)
        
        return has_greening, marked_image
        
    except Exception as e:
        print(f"Error in detect_greening: {str(e)}")
        return False, image

def detect_brown_spots(image):
    """
    Detect brown spots on oranges
    """
    try:
        # Convert to multiple color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Brown color range in HSV
        lower_brown = np.array([10, 10, 10])
        upper_brown = np.array([60, 80, 100])
        
        # Create brown mask
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Additional check in LAB space for brown
        l, a, b = cv2.split(lab)
        _, a_thresh = cv2.threshold(a, 128, 255, cv2.THRESH_BINARY)
        _, b_thresh = cv2.threshold(b, 128, 255, cv2.THRESH_BINARY)
        
        # Combine masks
        combined_mask = cv2.bitwise_and(brown_mask, cv2.bitwise_and(a_thresh, b_thresh))
        
        # Clean up mask
        kernel = np.ones((3,3), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marked_image = image.copy()
        
        min_area = 20  # Minimum area for brown spots
        max_area = 1000  # Maximum area for brown spots
        brown_spot_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Draw brown spots in brown color
                cv2.drawContours(marked_image, [contour], -1, (42, 42, 165), 2)  # BGR brown
                brown_spot_count += 1
        
        return brown_spot_count >= 2, marked_image
        
    except Exception as e:
        print(f"Error in detect_brown_spots: {str(e)}")
        return False, image

def classify_orange(image_path, output_base):
    """
    Classify an orange image and save it to appropriate folder
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        
        # Check for greening first
        has_greening, marked_greening = detect_greening(image)
        
        # Only check for other diseases if no greening is detected
        if has_greening:
            output_folder = os.path.join(output_base, 'greening_classed')
            output_image = marked_greening
        else:
            # Check for brown spots
            has_brown_spots, marked_brown = detect_brown_spots(image)
            
            if has_brown_spots:
                output_folder = os.path.join(output_base, 'brownspot_classed')
                output_image = marked_brown
            else:
                output_folder = os.path.join(output_base, 'fresh_classed')
                output_image = image
        
        # Save image
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, output_image)
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def process_train_folders():
    # Base paths
    train_folder = "train"  # Adjust if your train folder is in a different location
    output_folder = "orange_no_bg"  # Single output folder for all processed images
    
    # Create the output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all subfolders in train directory
    train_subfolders = [f for f in Path(train_folder).iterdir() if f.is_dir()]
    
    print(f"Found {len(train_subfolders)} subfolders in train directory")
    
    # Process each subfolder
    for subfolder in train_subfolders:
        print(f"\nProcessing folder: {subfolder.name}")
        
        # Process images in this subfolder
        remove_background_batch(str(subfolder), output_folder)

def main():
    # First, remove backgrounds
    process_train_folders()
    
    # Then classify the background-removed images
    print("\nClassifying oranges...")
    output_base = 'classified_oranges'
    os.makedirs(output_base, exist_ok=True)
    
    # Get all images from the background-removed folder
    input_folder = "orange_no_bg"
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to classify")
    
    # Process each image
    for img_file in tqdm(image_files):
        img_path = os.path.join(input_folder, img_file)
        classify_orange(img_path, output_base)

if __name__ == "__main__":
    main()
