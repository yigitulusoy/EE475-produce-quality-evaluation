import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans

################################################################################
# 1. YOLO + Fallback Color/Curvature Segmentation for Background Removal
###############################################################################
"""
def color_curvature_segmentation(image):
    

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for different apple colors
    # Red (two ranges because red wraps around in HSV)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    # Green and yellow ranges
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([35, 255, 255])

    # Create masks for all colors
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine all color masks
    color_mask = cv2.bitwise_or(
        cv2.bitwise_or(mask_red1, mask_red2),
        cv2.bitwise_or(mask_green, mask_yellow)
    )

    # Apply morphological operations to clean up the mask
    kernel = np.ones((7, 7), np.uint8)  # Larger kernel
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    # Additional dilation to capture more of the apple
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)

    # Find contours for curvature analysis
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create final mask
    final_mask = np.zeros_like(color_mask)

    min_area = 500  # Increased minimum area

    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Filter by circularity/roundness
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.4:  # Relaxed circularity threshold
            # Fill the contour and add some padding
            cv2.drawContours(final_mask, [contour], -1, 255, -1)
            # Add padding around the contour
            cv2.drawContours(final_mask, [contour], -1, 255, 5)

    # Final cleanup
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    # Ensure mask is binary (0/255) with correct dimensions
    final_mask = (final_mask > 0).astype(np.uint8) * 255

    return final_mask


def remove_background_batch(input_folder, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)

    # Use CUDA if available
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
            # Load the image
            image = cv2.imread(str(img_path))
            if image is None:
                pbar.write(f"Failed to read {img_path.name}")
                continue

            # Try YOLO first
            results = model(image, conf=0.3)
            result = results[0]  # get first batch result

            if not hasattr(result, 'masks') or result.masks is None or len(result.masks) == 0:
                # YOLO failed => fallback method
                pbar.write(f"YOLO failed for {img_path.name}, using fallback method")
                fallback_mask = color_curvature_segmentation(image)
                mask_binary = fallback_mask // 255
            else:
                # Use YOLO mask from the first detected object
                mask = result.masks[0].data.cpu().numpy().squeeze()
                mask_binary = (mask > 0.5).astype(np.uint8)
                if mask_binary.shape != image.shape[:2]:
                    mask_binary = cv2.resize(mask_binary, (image.shape[1], image.shape[0]))

            # Create background-removed image
            background_removed = image.copy()
            background_removed[mask_binary == 0] = 0

            # Save
            output_path = os.path.join(output_folder, img_path.name)
            cv2.imwrite(output_path, background_removed)

        except Exception as e:
            pbar.write(f"Error processing {img_path.name}: {str(e)}")

    pbar.close()
    print("\nBackground removal completed!")


################################################################################
# 2. Optional Shiny Mask (Not used for classification, but shown for completeness)
################################################################################

def apply_shiny_mask(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in Path(input_folder).glob('*') if f.suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} images to process for shiny mask")

    pbar = tqdm(image_files, desc="Applying shiny masks", unit="image", dynamic_ncols=True)
    for img_path in pbar:
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                pbar.write(f"Failed to read {img_path.name}")
                continue

            # Threshold on brightness channel
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            brightness = hsv_image[:, :, 2]
            _, shiny_mask = cv2.threshold(brightness, 200, 255, cv2.THRESH_BINARY)

            # Invert shiny mask
            shiny_mask_inv = cv2.bitwise_not(shiny_mask)
            masked_image = cv2.bitwise_and(image, image, mask=shiny_mask_inv)

            # Save masked image
            output_path = os.path.join(output_folder, img_path.name)
            cv2.imwrite(output_path, masked_image)

        except Exception as e:
            pbar.write(f"Error processing {img_path.name}: {str(e)}")

    pbar.close()
    print("\nShiny masking completed (not used for classification)!")

"""
################################################################################
# 3. Defect Classification Functions
################################################################################

def check_forbidden_color_whole_apple(image):
    """
    Enhanced check for bad overall color with detailed reasons
    """
    try:
        marked_image = image.copy()
        
        # 1. Convert to multiple color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 2. Define problematic color ranges
        # Brown/tan in HSV - narrower range
        lower_brown_hsv = np.array([10, 50, 40])
        upper_brown_hsv = np.array([25, 255, 140])
        
        # Dark colors in LAB - adjusted threshold
        lower_dark_lab = np.array([0, 128, 128])
        upper_dark_lab = np.array([30, 255, 255])
        
        # Create masks
        brown_mask = cv2.inRange(hsv, lower_brown_hsv, upper_brown_hsv)
        dark_mask = cv2.inRange(lab, lower_dark_lab, upper_dark_lab)
        
        # 3. Calculate color statistics
        total_pixels = np.sum(image[:,:,0] > 0)  # Count only non-background pixels
        if total_pixels == 0:
            return False, image, "No apple detected"
            
        brown_ratio = np.sum(brown_mask > 0) / total_pixels
        dark_ratio = np.sum(dark_mask > 0) / total_pixels
        
        # 4. Check color uniformity
        non_zero_mask = image[:,:,0] > 0
        if np.sum(non_zero_mask) > 0:
            hsv_masked = hsv[non_zero_mask]
            std_h = np.std(hsv_masked[:,0])
            std_s = np.std(hsv_masked[:,1])
            std_v = np.std(hsv_masked[:,2])
            color_variance = (std_h + std_s + std_v) / 3
        else:
            color_variance = 0
        
        # 5. Check if it's a red apple
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_ratio = np.sum(red_mask > 0) / total_pixels
        
        # 6. Decision logic with detailed reasons
        is_bad_color = False
        detailed_reasons = []
        
        if red_ratio > 0.3:  # Red apple
            if brown_ratio > 0.6:
                is_bad_color = True
                detailed_reasons.append(f"severely brown ({brown_ratio:.1%} brown)")
        else:  # Non-red apple
            if brown_ratio > 0.75:
                is_bad_color = True
                detailed_reasons.append(f"too brown ({brown_ratio:.1%} brown)")
            
        if dark_ratio > 2.8:
            is_bad_color = True
            detailed_reasons.append(f"too dark ({dark_ratio:.1%} dark)")
            
        if color_variance < 5 and not red_ratio > 0.3:
            is_bad_color = True
            detailed_reasons.append(f"unusual color pattern (variance: {color_variance:.1f})")
        
        # 7. Add color information to image
        if is_bad_color:
            height, width = image.shape[:2]
            y_offset = 30  # Starting y position for text
            
            # Add apple type
            apple_type = "Red Apple" if red_ratio > 0.3 else "Non-red Apple"
            cv2.putText(marked_image, f"Type: {apple_type}",
                       (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6,
                       (0, 0, 255),
                       2)
            
            # Add each reason on a new line
            for i, reason in enumerate(detailed_reasons):
                cv2.putText(marked_image, reason,
                           (10, y_offset + (i+1)*25),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.6,
                           (0, 0, 255),
                           2)
            
        return is_bad_color, marked_image
        
    except Exception as e:
        print(f"Error in check_forbidden_color_whole_apple: {str(e)}")
        return False, image

def detect_black_spots(image):
    """
    Detect dark/black spots on the apple
    Returns: (bool, image) - Whether black spots were detected and the marked image
    """
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Broader range to catch severe dark spots
        lower_black = np.array([5, 5, 5])
        upper_black = np.array([20, 20, 20])  # Increased to catch darker brown areas
        
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        
        kernel = np.ones((5,5), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        marked_image = image.copy()
        
        # Decreased minimum area to catch obvious defects
        min_area = 100  # Decreased to catch smaller but severe defects
        max_area = 500  # Increased to catch larger defects
        
        has_black_spots = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Relaxed aspect ratio check for obvious defects
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                
                # Check entire apple for severe defects
                if 0.1 < aspect_ratio < 10:  # Relaxed ratio
                    has_black_spots = True
                    cv2.drawContours(marked_image, [contour], -1, (0, 0, 255), 2)
        
        return has_black_spots, marked_image
        
    except Exception as e:
        print(f"Error in detect_black_spots: {str(e)}")
        return False, image

def is_red_apple(image):
    """
    Check if the apple is predominantly red
    Returns: bool - Whether the apple is red
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red color ranges (both lower and upper red hues in HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)
    
    red_pixels = np.sum(red_mask > 0)
    total_pixels = red_mask.size
    red_ratio = red_pixels / total_pixels
    
    return red_ratio > 0.3  # If more than 30% is red

def detect_brown_spots(image):
    """
    Enhanced brown spot detection using color gradients and texture analysis
    """
    try:
        # 1. Convert to multiple color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 2. Create brown color mask
        lower_brown = np.array([0, 30, 20])
        upper_brown = np.array([30, 255, 210])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # 3. Detect color gradients
        # Compute gradients in LAB space
        L = lab[:,:,0]
        gradient_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Threshold for significant color changes
        gradient_mask = gradient_magnitude > np.mean(gradient_magnitude) * 1.5
        gradient_mask = gradient_mask.astype(np.uint8) * 255
        
        # 4. Combine brown color and gradient information
        combined_mask = cv2.bitwise_and(brown_mask, gradient_mask)
        
        # 5. Clean up the mask
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # 6. Find and analyze contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        marked_image = image.copy()
        min_area = 250  # Minimum area for a significant brown spot
        
        has_brown_spots = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Get region of interest
                x, y, w, h = cv2.boundingRect(contour)
                roi = image[y:y+h, x:x+w]
                
                # Skip if in stem area
                if y < image.shape[0] * 0.2:
                    continue
                
                # Calculate texture features in the ROI
                if roi.size > 0:  # Check if ROI is not empty
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    texture_variance = np.var(gray_roi)
                    
                    # If texture is significantly different from surroundings
                    if texture_variance > 100:  # Adjust threshold as needed
                        has_brown_spots = True
                        cv2.drawContours(marked_image, [contour], -1, (0, 0, 255), 2)
                        # Add text near the spot
                        cv2.putText(marked_image, 'defect',
                                  (x, y-5),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5,
                                  (0, 0, 255),
                                  1)
        
        return has_brown_spots, marked_image
        
    except Exception as e:
        print(f"Error in detect_brown_spots: {str(e)}")
        return False, image

def check_overall_darkness(image):
    """
    Check if the apple is too dark overall. 
    Returns: (is_too_dark, marked_image)
      - is_too_dark (bool): True if the apple is classified as too dark
      - marked_image (numpy array): copy of the image with text overlay if too dark
    """
    try:
        marked_image = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 1) If it's a green apple, skip the darkness check
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])
        if green_ratio > 0.3:  # More than 30% green => likely a green apple
            return False, image

        # 2) Get the brightness channel
        value_channel = hsv[:, :, 2]
        avg_brightness = np.mean(value_channel)  # Mean brightness
        total_pixels = value_channel.size

        # 3) Calculate how many pixels are 'dark' below a threshold (e.g., < 50)
        #    The ratio of these dark pixels indicates how uniformly dark the apple is.
        dark_pixels = np.sum(value_channel < 50)
        dark_ratio = dark_pixels / total_pixels

        # 4) Check if it's likely a red apple (≥30% red-hued)
        red_lower = np.array([170, 50, 50])
        red_upper = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        red_ratio = np.sum(red_mask > 0) / total_pixels

        # 5) Combine conditions so we only flag apples that are truly dark overall.
        #    For red apples, use slightly lower brightness threshold and higher dark ratio.
        #    For non-red apples, use a more moderate threshold.
        is_too_dark = False
        if red_ratio > 0.3:
            # Likely a red apple: require average brightness AND majority of pixels to be dark
            if avg_brightness < 40 and dark_ratio > 0.70:
                is_too_dark = True
        else:
            # Not predominantly red
            if avg_brightness < 45 and dark_ratio > 0.65:
                is_too_dark = True

        # 6) If too dark, write text on the image
        if is_too_dark:
            h, w = marked_image.shape[:2]
            cv2.putText(marked_image,
                        'too dark apple :(',
                        (w // 4, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),  # Red color
                        2)

        return is_too_dark, marked_image

    except Exception as e:
        print(f"Error in check_overall_darkness: {str(e)}")
        return False, image

def detect_wrinkles(image):
    """
    Enhanced wrinkle detection with strict size thresholds
    Returns: (bool, marked_image) - Whether wrinkles were detected and marked image
    """
    try:
        marked_image = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding with stricter parameters
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            4
        )
        
        # Edge detection with stricter parameters
        edges = cv2.Canny(blurred, 100, 200)
        
        # Combine binary and edge detection
        combined = cv2.bitwise_and(binary, edges)
        
        # Ignore stem area (top 25% of image)
        height = combined.shape[0]
        combined[:int(height*0.25), :] = 0
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_wrinkles = False
        wrinkle_count = 0
        valid_wrinkles = []
        
        # Calculate apple size for relative thresholds
        apple_area = np.sum(image[:,:,0] > 0)  # Non-black pixels
        min_wrinkle_area = apple_area * 0.001  # Minimum 0.1% of apple size
        max_wrinkle_area = apple_area * 0.015  # Maximum 1.5% of apple size
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip if area is too small or too large relative to apple size
            if area < min_wrinkle_area or area > max_wrinkle_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            
            # Stricter shape criteria for wrinkles
            if (aspect_ratio < 0.15 or aspect_ratio > 6.0) and area > min_wrinkle_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Very strict circularity threshold for wrinkles
                    if circularity < 0.2:  # Wrinkles are very non-circular
                        # Check if the contour is significantly elongated
                        _, (width, height), angle = cv2.minAreaRect(contour)
                        if min(width, height) > 0:
                            elongation = max(width, height) / min(width, height)
                            
                            # Must be very elongated and not too thick
                            if elongation > 4 and min(width, height) < max_wrinkle_area * 0.1:
                                # Skip if it's likely a natural apple feature (like color transition)
                                roi = image[max(0, y-5):min(image.shape[0], y+h+5), 
                                          max(0, x-5):min(image.shape[1], x+w+5)]
                                if roi.size > 0:
                                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                    std_dev = np.std(hsv_roi[:,:,1])  # Check saturation variation
                                    
                                    # Only count as wrinkle if there's significant texture variation
                                    if std_dev > 20:
                                        wrinkle_count += 1
                                        valid_wrinkles.append(contour)
        
        # Consider it wrinkly if enough valid wrinkles are found and they form a pattern
        if wrinkle_count >= 4:
            if len(valid_wrinkles) > 0:
                # Calculate centroid distances
                centroids = []
                for contour in valid_wrinkles:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centroids.append((cx, cy))
                
                # Check if wrinkles are spread out
                if len(centroids) >= 2:
                    distances = []
                    for i in range(len(centroids)):
                        for j in range(i+1, len(centroids)):
                            dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                                         (centroids[i][1] - centroids[j][1])**2)
                            distances.append(dist)
                    
                    avg_distance = np.mean(distances)
                    if avg_distance > apple_area ** 0.5 * 0.15:  # Scale with apple size
                        has_wrinkles = True
                        # Draw only valid wrinkles
                        for contour in valid_wrinkles:
                            cv2.drawContours(marked_image, [contour], -1, (0, 0, 255), 2)
                        
                        cv2.putText(marked_image, f'Wrinkles found: {wrinkle_count}',
                                   (10, image.shape[0] - 40),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.6,
                                   (0, 0, 255),
                                   2)
        
        return has_wrinkles, marked_image
        
    except Exception as e:
        print(f"Error in detect_wrinkles: {str(e)}")
        return False, image

"""
def detect_fungus_infection(image):
  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Higher threshold for white/gray detection
    _, mask_fungus = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)  # Increased from 200 to 220
    
    # Calculate percentage of white pixels
    total_pixels = image.shape[0] * image.shape[1]
    white_pixels = np.sum(mask_fungus == 255)
    white_ratio = white_pixels / total_pixels
    
    return white_ratio > 0.1  # At least 10% must be very bright

"""


################################################################################
# 4. Classification Pipeline
################################################################################

def classify_apple(image):
    """
    Classify an apple as healthy or defected
    Returns: (label, reasons, marked_image)
    """
    reasons = []
    marked_image = image.copy()
    y_offset = 30  # Starting y position for text
    
    # First check if it's a red apple
    is_red = is_red_apple(image)
    apple_type = "Red Apple" if is_red else "Non-red Apple"
    cv2.putText(marked_image, f"Type: {apple_type}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),  # Green color for type
                2)
    y_offset += 25
    
    # Check overall color first
    has_bad_color, color_marked_image = check_forbidden_color_whole_apple(image)
    if has_bad_color:
        reasons.append("Bad overall color")
        marked_image = color_marked_image
        y_offset += 25  # Color info is already added by the function
    
    # Check for black spots
    has_black_spots, black_spots_image = detect_black_spots(image)
    if has_black_spots:
        reasons.append("Black spots detected")
        if len(reasons) == 1:  # If this is the first defect
            marked_image = black_spots_image
        else:
            black_mask = cv2.absdiff(black_spots_image, image)
            marked_image = cv2.add(marked_image, black_mask)
        
        cv2.putText(marked_image, "Black spots found",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),  # Red color
                    2)
        y_offset += 25
    
    # Check for brown spots
    has_brown_spots, brown_spots_image = detect_brown_spots(image)
    if has_brown_spots:
        # For red apples, require larger or more prominent spots
        if not is_red or (is_red and has_significant_defect(brown_spots_image)):
            reasons.append("Brown spots detected")
            if len(reasons) == 1:
                marked_image = brown_spots_image
            else:
                brown_mask = cv2.absdiff(brown_spots_image, image)
                marked_image = cv2.add(marked_image, brown_mask)
            
            cv2.putText(marked_image, "Brown spots found",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)
            y_offset += 25
    
    # Check overall darkness
    is_too_dark, dark_image = check_overall_darkness(image)
    if is_too_dark:
        reasons.append("Apple too dark")
        if len(reasons) == 1:
            marked_image = dark_image
        else:
            dark_mask = cv2.absdiff(dark_image, image)
            marked_image = cv2.add(marked_image, dark_mask)
        
        cv2.putText(marked_image, "Too dark overall",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2)
        y_offset += 25
    
    # Check for wrinkles
    has_wrinkles, wrinkle_image = detect_wrinkles(image)
    if has_wrinkles:
        reasons.append("Wrinkly Apple")
        if len(reasons) == 1:
            marked_image = wrinkle_image
        else:
            wrinkle_mask = cv2.absdiff(wrinkle_image, image)
            marked_image = cv2.add(marked_image, wrinkle_mask)
        
        cv2.putText(marked_image, "Wrinkles detected",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2)
        y_offset += 25

    # Add final classification at the bottom
    height = image.shape[0]
    if len(reasons) > 0:
        cv2.putText(marked_image, "CLASSIFICATION: DEFECTED",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),  # Red
                    2)
        return "Defect", reasons, marked_image
    else:
        cv2.putText(marked_image, "CLASSIFICATION: HEALTHY",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),  # Green
                    2)
        return "Healthy", [], marked_image

def has_significant_defect(marked_image):
    """
    Helper function to check if the detected defects are significant enough
    Returns: bool - Whether the defects are significant
    """
    # Convert to grayscale
    gray = cv2.cvtColor(marked_image, cv2.COLOR_BGR2GRAY)
    
    # Find contours of the markings (green markings from brown spot detection)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check size and density of marked areas
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:  # Larger minimum area for red apples
            return True
    
    return False

def classify_apples(input_folder, output_folder):
    """
    Classify apples as "Defect" or "Healthy" by checking each defect function.
    Saves the original (unmasked) image into output_folder/<label>/ with reason printed.
    Calculates precision and recall using corresponding label files (0=defect, 1=healthy).
    """
    # Prepare output subfolders
    defect_folder = os.path.join(output_folder, "Defect")
    healthy_folder = os.path.join(output_folder, "Healthy")
    os.makedirs(defect_folder, exist_ok=True)
    os.makedirs(healthy_folder, exist_ok=True)

    # Get paths
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in Path(input_folder).glob('*') if f.suffix.lower() in image_extensions]
    
    # Fix: Use absolute path to labels folder
    labels_folder = "Apple_Yolo/train/labels"  # Direct path to labels folder

    # For performance metrics
    TP = 0  # true positive
    FP = 0  # false positive
    TN = 0  # true negative
    FN = 0  # false negative

    pbar = tqdm(image_files, desc="Classifying apples", unit="image", dynamic_ncols=True)
    for img_path in pbar:
        image = cv2.imread(str(img_path))
        if image is None:
            pbar.write(f"Failed to read {img_path.name}")
            continue

        # Get corresponding label file - use original image name without 'nobg_' prefix
        original_name = img_path.name
        if original_name.startswith('nobg_'):
            original_name = original_name[5:]  # Remove 'nobg_' prefix
        
        label_path = os.path.join(labels_folder, f"{Path(original_name).stem}.txt")
        
        try:
            with open(label_path, 'r') as f:
                first_char = f.read(1)
                ground_truth = "Defect" if first_char == '0' else "Healthy"
        except (FileNotFoundError, IOError):
            pbar.write(f"Warning: No label file found for {original_name}")
            continue

        predicted_label, reasons, marked_image = classify_apple(image)

        # Update performance metrics
        if ground_truth == "Defect" and predicted_label == "Defect":
            TP += 1
        elif ground_truth == "Defect" and predicted_label == "Healthy":
            FN += 1
        elif ground_truth == "Healthy" and predicted_label == "Defect":
            FP += 1
        elif ground_truth == "Healthy" and predicted_label == "Healthy":
            TN += 1

        # Save to the appropriate folder
        if predicted_label == "Defect":
            output_path = os.path.join(defect_folder, img_path.name)
            cv2.imwrite(output_path, marked_image)  # Save marked image instead
            if len(reasons) > 0:
                pbar.write(f"{img_path.name} classified as DEFECT because: {', '.join(reasons)}")
        else:
            output_path = os.path.join(healthy_folder, img_path.name)
            cv2.imwrite(output_path, image)  # Save original for healthy apples

    pbar.close()

    # Calculate metrics
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    print("\nClassification results:")
    print(f"  True Positives (Defect predicted as Defect): {TP}")
    print(f"  False Positives (Healthy predicted as Defect): {FP}")
    print(f"  True Negatives (Healthy predicted as Healthy): {TN}")
    print(f"  False Negatives (Defect predicted as Healthy): {FN}")
    print(f"  Precision (Defect class): {precision:.4f}")
    print(f"  Recall (Defect class): {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")


################################################################################
# 5. Example Main
################################################################################

if __name__ == "__main__":
    # Example usage with absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(current_dir, "Apple_Yolo/train/images")
    bg_removed_folder = os.path.join(current_dir, "Background_Removed")
    classification_folder = os.path.join(current_dir, "Classification")

    # 1) Remove background
    #remove_background_batch(input_folder, bg_removed_folder)

    # 2) Classify apples using the background-removed images
    classify_apples(bg_removed_folder, classification_folder)

    print("\nAll steps completed.")
