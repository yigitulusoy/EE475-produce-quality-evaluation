import os
import cv2
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from shutil import copy2
from tqdm import tqdm
import torch
from ultralytics import YOLO
from pathlib import Path

################################################################################
# IMAGE-PROCESSING FUNCTIONS (From Your Second Code)
################################################################################

def color_curvature_segmentation(image):
    """
    Fallback segmentation approach if YOLO fails.
    Returns a binary mask of the apple in the image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for different apple colors
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([35, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    color_mask = cv2.bitwise_or(
        cv2.bitwise_or(mask_red1, mask_red2),
        cv2.bitwise_or(mask_green, mask_yellow)
    )

    kernel = np.ones((7, 7), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_mask = np.zeros_like(color_mask)
    min_area = 500

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.4:
            cv2.drawContours(final_mask, [contour], -1, 255, -1)
            # Add a bit of padding
            cv2.drawContours(final_mask, [contour], -1, 255, 5)

    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = (final_mask > 0).astype(np.uint8) * 255

    return final_mask


def remove_background_batch(input_folder, output_folder):
    """
    Removes background for all images in `input_folder`, saving results to `output_folder`.
    Uses YOLOv8 segmentation, and if that fails, uses the fallback color/curvature method.
    """
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = YOLO("yolov8n-seg.pt")
    model.to(device)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in Path(input_folder).glob('*') if f.suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} images to process for background removal")
    for img_path in tqdm(image_files, desc="Removing background", unit="image", dynamic_ncols=True):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Failed to read {img_path.name}")
                continue

            results = model(image, conf=0.3)
            result = results[0]

            # If YOLO fails => fallback
            if not hasattr(result, 'masks') or result.masks is None or len(result.masks) == 0:
                print(f"YOLO failed for {img_path.name}, using fallback method")
                fallback_mask = color_curvature_segmentation(image)
                mask_binary = fallback_mask // 255
            else:
                mask = result.masks[0].data.cpu().numpy().squeeze()
                mask_binary = (mask > 0.5).astype(np.uint8)
                if mask_binary.shape != image.shape[:2]:
                    mask_binary = cv2.resize(mask_binary, (image.shape[1], image.shape[0]))

            background_removed = image.copy()
            background_removed[mask_binary == 0] = 0

            output_path = os.path.join(output_folder, img_path.name)
            cv2.imwrite(output_path, background_removed)

        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")

    print("\nBackground removal completed!")


def check_if_green_apple(hsv, apple_mask):
    """
    Helper function to identify green apples
    """
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_mask = cv2.bitwise_and(green_mask, apple_mask)
    if np.sum(apple_mask > 0) == 0:
        return False
    green_ratio = np.sum(green_mask > 0) / np.sum(apple_mask > 0)
    return green_ratio > 0.3


def check_forbidden_color_whole_apple(image):
    """
    Checks for very dark or brown coloration that suggests a defect.
    """
    try:
        marked_image = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        apple_mask = (image[:,:,0] > 0).astype(np.uint8) * 255
        is_green = check_if_green_apple(hsv, apple_mask)

        lower_brown_hsv = np.array([8, 50, 30])
        upper_brown_hsv = np.array([25, 255, 180])

        if is_green:
            lower_dark_lab = np.array([0, 127, 127])
            upper_dark_lab = np.array([40, 255, 255])
            dark_ratio_threshold = 0.15
        else:
            lower_dark_lab = np.array([0, 127, 127])
            upper_dark_lab = np.array([60, 255, 255])
            dark_ratio_threshold = 0.35

        brown_mask = cv2.bitwise_and(cv2.inRange(hsv, lower_brown_hsv, upper_brown_hsv), apple_mask)
        dark_mask = cv2.bitwise_and(cv2.inRange(lab, lower_dark_lab, upper_dark_lab), apple_mask)

        mask = cv2.bitwise_or(brown_mask, dark_mask)
        total_apple_pixels = np.sum(apple_mask > 0)
        if total_apple_pixels == 0:
            return False, image, mask

        brown_ratio = np.sum(brown_mask > 0) / total_apple_pixels
        dark_ratio = np.sum(dark_mask > 0) / total_apple_pixels

        is_bad_color = False
        detailed_reasons = []

        if brown_ratio > 0.60:
            is_bad_color = True
            detailed_reasons.append(f"severe browning ({brown_ratio:.1%} brown)")

        if dark_ratio > dark_ratio_threshold:
            is_bad_color = True
            detailed_reasons.append(f"too dark ({dark_ratio:.1%} dark)")

        hsv_apple = hsv[apple_mask > 0]
        if hsv_apple.size > 0:
            avg_value = np.mean(hsv_apple[:,2])
            brightness_threshold = 65 if is_green else 75
            if avg_value < brightness_threshold:
                is_bad_color = True
                detailed_reasons.append(f"severe discoloration (brightness: {avg_value:.1f})")

        if is_bad_color:
            for i, reason in enumerate(detailed_reasons):
                cv2.putText(marked_image, reason,
                            (10, 30 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2)

        return is_bad_color, marked_image, mask

    except Exception as e:
        print(f"Error in check_forbidden_color_whole_apple: {str(e)}")
        empty_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        return False, image, empty_mask


def detect_black_spots(image):
    """
    Looks for black or dark brown spots on the apple.
    """
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        apple_mask = (image[:,:,0] > 0).astype(np.uint8) * 255

        lower_black_hsv = np.array([0, 0, 0])
        upper_black_hsv = np.array([180, 100, 60])
        lower_brown_hsv = np.array([0, 20, 20])
        upper_brown_hsv = np.array([30, 200, 80])

        black_mask = cv2.inRange(hsv, lower_black_hsv, upper_brown_hsv)
        brown_mask = cv2.inRange(hsv, lower_brown_hsv, upper_brown_hsv)

        l_channel = lab[:,:,0]
        a_channel = lab[:,:,1]
        b_channel = lab[:,:,2]

        l_grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        l_grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        a_grad_x = cv2.Sobel(a_channel, cv2.CV_64F, 1, 0, ksize=3)
        a_grad_y = cv2.Sobel(a_channel, cv2.CV_64F, 0, 1, ksize=3)
        b_grad_x = cv2.Sobel(b_channel, cv2.CV_64F, 1, 0, ksize=3)
        b_grad_y = cv2.Sobel(b_channel, cv2.CV_64F, 0, 1, ksize=3)

        l_grad_mag = np.sqrt(l_grad_x**2 + l_grad_y**2)
        a_grad_mag = np.sqrt(a_grad_x**2 + a_grad_y**2)
        b_grad_mag = np.sqrt(b_grad_x**2 + b_grad_y**2)

        edge_mask = np.zeros_like(l_channel, dtype=np.uint8)
        edge_mask[(l_grad_mag > 20) & (a_grad_mag > 10) & (b_grad_mag > 10)] = 255

        dark_lab_mask = (l_channel < 45).astype(np.uint8) * 255

        combined_mask = cv2.bitwise_or(black_mask, brown_mask)
        combined_mask = cv2.bitwise_or(combined_mask, dark_lab_mask)
        combined_mask = cv2.bitwise_and(combined_mask, apple_mask)

        kernel = np.ones((3,3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        marked_image = image.copy()
        has_defects = False
        spot_count = 0
        total_dark_area = 0
        apple_area = np.sum(apple_mask > 0)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 25 < area < 1000:
                x, y, w, h = cv2.boundingRect(contour)
                roi = image[y:y+h, x:x+w]
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(roi_mask, [contour], -1, 255, -1)

                if roi.size > 0:
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                    avg_value = np.mean(hsv_roi[:,:,2][roi_mask > 0])
                    avg_l = np.mean(lab_roi[:,:,0][roi_mask > 0])

                    edge_roi = edge_mask[y:y+h, x:x+w]
                    edge_presence = np.sum(edge_roi & roi_mask) > 0

                    if ((avg_value < 85 and avg_l < 55) or
                        (avg_value < 75 and edge_presence)):
                        color_std = np.std(lab_roi[:,:,0][roi_mask > 0])
                        if color_std < 25:
                            spot_count += 1
                            total_dark_area += area
                            cv2.drawContours(marked_image, [contour], -1, (0, 0, 255), 2)

        dark_area_ratio = total_dark_area / apple_area if apple_area > 0 else 0
        has_defects = ((spot_count >= 2 and dark_area_ratio > 0.005) or
                       (spot_count >= 1 and dark_area_ratio > 0.02))

        if has_defects:
            cv2.putText(marked_image,
                        f"Black spots: {spot_count}, Area: {dark_area_ratio:.1%}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

        return has_defects, marked_image

    except Exception as e:
        print(f"Error in detect_black_spots: {str(e)}")
        return False, image


def is_red_apple(image):
    """
    Check if the apple is predominantly red.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    red_pixels = np.sum(red_mask > 0)
    total_pixels = np.sum(image[:,:,0] > 0)
    red_ratio = red_pixels / total_pixels if total_pixels else 0
    return red_ratio > 0.40


def detect_brown_spots(image):
    """
    Simplified and more sensitive brown spot detection
    """
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Broader range for brown detection
        lower_brown = np.array([0, 20, 20])  # Much lower saturation threshold
        upper_brown = np.array([25, 255, 200])  # Wider hue range
        
        min_area = 50  # Smaller minimum area
        brown_ratio_threshold = 0.03  # Lower ratio threshold
        texture_variance_threshold = 15  # Lower texture threshold

        brown_mask_hsv = cv2.inRange(hsv, lower_brown, upper_brown)

        # Get apple mask (non-black pixels)
        apple_mask = (image[:,:,0] > 0).astype(np.uint8) * 255
        brown_mask = cv2.bitwise_and(brown_mask_hsv, apple_mask)

        # Gradient detection for texture
        L = lab[:,:,0]
        grad_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mask = (grad_mag > 15).astype(np.uint8) * 255  # Lower gradient threshold

        combined_mask = cv2.bitwise_and(brown_mask, gradient_mask)

        # Morphological operations
        kernel = np.ones((3,3), np.uint8)  # Smaller kernel
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marked_image = image.copy()
        has_brown_spots = False
        spot_count = 0
        spot_areas = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Skip very top portion (stem area)
                if y < image.shape[0] * 0.1:  # Reduced from 0.2
                    continue

                roi = image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_brown_mask = cv2.inRange(roi_hsv, lower_brown, upper_brown)
                brown_pixels = np.sum(roi_brown_mask > 0)
                total_roi_pixels = roi.shape[0] * roi.shape[1]

                if total_roi_pixels == 0:
                    continue

                ratio = brown_pixels / total_roi_pixels
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                texture_variance = np.var(gray_roi)

                if ratio > brown_ratio_threshold and texture_variance > texture_variance_threshold:
                    has_brown_spots = True
                    spot_count += 1
                    spot_areas.append(area)
                    
                    # Draw filled contour with transparency
                    overlay = marked_image.copy()
                    cv2.drawContours(overlay, [contour], -1, (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.4, marked_image, 0.6, 0, marked_image)
                    
                    # Draw contour border
                    cv2.drawContours(marked_image, [contour], -1, (0, 0, 255), 2)
                    
                    # Add spot number and size
                    cv2.putText(marked_image,
                              f"#{spot_count} ({area:.0f}px)",
                              (x, y-5),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5,
                              (0, 0, 255),
                              2)

        if has_brown_spots:
            total_area = sum(spot_areas)
            cv2.putText(marked_image,
                        f"Brown spots: {spot_count}, Total area: {total_area:.0f}px",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

        return has_brown_spots, marked_image

    except Exception as e:
        print(f"Error in detect_brown_spots: {str(e)}")
        return False, image


def check_overall_darkness(image):
    """
    Check if the apple is too dark overall.
    """
    try:
        marked_image = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])
        if green_ratio > 0.3:
            return False, image

        value_channel = hsv[:, :, 2]
        avg_brightness = np.mean(value_channel)
        total_pixels = value_channel.size
        dark_pixels = np.sum(value_channel < 50)
        dark_ratio = dark_pixels / total_pixels

        red_lower = np.array([170, 50, 50])
        red_upper = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        red_ratio = np.sum(red_mask > 0) / total_pixels

        is_too_dark = False
        if red_ratio > 0.3:
            if avg_brightness < 35 and dark_ratio > 0.80:
                is_too_dark = True
        else:
            if avg_brightness < 40 and dark_ratio > 0.85:
                is_too_dark = True

        if is_too_dark:
            h, w = marked_image.shape[:2]
            cv2.putText(marked_image,
                        'too dark apple :(',
                        (w // 4, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2)

        return is_too_dark, marked_image

    except Exception as e:
        print(f"Error in check_overall_darkness: {str(e)}")
        return False, image


def detect_wrinkles(image):
    """
    Strict wrinkle detection that ignores straight lines and edges
    """
    try:
        marked_image = image.copy()
        
        # Create apple mask (where image is not black)
        apple_mask = (image[:,:,0] > 0).astype(np.uint8) * 255
        
        # Erode the mask slightly to avoid edge detection at boundaries
        kernel = np.ones((5,5), np.uint8)
        apple_mask_eroded = cv2.erode(apple_mask, kernel, iterations=1)
        
        # Only process the apple area
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=apple_mask_eroded)
        
        # Apply CLAHE with stricter parameters
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Stronger blur to remove noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Stricter adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            4
        )

        # Stricter edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        combined = cv2.bitwise_and(binary, edges)
        combined = cv2.bitwise_and(combined, apple_mask_eroded)

        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.dilate(combined, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_wrinkles = False
        wrinkle_count = 0
        valid_wrinkles = []

        apple_area = np.sum(apple_mask > 0)
        min_wrinkle_area = apple_area * 0.001
        max_wrinkle_area = apple_area * 0.03

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_wrinkle_area or area > max_wrinkle_area:
                continue

            # Check for straightness using polynomial approximation
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If too few points after approximation, it's too straight
            if len(approx) < 4:
                continue
                
            # Calculate contour curvature
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Convert contour to points array for curvature calculation
            points = contour.squeeze()
            if len(points.shape) < 2:
                continue
                
            # Calculate local curvature variations
            curves = []
            for i in range(2, len(points)-2):
                p1 = points[i-2]
                p2 = points[i]
                p3 = points[i+2]
                if len(p1) == 2 and len(p2) == 2 and len(p3) == 2:
                    v1 = p2 - p1
                    v2 = p3 - p2
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        angle = np.abs(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))
                        curves.append(angle)
            
            # Skip if not enough points for curvature calculation
            if not curves:
                continue
                
            # Skip if curvature is too uniform (likely a straight line or circle)
            curve_std = np.std(curves)
            if curve_std < 0.2:  # Require significant curvature variation
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h

            # Check if the contour is not too straight using aspect ratio
            if 0.1 < aspect_ratio < 5.0:
                roi = combined[y:y+h, x:x+w]
                if roi.size > 0:
                    edge_density = np.sum(roi) / (roi.size * 255)
                    if edge_density > 0.3:
                        pad = 5
                        y1 = max(0, y-pad)
                        y2 = min(gray.shape[0], y+h+pad)
                        x1 = max(0, x-pad)
                        x2 = min(gray.shape[1], x+w+pad)
                        
                        roi_gray = gray[y:y+h, x:x+w]
                        surr_gray = gray[y1:y2, x1:x2]
                        
                        if roi_gray.size > 0 and surr_gray.size > 0:
                            roi_std = np.std(roi_gray)
                            contrast_ratio = abs(np.mean(roi_gray) - np.mean(surr_gray))
                            
                            if roi_std > 20 and contrast_ratio > 30:
                                wrinkle_count += 1
                                valid_wrinkles.append(contour)

        if wrinkle_count >= 4:
            if len(valid_wrinkles) > 0:
                centroids = []
                for contour in valid_wrinkles:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centroids.append((cx, cy))

                if len(centroids) >= 4:
                    distances = []
                    for i in range(len(centroids)):
                        for j in range(i+1, len(centroids)):
                            dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 +
                                         (centroids[i][1] - centroids[j][1])**2)
                            distances.append(dist)

                    if distances:
                        avg_distance = np.mean(distances)
                        if avg_distance > apple_area ** 0.5 * 0.15:
                            has_wrinkles = True
                            for contour in valid_wrinkles:
                                cv2.drawContours(marked_image, [contour], -1, (0, 0, 255), 2)

                            cv2.putText(marked_image,
                                      f'Dehydration (wrinkles: {wrinkle_count})',
                                      (10, image.shape[0] - 40),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6,
                                      (0, 0, 255),
                                      2)

        return has_wrinkles, marked_image

    except Exception as e:
        print(f"Error in detect_wrinkles: {str(e)}")
        return False, image


def detect_spots_fourier(image):
    """
    Detect periodic patterns/spots using Fourier analysis
    (skipped for red apples).
    """
    try:
        if is_red_apple(image):
            return False, image, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        apple_mask = (image[:,:,0] > 0).astype(np.uint8) * 255
        gray = cv2.bitwise_and(gray, apple_mask)

        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)

        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        radii = [(2, 8), (8, 15), (15, 30)]
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        r_matrix = np.sqrt(x*x + y*y)

        filtered_images = []
        for r_min, r_max in radii:
            mask = np.zeros((rows, cols), np.float32)
            ring = (r_matrix >= r_min) & (r_matrix <= r_max)
            mask[ring] = 1

            f_filtered = f_shift * mask
            f_inverse = np.fft.ifftshift(f_filtered)
            img_back = np.abs(np.fft.ifft2(f_inverse))
            filtered_images.append(img_back)

        combined = np.zeros_like(filtered_images[0])
        for img in filtered_images:
            combined += img

        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(combined)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        marked_image = image.copy()
        valid_regions = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 40 < area < 1500:
                x_ = stats[i, cv2.CC_STAT_LEFT]
                y_ = stats[i, cv2.CC_STAT_TOP]
                w_ = stats[i, cv2.CC_STAT_WIDTH]
                h_ = stats[i, cv2.CC_STAT_HEIGHT]

                roi = gray[y_:y_+h_, x_:x_+w_]
                if roi.size == 0:
                    continue

                pad = 8
                y1 = max(0, y_-pad)
                y2 = min(image.shape[0], y_+h_+pad)
                x1 = max(0, x_-pad)
                x2 = min(image.shape[1], x_+w_+pad)

                roi_lab = lab[y_:y_+h_, x_:x_+w_]
                surr_lab = lab[y1:y2, x1:x2]

                roi_color_mean = np.mean(roi_lab, axis=(0,1))
                surr_color_mean = np.mean(surr_lab, axis=(0,1))
                color_diff = np.linalg.norm(roi_color_mean - surr_color_mean)

                roi_fft = np.fft.fft2(roi)
                roi_magnitude = np.abs(roi_fft)
                freq_mean = np.mean(roi_magnitude)
                freq_std = np.std(roi_magnitude)
                peak_ratio = np.max(roi_magnitude) / (freq_mean + 1e-8)

                if (peak_ratio > 2 and freq_std > freq_mean*0.5) and (color_diff > 12.0):
                    roi_var = np.var(roi_lab[:,:,0])
                    surr_var = np.var(surr_lab[:,:,0])
                    var_ratio = roi_var / (surr_var + 1e-8)

                    if var_ratio > 1.15 or var_ratio < 0.85:
                        valid_regions.append((x_, y_, w_, h_))
                        cv2.rectangle(marked_image, (x_, y_), (x_+w_, y_+h_), (0, 0, 255), 2)
                        cv2.putText(marked_image,
                                    f"dC:{color_diff:.1f}",
                                    (x_, y_-5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    (0, 0, 255),
                                    1)

        has_periodic_defects = len(valid_regions) >= 2
        if has_periodic_defects:
            cv2.putText(marked_image,
                        f"Periodic patterns: {len(valid_regions)}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

        return has_periodic_defects, marked_image, enhanced

    except Exception as e:
        print(f"Error in detect_spots_fourier: {str(e)}")
        return False, image, None


def has_significant_defect(marked_image):
    """
    Helper to check if there's a large highlighted defect area.
    """
    gray = cv2.cvtColor(marked_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            return True
    return False


def classify_apple(image):
    """
    Image-processing-based classification pipeline.
    Returns: (label, reasons, marked_image)
      label: "Defect" or "Healthy"
    """
    reasons = []
    marked_image = image.copy()
    y_offset = 30

    # 1) Red vs Non-red
    is_red = is_red_apple(image)
    apple_type = "Red Apple" if is_red else "Non-red Apple"
    cv2.putText(marked_image, f"Type: {apple_type}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2)
    y_offset += 25

    # 2) Forbidden color
    has_bad_color, color_marked_image, _ = check_forbidden_color_whole_apple(image)
    if has_bad_color:
        reasons.append("Bad overall color")
        marked_image = color_marked_image

    # 3) Black spots
    has_black_spots, black_spots_image = detect_black_spots(image)
    if has_black_spots:
        reasons.append("Black spots detected")
        if len(reasons) == 1:
            marked_image = black_spots_image
        else:
            black_mask = cv2.absdiff(black_spots_image, image)
            marked_image = cv2.add(marked_image, black_mask)

    # 4) Brown spots
    has_brown_spots, brown_spots_image = detect_brown_spots(image)
    if has_brown_spots:
        # Additional check for significance
        if not is_red or (is_red and has_significant_defect(brown_spots_image)):
            reasons.append("Brown spots detected")
            if len(reasons) == 1:
                marked_image = brown_spots_image
            else:
                brown_mask = cv2.absdiff(brown_spots_image, image)
                marked_image = cv2.add(marked_image, brown_mask)

    # 5) Overall darkness
    is_too_dark, dark_image = check_overall_darkness(image)
    if is_too_dark:
        reasons.append("Apple too dark")
        if len(reasons) == 1:
            marked_image = dark_image
        else:
            dark_mask = cv2.absdiff(dark_image, image)
            marked_image = cv2.add(marked_image, dark_mask)

    # 6) Wrinkles
    has_wrinkles, wrinkle_image = detect_wrinkles(image)
    if has_wrinkles:
        reasons.append("Wrinkly Apple")
        if len(reasons) == 1:
            marked_image = wrinkle_image
        else:
            wrinkle_mask = cv2.absdiff(wrinkle_image, image)
            marked_image = cv2.add(marked_image, wrinkle_mask)

    # 7) Fourier-based spot detection
    has_fourier_spots, fourier_spots_image, _ = detect_spots_fourier(image)
    if has_fourier_spots:
        reasons.append("Spots detected (Fourier)")
        if len(reasons) == 1:
            marked_image = fourier_spots_image
        else:
            fourier_mask = cv2.absdiff(fourier_spots_image, image)
            marked_image = cv2.add(marked_image, fourier_mask)

    if len(reasons) > 0:
        cv2.putText(marked_image, "CLASSIFICATION: DEFECTED",
                    (10, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)
        return "Defect", reasons, marked_image
    else:
        # Don't write "HEALTHY" text, just return the image with apple type
        return "Healthy", [], marked_image


def get_defect_types(image):
    """
    Returns a list of defect types found in the image without affecting classification.
    Defect types are now using standardized apple disease terminology.
    """
    defect_types = []
    
    # Check each type of defect
    has_bad_color, _, _ = check_forbidden_color_whole_apple(image)
    if has_bad_color:
        defect_types.append("Russeting (Discoloration)")
    
    has_black_spots, _ = detect_black_spots(image)
    if has_black_spots:
        defect_types.append("Sunburn")
    
    has_brown_spots, brown_spots_image = detect_brown_spots(image)
    if has_brown_spots and (not is_red_apple(image) or has_significant_defect(brown_spots_image)):
        defect_types.append("Chilling Injury/Bruising")
    
    is_too_dark, _ = check_overall_darkness(image)
    if is_too_dark:
        defect_types.append("Russeting (Dark)")
    
    has_wrinkles, _ = detect_wrinkles(image)
    if has_wrinkles:
        defect_types.append("Dehydration")
    
    has_fourier_spots, _, _ = detect_spots_fourier(image)
    if has_fourier_spots:
        defect_types.append("Fungal/Bacterial Infection")
    
    return defect_types


################################################################################
# DEEP LEARNING + IMAGE PROCESSING MERGED PIPELINE
################################################################################
if __name__ == "__main__":
    # ------------------
    # A. Deep Learning
    # ------------------
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        directory='dataset_bgremoved',   # <-- Set your correct path
        target_size=(224, 224),     # <-- Adjust if needed
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    # Load model
    try:
        model = tf.keras.models.load_model('apple_defect_modeltrainingeebackground.h5')
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Evaluate
    print("\nEvaluating model on test set...")
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # -------------
    # B. Prediction
    # -------------
    os.makedirs('predicted/healthy', exist_ok=True)
    os.makedirs('predicted/defect', exist_ok=True)

    # Initialize counters for metrics
    true_positives = 0  # Correctly identified defects
    false_positives = 0  # Healthy apples wrongly classified as defects
    false_negatives = 0  # Defect apples wrongly classified as healthy
    true_negatives = 0  # Correctly identified healthy apples

    # Add these before the prediction loop:
    defect_counts = {
        "Russeting (Discoloration)": 0,
        "Sunburn": 0,
        "Chilling Injury/Bruising": 0,
        "Russeting (Dark)": 0,
        "Dehydration": 0,
        "Fungal/Bacterial Infection": 0
    }
    total_images = 0
    total_defected_apples = 0

    print("\nGenerating predictions...")
    test_generator.reset()
    preds = model.predict(test_generator)

    # For each sample, if DL is confident => use that. Otherwise => fallback
    for (confidence, path) in zip(preds, test_generator.filepaths):
        confidence_val = confidence[0]
        healthy_conf = confidence_val * 100
        defect_conf = (1 - confidence_val) * 100
        
        # Read image for defect type detection
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"Error: cannot read image {path}, skipping.")
            continue

        # Determine ground truth from original path
        is_defect_ground_truth = 'defect' in path.lower()

        # If the DL model is sure about healthy...
        if healthy_conf >= 80:
            final_label = "Healthy"
            out_path = os.path.join('predicted/healthy', os.path.basename(path))
            cv2.imwrite(out_path, img_bgr)  # Save original image without markings
            print(f"{os.path.basename(path)} => [Deep Learning] Healthy ({healthy_conf:.2f}%)")
            if is_defect_ground_truth:
                false_negatives += 1
            else:
                true_negatives += 1

        # If the DL model is sure about defect...
        elif defect_conf >= 80:
            final_label = "Defect"
            _, reasons, marked_img = classify_apple(img_bgr)  # Only run defect detection for defective apples
            out_path = os.path.join('predicted/defect', os.path.basename(path))
            cv2.imwrite(out_path, marked_img)
            total_defected_apples += 1
            defect_types = get_defect_types(img_bgr)
            if defect_types:
                for defect in defect_types:
                    defect_counts[defect] += 1
            defect_str = ", ".join(defect_types) if defect_types else "No specific defects detected"
            print(f"{os.path.basename(path)} => [Deep Learning] Defect ({defect_conf:.2f}%) - Types: {defect_str}")
            if is_defect_ground_truth:
                true_positives += 1
            else:
                false_positives += 1

        else:
            # Fallback to Image-Processing
            fallback_label, reasons, marked_img = classify_apple(img_bgr)

            if fallback_label == "Defect":
                final_label = "Defect"
                out_path = os.path.join('predicted/defect', os.path.basename(path))
                cv2.imwrite(out_path, marked_img)
                total_defected_apples += 1
                defect_types = get_defect_types(img_bgr)
                if defect_types:
                    for defect in defect_types:
                        defect_counts[defect] += 1
                defect_str = ", ".join(defect_types) if defect_types else "No specific defects detected"
                print(f"{os.path.basename(path)} => [Fallback] Defect - Types: {defect_str}")
                if is_defect_ground_truth:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                final_label = "Healthy"
                out_path = os.path.join('predicted/healthy', os.path.basename(path))
                cv2.imwrite(out_path, img_bgr)  # Save original image without markings
                print(f"{os.path.basename(path)} => [Fallback] Healthy")
                if is_defect_ground_truth:
                    false_negatives += 1
                else:
                    true_negatives += 1

        total_images += 1

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    print("\nClassification Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")

    print("\nDefect Type Statistics:")
    print("-" * 50)
    print(f"Total Images: {total_images}")
    print(f"Total Defected Apples: {total_defected_apples}")
    print("\nDefect Type Distribution among Defected Apples:")
    for defect_type, count in defect_counts.items():
        if total_defected_apples > 0:
            percentage = (count / total_defected_apples) * 100
            print(f"{defect_type}: {count} occurrences ({percentage:.2f}% of defected apples)")
    print("-" * 50)

    print("\nAll images classified! No image remains uncertain.")
    print("Results saved in 'predicted/healthy' and 'predicted/defect'.")
