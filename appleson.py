import os
import cv2
import numpy as np
import tensorflow as tf
import torch
from ultralytics import YOLO
from shutil import copy2
from pathlib import Path

################################################################################
# 1. BACKGROUND REMOVAL FUNCTIONS
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
            # Add small padding
            cv2.drawContours(final_mask, [contour], -1, 255, 5)

    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = (final_mask > 0).astype(np.uint8) * 255

    return final_mask


def remove_background_batch(input_folder, output_folder):
    """
    Removes background for all images in `input_folder`, saving results to `output_folder`.
    Uses YOLOv8 segmentation, and if that fails, uses fallback color/curvature method.
    """
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[BG-Removal] Using device: {device}")

    # Load YOLOv8n-seg
    model = YOLO("yolov8n-seg.pt")
    model.to(device)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in Path(input_folder).glob('*') if f.suffix.lower() in image_extensions]

    print(f"[BG-Removal] Found {len(image_files)} images in '{input_folder}'...")
    for img_path in image_files:
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[BG-Removal] Failed to read {img_path.name}")
                continue

            # Try YOLO first
            results = model(image, conf=0.3)
            result = results[0]

            # If YOLO fails => fallback
            if not hasattr(result, 'masks') or result.masks is None or len(result.masks) == 0:
                print(f"[BG-Removal] YOLO failed for {img_path.name}, using fallback...")
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
            print(f"[BG-Removal] Error processing {img_path.name}: {str(e)}")

    print(f"[BG-Removal] Done removing backgrounds for '{input_folder}'!\n")


################################################################################
# 2. IMAGE-PROCESSING CLASSIFICATION PIPELINE
################################################################################

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
    if total_pixels == 0:
        return False
    return (red_pixels / total_pixels) > 0.40


def check_if_green_apple(hsv, apple_mask):
    """
    Helper to identify green apples
    """
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_mask = cv2.bitwise_and(green_mask, apple_mask)

    total_apple_pixels = np.sum(apple_mask > 0)
    if total_apple_pixels == 0:
        return False

    green_ratio = np.sum(green_mask > 0) / total_apple_pixels
    return green_ratio > 0.3


def check_forbidden_color_whole_apple(image):
    """
    Checks for dark or brown coloration that suggests a defect.
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

        total_apple_pixels = np.sum(apple_mask > 0)
        if total_apple_pixels == 0:
            return False, image, None

        # Ratios
        brown_ratio = np.sum(brown_mask > 0) / total_apple_pixels
        dark_ratio = np.sum(dark_mask > 0) / total_apple_pixels

        is_bad_color = False
        reasons = []

        if brown_ratio > 0.60:
            is_bad_color = True
            reasons.append(f"severe browning ({brown_ratio:.1%} brown)")

        if dark_ratio > dark_ratio_threshold:
            is_bad_color = True
            reasons.append(f"too dark ({dark_ratio:.1%} dark)")

        hsv_apple = hsv[apple_mask > 0]
        if hsv_apple.size > 0:
            avg_value = np.mean(hsv_apple[:,2])
            brightness_threshold = 65 if is_green else 75
            if avg_value < brightness_threshold:
                is_bad_color = True
                reasons.append(f"discoloration (brightness: {avg_value:.1f})")

        if is_bad_color:
            for i, r in enumerate(reasons):
                cv2.putText(marked_image, r, (10, 30 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return is_bad_color, marked_image, None

    except Exception as e:
        print(f"[ForbiddenColorError] {e}")
        return False, image, None


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

        # Edge detection
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
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(roi_mask, [contour], -1, 255, -1)

                roi_lab = lab[y:y+h, x:x+w]
                if roi_lab.size == 0:
                    continue

                avg_value = np.mean(cv2.cvtColor(roi_lab, cv2.COLOR_Lab2BGR)[:,:,2][roi_mask > 0])
                avg_l = np.mean(roi_lab[:,:,0][roi_mask > 0])
                edge_roi = edge_mask[y:y+h, x:x+w]
                edge_presence = np.sum(edge_roi & roi_mask) > 0

                if ((avg_value < 85 and avg_l < 55) or
                    (avg_value < 75 and edge_presence)):
                    color_std = np.std(roi_lab[:,:,0][roi_mask > 0])
                    if color_std < 25:
                        spot_count += 1
                        total_dark_area += area
                        cv2.drawContours(marked_image, [contour], -1, (0, 0, 255), 2)

        dark_area_ratio = total_dark_area / apple_area if apple_area else 0
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
        print(f"[BlackSpotsError] {e}")
        return False, image


def detect_brown_spots(image):
    """
    Brown spot detection with combined HSV & Lab approach,
    excluding strong red areas for red apples.
    """
    try:
        red_apple = is_red_apple(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        lower_red1 = np.array([0, 60, 60])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 60, 60])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        strong_red_mask = cv2.bitwise_or(mask_red1, mask_red2)

        if red_apple:
            lower_brown = np.array([5, 40, 40])
            upper_brown = np.array([17, 255, 200])
            min_area = 300
            brown_ratio_threshold = 0.2
            texture_variance_threshold = 80
        else:
            lower_brown = np.array([2, 30, 30])
            upper_brown = np.array([18, 255, 200])
            min_area = 200
            brown_ratio_threshold = 0.1
            texture_variance_threshold = 30

        brown_mask_hsv = cv2.inRange(hsv, lower_brown, upper_brown)
        exclude_red = cv2.bitwise_not(strong_red_mask)
        brown_mask = cv2.bitwise_and(brown_mask_hsv, exclude_red)

        L = lab[:,:,0]
        grad_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        grad_thresh = 40 if red_apple else 30
        gradient_mask = (grad_mag > grad_thresh).astype(np.uint8) * 255

        combined_mask = cv2.bitwise_and(brown_mask, gradient_mask)

        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marked_image = image.copy()
        has_brown_spots = False

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                if y < image.shape[0] * 0.2:
                    continue

                roi = image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_brown_mask = cv2.inRange(roi_hsv, lower_brown, upper_brown)

                if red_apple:
                    roi_exclude_red = cv2.bitwise_not(cv2.inRange(roi_hsv, lower_red1, upper_red1))
                    roi_exclude_red2 = cv2.bitwise_not(cv2.inRange(roi_hsv, lower_red2, upper_red2))
                    roi_no_red = cv2.bitwise_and(roi_brown_mask, roi_exclude_red)
                    roi_no_red = cv2.bitwise_and(roi_no_red, roi_exclude_red2)
                    brown_pixels = np.sum(roi_no_red > 0)
                else:
                    brown_pixels = np.sum(roi_brown_mask > 0)

                total_roi_pixels = roi.shape[0] * roi.shape[1]
                if total_roi_pixels == 0:
                    continue

                ratio = brown_pixels / total_roi_pixels
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                texture_variance = np.var(gray_roi)

                if ratio > brown_ratio_threshold and texture_variance > texture_variance_threshold:
                    has_brown_spots = True
                    cv2.drawContours(marked_image, [contour], -1, (0, 0, 255), 2)
                    cv2.putText(marked_image,
                                f"defect ({ratio:.1%})",
                                (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1)

        return has_brown_spots, marked_image

    except Exception as e:
        print(f"[BrownSpotsError] {e}")
        return False, image


def check_overall_darkness(image):
    """
    Check if the apple is too dark overall.
    """
    try:
        marked_image = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Quick green check
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
            cv2.putText(marked_image, 'too dark apple :(', (w // 4, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        return is_too_dark, marked_image

    except Exception as e:
        print(f"[DarknessError] {e}")
        return False, image


def detect_wrinkles(image):
    """
    Enhanced wrinkle detection
    """
    try:
        marked_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            4
        )

        edges = cv2.Canny(blurred, 100, 200)
        combined = cv2.bitwise_and(binary, edges)

        height = combined.shape[0]
        combined[:int(height*0.25), :] = 0

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        has_wrinkles = False
        wrinkle_count = 0
        valid_wrinkles = []

        apple_area = np.sum(image[:,:,0] > 0)
        min_wrinkle_area = apple_area * 0.001
        max_wrinkle_area = apple_area * 0.015

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_wrinkle_area or area > max_wrinkle_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h

            if (aspect_ratio < 0.15 or aspect_ratio > 6.0) and area > min_wrinkle_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.2:
                    _, (width, height_box), angle = cv2.minAreaRect(contour)
                    if min(width, height_box) > 0:
                        elongation = max(width, height_box) / min(width, height_box)
                        if elongation > 4 and min(width, height_box) < max_wrinkle_area * 0.1:
                            roi = image[max(0, y-5):min(image.shape[0], y+h+5),
                                        max(0, x-5):min(image.shape[1], x+w+5)]
                            if roi.size > 0:
                                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                std_dev = np.std(hsv_roi[:,:,1])
                                if std_dev > 20:
                                    wrinkle_count += 1
                                    valid_wrinkles.append(contour)

        if wrinkle_count >= 4 and len(valid_wrinkles) > 0:
            centroids = []
            for contour in valid_wrinkles:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))

            if len(centroids) >= 2:
                distances = []
                for i in range(len(centroids)):
                    for j in range(i+1, len(centroids)):
                        dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 +
                                       (centroids[i][1] - centroids[j][1])**2)
                        distances.append(dist)

                avg_distance = np.mean(distances)
                if avg_distance > apple_area ** 0.5 * 0.15:
                    has_wrinkles = True
                    for contour in valid_wrinkles:
                        cv2.drawContours(marked_image, [contour], -1, (0, 0, 255), 2)

                    cv2.putText(marked_image, f'Wrinkles found: {wrinkle_count}',
                                (10, image.shape[0] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)

        return has_wrinkles, marked_image

    except Exception as e:
        print(f"[WrinklesError] {e}")
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

        # Adjusted frequency bands
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

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
                        cv2.putText(marked_image, f"dC:{color_diff:.1f}",
                                    (x_, y_-5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (0, 0, 255), 1)

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
        print(f"[FourierSpotsError] {e}")
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
    Image-processing pipeline => label = "Defect" or "Healthy".
    Returns: (label, reasons, marked_image)
    """
    reasons = []
    marked_image = image.copy()

    # 1) Forbidden color
    has_bad_color, color_marked, _ = check_forbidden_color_whole_apple(image)
    if has_bad_color:
        reasons.append("Bad color")
        marked_image = color_marked

    # 2) Black spots
    has_black, black_spots_image = detect_black_spots(image)
    if has_black:
        reasons.append("Black spots detected")
        if len(reasons) == 1:
            marked_image = black_spots_image
        else:
            black_mask = cv2.absdiff(black_spots_image, image)
            marked_image = cv2.add(marked_image, black_mask)

    # 3) Brown spots
    has_brown, brown_spots_image = detect_brown_spots(image)
    if has_brown:
        if is_red_apple(image):
            # For red apples, check significance
            if has_significant_defect(brown_spots_image):
                reasons.append("Brown spots detected")
                if len(reasons) == 1:
                    marked_image = brown_spots_image
                else:
                    brown_mask = cv2.absdiff(brown_spots_image, image)
                    marked_image = cv2.add(marked_image, brown_mask)
        else:
            reasons.append("Brown spots detected")
            if len(reasons) == 1:
                marked_image = brown_spots_image
            else:
                brown_mask = cv2.absdiff(brown_spots_image, image)
                marked_image = cv2.add(marked_image, brown_mask)

    # 4) Overall darkness
    is_dark, dark_img = check_overall_darkness(image)
    if is_dark:
        reasons.append("Too dark")
        if len(reasons) == 1:
            marked_image = dark_img
        else:
            dark_mask = cv2.absdiff(dark_img, image)
            marked_image = cv2.add(marked_image, dark_mask)

    # 5) Wrinkles
    has_wrinkles, wrinkle_image = detect_wrinkles(image)
    if has_wrinkles:
        reasons.append("Wrinkly apple")
        if len(reasons) == 1:
            marked_image = wrinkle_image
        else:
            wrinkle_mask = cv2.absdiff(wrinkle_image, image)
            marked_image = cv2.add(marked_image, wrinkle_mask)

    # 6) Fourier-based spots
    has_fourier_spots, fourier_image, _ = detect_spots_fourier(image)
    if has_fourier_spots:
        reasons.append("Spots detected (Fourier)")
        if len(reasons) == 1:
            marked_image = fourier_image
        else:
            fourier_mask = cv2.absdiff(fourier_image, image)
            marked_image = cv2.add(marked_image, fourier_mask)

    if len(reasons) > 0:
        cv2.putText(marked_image, "CLASSIFICATION: DEFECTED",
                    (10, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return "Defect", reasons, marked_image
    else:
        cv2.putText(marked_image, "CLASSIFICATION: HEALTHY",
                    (10, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return "Healthy", reasons, marked_image


################################################################################
# 3. MAIN SCRIPT
################################################################################
if __name__ == "__main__":

    # A. We have 2 subfolders: "defect" and "healthy" under "appledataset"
    input_root = "appledataset"
    bg_removed_root = "Background_Removed"

    subfolders = ["defect", "healthy"]

    # 1) Remove backgrounds separately for each subfolder
    for sub in subfolders:
        in_sub_path = os.path.join(input_root, sub)
        out_sub_path = os.path.join(bg_removed_root, sub)
        remove_background_batch(in_sub_path, out_sub_path)

    # 2) Load deep-learning model
    print("[INFO] Loading Keras model...")
    try:
        model = tf.keras.models.load_model("apple_defect_modeltrainingee.h5")
        print("[INFO] Model loaded successfully!\n")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        raise

    # Create final output folders
    os.makedirs("predicted/defect", exist_ok=True)
    os.makedirs("predicted/healthy", exist_ok=True)
    os.makedirs("predicted_removed/defect", exist_ok=True)
    os.makedirs("predicted_removed/healthy", exist_ok=True)

    # We'll classify all background-removed images. We'll read them subfolder by subfolder.
    for sub in subfolders:
        bg_removed_sub = os.path.join(bg_removed_root, sub)  # e.g. Background_Removed/defect
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        bg_files = [f for f in Path(bg_removed_sub).glob('*') if f.suffix.lower() in image_extensions]
        bg_files.sort()

        print(f"[CLASSIFY] Found {len(bg_files)} images in '{bg_removed_sub}' to classify...")

        def preprocess_for_model(img_bgr, target_size=(224, 224)):
            # Basic resizing and normalization
            img_resized = cv2.resize(img_bgr, target_size)
            img_resized = img_resized.astype(np.float32) / 255.0
            return img_resized

        # We can do small batches for model.predict()
        BATCH_SIZE = 16
        for start_idx in range(0, len(bg_files), BATCH_SIZE):
            batch_files = bg_files[start_idx:start_idx + BATCH_SIZE]
            batch_imgs = []
            file_to_index = {}  # track which indices belong to which file

            # Load images in this batch
            for i, bf in enumerate(batch_files):
                bgr = cv2.imread(str(bf))
                if bgr is None:
                    print(f"[WARN] Cannot read {bf.name}, skipping.")
                    file_to_index[bf] = None
                    continue

                preprocessed = preprocess_for_model(bgr)
                file_to_index[bf] = len(batch_imgs)
                batch_imgs.append(preprocessed)

            if len(batch_imgs) == 0:
                continue

            # Convert to numpy for model
            input_array = np.array(batch_imgs)
            preds = model.predict(input_array)  # shape: (batch_size, 1)

            # Go through each file in the batch
            for bf in batch_files:
                if file_to_index[bf] is None:
                    continue  # couldn't read it

                idx_in_batch = file_to_index[bf]
                confidence_val = preds[idx_in_batch][0]
                healthy_conf = confidence_val * 100
                defect_conf = (1 - confidence_val) * 100

                # The original image is in appledataset/<sub>
                orig_path = os.path.join(input_root, sub, bf.name)
                orig_bgr = cv2.imread(orig_path)
                if orig_bgr is None:
                    print(f"[WARN] Original missing: {bf.name} => skipping.")
                    continue

                # Decision
                if healthy_conf >= 80:
                    # Model says healthy
                    copy2(orig_path, f"predicted/healthy")
                    copy2(str(bf), f"predicted_removed/healthy")
                    print(f"{bf.name} => [DL] Healthy ({healthy_conf:.2f}%)")

                elif defect_conf >= 80:
                    # Model says defect
                    copy2(orig_path, f"predicted/defect")
                    copy2(str(bf), f"predicted_removed/defect")
                    print(f"{bf.name} => [DL] Defect ({defect_conf:.2f}%)")

                else:
                    # Fallback => use image-processing on the BG-removed image
                    fallback_bgr = cv2.imread(str(bf))
                    if fallback_bgr is None:
                        print(f"[WARN] Cannot read BG-removed {bf.name}, skipping fallback.")
                        continue

                    label, reasons, marked_img = classify_apple(fallback_bgr)
                    if label == "Defect":
                        copy2(orig_path, "predicted/defect")
                        out_path = os.path.join("predicted_removed/defect", bf.name)
                        cv2.imwrite(out_path, marked_img)
                        print(f"{bf.name} => [Fallback] Defect. {reasons}")
                    else:
                        copy2(orig_path, "predicted/healthy")
                        out_path = os.path.join("predicted_removed/healthy", bf.name)
                        cv2.imwrite(out_path, marked_img)
                        print(f"{bf.name} => [Fallback] Healthy.")

    print("\nAll images classified!")
    print("Original images => predicted/(defect|healthy)")
    print("BG-removed images => predicted_removed/(defect|healthy)")
    print("No images remain uncertain. Done!")
