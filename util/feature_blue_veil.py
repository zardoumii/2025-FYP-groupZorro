import os
import pandas as pd
import cv2
import numpy as np

def BlueWhiteVeilForAll(image_folder, mask_folder, output_csv):
    """
    Computes the blue-white veil percentage for each image+mask pair and saves results.
    """

    def compute_blue_white_veil_percentage(image_path, mask_path):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            return 'N/A'

        # Make sure mask is binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Convert image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Only consider pixels inside the mask
        masked_hsv = hsv_image[binary_mask == 255]

        if len(masked_hsv) == 0:
            return 'N/A'

        # Blue region: typical blue veil range
        blue_mask = (
            (masked_hsv[:, 0] >= 100) & (masked_hsv[:, 0] <= 140) &  # Hue blue
            (masked_hsv[:, 1] >= 50) & (masked_hsv[:, 1] <= 255) &    # Saturation normal
            (masked_hsv[:, 2] >= 50) & (masked_hsv[:, 2] <= 200)      # Value not shiny
        )

        # White region: low Saturation, high Value
        white_mask = (
            (masked_hsv[:, 1] <= 30) &   # Low saturation
            (masked_hsv[:, 2] >= 200)    # High value (brightness)
        )

        # Combine both blue and white areas
        blue_white_pixels = np.sum(blue_mask | white_mask)

        total_pixels = len(masked_hsv)

        blue_white_veil_percentage = (blue_white_pixels / total_pixels) * 100
        return blue_white_veil_percentage

    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png') and not f.startswith('._')]
    results = []

    for mask_filename in mask_files:
        base_name = mask_filename.replace('_mask', '')
        image_filename = base_name

        image_path = os.path.join(image_folder, image_filename)
        mask_path = os.path.join(mask_folder, mask_filename)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            score = 'N/A'
        else:
            try:
                score = compute_blue_white_veil_percentage(image_path, mask_path)
            except Exception as e:
                print(f"Error processing {mask_filename}: {e}")
                score = 'N/A'

        results.append({'filename': image_filename, 'blue_white_veil_score': score})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Blue-White veil scores saved to: {output_csv}")
