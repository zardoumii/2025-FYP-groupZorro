import numpy as np
import cv2
import os
import pandas as pd
from os.path import join 

def colorvariationscore(image_path, mask_path):
    """
    Calculates the color variation score of a lesion based on standard deviation in RGB channels.

    Args:
        image_path (str): Path to the original RGB image.
        mask_path (str): Path to the binary mask image.

    Returns:
        float or None: The average standard deviation of RGB values within the masked area,
                       or None if inputs are invalid or lesion area is empty.
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Could not load {image_path} or {mask_path}")
        return None


    mask = (mask > 0).astype(np.uint8)


    lesion_pixels = image[mask == 1]

    if lesion_pixels.size == 0:
        return None

 
    std_per_channel = np.std(lesion_pixels, axis=0)
    avg_std = np.mean(std_per_channel)

    return avg_std

def ColorVariationForAll(image_folder, mask_folder, output_csv):
    """
    Computes color variation scores for all image-mask pairs in specified folders and stores results.
    
    """
    output_csv = os.path.join(output_csv, 'color_variation_scores.csv')
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]
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
                score = colorvariationscore(image_path, mask_path)
            except Exception as e:
                print(f"Error processing {mask_filename}: {e}")
                score = 'N/A'

        results.append({'filename': image_filename, 'color_variation_score': score})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Color variation scores saved to: {output_csv}")