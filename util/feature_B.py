import cv2
import numpy as np
import os
import pandas as pd
import shutil
from os.path import join

def measureborderirregularity(mask_path):
    """
    Measures the border irregularity of a binary mask using the circularity formula.

    Args:
        mask_path (str): Path to the binary mask image file.

    Returns:
        float or None: The border irregularity score based on the contour's circularity,
                       or None if the mask cannot be processed.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not load: {mask_path}")
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if area == 0:
        return None

    # Circularity formula
    irregularity = (perimeter ** 2) / (4 * np.pi * area)
    return irregularity

def IrregularityForAll(masked_path, output_csv):
    """
    Processes all mask files in the given folder and calculates their border irregularity scores.
    Saves the results to a CSV file.
    """
    output_csv = os.path.join(output_csv, 'irregularity_scores.csv')
    mask_files = [f for f in os.listdir(masked_path) if f.endswith('.png') and not f.startswith('._')]

    results = []

    for filename in mask_files:
        file_path = join(masked_path, filename)
        try:
            score = measureborderirregularity(file_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            score = 'N/A'

        clean_filename = filename.replace('_mask', '')
        results.append({'filename': clean_filename, 'irregularity_score': score})


    df = pd.DataFrame(results, columns=['filename', 'irregularity_score'])
    df.to_csv(output_csv, index=False)
    print(f"Irregularity scores saved to: {output_csv}")