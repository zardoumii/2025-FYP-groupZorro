import cv2
import numpy as np
import pandas as pd
from math import sqrt, floor, ceil, nan, pi
from skimage import color, exposure
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.transform import rotate
from skimage import morphology
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar, circstd
from statistics import variance, stdev
from scipy.spatial import ConvexHull
from concurrent.futures import ProcessPoolExecutor
import os
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



def find_midpoint_v4(mask):
        """
        Find the midpoint of a binary mask using the horizontal center of mass and 
        return the index of the column that is closest to the center of mass.

        """
        # sum all pixels in each column
        summed = np.sum(mask, axis=0)
        # column where half of the total pixel mass is
        half_sum = np.sum(summed) / 2
        for i, n in enumerate(np.add.accumulate(summed)):
            if n > half_sum:
                return i

def crop(mask):
        mid = find_midpoint_v4(mask)
        y_nonzero, x_nonzero = np.nonzero(mask)
        y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
        x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
        x_dist = max(np.abs(x_lims - mid))
        x_lims = [mid - x_dist, mid+x_dist]
        return mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]
def get_asymmetry(mask):
    # mask = color.rgb2gray(mask)
    scores = []
    for _ in range(6):
        segment = crop(mask)
        (np.sum(segment))
        #scores.append(np.sum(np.logical_xor(segment, np.flip(segment))) / (np.sum(segment)))
        denom = np.sum(segment)
        if denom == 0:
            scores.append(np.nan)  # or maybe 0 or some default value
        else:
            scores.append(np.sum(np.logical_xor(segment, np.flip(segment))) / denom)
        mask = rotate(mask, 30)
    #return sum(scores) / len(scores)
    return np.nanmean(scores)

def processmaskasymmetry(mask_path):
    """
    Processes a single mask image and calculates its asymmetry score.

    Args:
        mask_path (str): Path to the mask image.

    Returns:
        float or str: The asymmetry score, or 'N/A' if the mask is invalid.
    """
    try:
        # Load the mask as a grayscale image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError("Failed to load mask image.")

        # Ensure the mask is binary (convert to 0 and 1)
        _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # Calculate the asymmetry score
        asymmetry_score = get_asymmetry(binary_mask)

        return asymmetry_score

    except Exception as e:
        print(f"⚠️ Skipping bad mask {mask_path} due to error: {e}")
        return 'N/A'


def plot_asymmetry_scores_from_df(df):

    def color_code_asymmetry(score):
        if score < 100:
            return 'green'
        elif score < 200:
            return 'yellow'
        elif score < 300:
            return 'orange'
        else:
            return 'red'

    df['color'] = df['asymmetry_score'].apply(color_code_asymmetry)

    df = df.sort_values('asymmetry_score')

    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(df)), df['asymmetry_score'], color=df['color'])
    plt.xlabel('Asymmetry Score')
    plt.ylabel('Lesions')
    plt.title('Asymmetry Scores')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # legend
    green_patch = mpatches.Patch(color='green', label='Safe (<100)')
    yellow_patch = mpatches.Patch(color='yellow', label='Caution (100-199)')
    orange_patch = mpatches.Patch(color='orange', label='Warning (200-299)')
    red_patch = mpatches.Patch(color='red', label='High Risk (300+)')
    plt.legend(handles=[green_patch, yellow_patch, orange_patch, red_patch], loc='lower right')

    plt.tight_layout()
    plt.show()

    # top 20 lesions with highest asymmetry
    print("\nTop 20 lesions with highest asymmetry:")
    top_20 = df.sort_values('asymmetry_score', ascending=False).head(20)
    print(top_20[['filename', 'asymmetry_score', 'color']])
