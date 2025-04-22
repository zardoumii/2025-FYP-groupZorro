import cv2
import numpy as np
from scipy.spatial import ConvexHull
from skimage.measure import label, regionprops

def measure_streaks(mask):
    # Count number of disjoint streaks (connected components on border)
    label_image = label(mask)
    return len(regionprops(label_image))

def compactness_score(mask):
    # Compactness = Perimeter^2 / (4π * Area)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.nan

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area == 0:
        return np.nan

    compactness = (perimeter ** 2) / (4 * np.pi * area)
    return compactness

def convexity_score(mask):
    # Convexity = Area / Convex Hull Area
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.nan

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return np.nan

    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return np.nan

    convexity = area / hull_area
    return convexity

def processmaskborderirregularity(mask_path):
    """
    Processes a single mask image and calculates its border irregularity scores.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    streaks = measure_streaks(binary_mask)
    compactness = compactness_score(binary_mask)
    convexity = convexity_score(binary_mask)

    return streaks, compactness, convexity
