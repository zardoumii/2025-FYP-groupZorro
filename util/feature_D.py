import cv2
import numpy as np
import itertools

def calculatediameter(mask_path):
    """
    Calculates the maximum diameter of a lesion in a binary mask image.

    Args:
        mask_path (str): Path to the binary mask image file.

    Returns:
        float: The longest Euclidean distance between any two points on the largest contour,
               representing the lesion's maximum diameter.
    """
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask file: {mask_path}")

    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0

    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour[:, 0, :]

    max_distance = 0
    for p1, p2 in itertools.combinations(points, 2):
        dist = np.linalg.norm(p1 - p2)
        if dist > max_distance:
            max_distance = dist

    return max_distance