import cv2
import numpy as np
from scipy.spatial import ConvexHull

def calculatediameter(mask_path):
    """
    Calculates the maximum diameter of a lesion in a binary mask image using the convex hull.

    Args:
        mask_path (str): Path to the binary mask image file.

    Returns:
        float: The longest Euclidean distance between any two points on the convex hull,
               representing the lesion's maximum diameter.
    """
  
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask file: {mask_path}")


    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)


    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in mask: {mask_path}")
        return 0

    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour[:, 0, :]  


    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

   
    max_distance = 0
    for i in range(len(hull_points)):
        for j in range(i + 1, len(hull_points)):
            dist = np.linalg.norm(hull_points[i] - hull_points[j])
            if dist > max_distance:
                max_distance = dist

    return max_distance