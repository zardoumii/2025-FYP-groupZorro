import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
from math import sqrt

def get_diameter(mask):
    """
    Calculates the diameter of the lesion based on the binary mask.
    Uses the maximum distance between any two boundary points (longest axis).
    
    Args:
        mask (2D np.array): Binary mask of the lesion.
    
    Returns:
        float: Estimated diameter in pixels.
    """
    # Label connected regions in mask
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)

    if len(props) == 0:
        return np.nan  # No lesion detected

    region = props[0]

    # Get the coordinates of the lesion
    coords = region.coords

    # Compute pairwise distances (brute-force)
    max_distance = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist > max_distance:
                max_distance = dist

    return max_distance
