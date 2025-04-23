import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
from math import sqrt

def get_diameter(mask):
    """
    Approximate diameter using the bounding box diagonal.
    Much faster than brute-force distance calculation.
    """
    labeled = label(mask)
    props = regionprops(labeled)

    if not props:
        return np.nan

    region = props[0]
    minr, minc, maxr, maxc = region.bbox
    diameter = np.sqrt((maxr - minr)**2 + (maxc - minc)**2)
    return diameter

