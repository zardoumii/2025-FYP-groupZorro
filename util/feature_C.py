import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_multicolor_rate(image_path, mask_path, n=5):
    """
    Calculates color variation by clustering colors in the masked lesion region.

    Args:
        image_path (str): Path to the RGB lesion image.
        mask_path (str): Path to the binary mask image.
        n (int): Number of color clusters.

    Returns:
        float: Percentage of dominant colors used to describe the lesion (as a proxy for variation).
    """
    # Load image and mask
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Create a mask boolean array (1 inside lesion, 0 outside)
    binary_mask = mask > 127

    # Extract only the lesion region's pixels
    lesion_pixels = image[binary_mask]

    if lesion_pixels.shape[0] == 0:
        return np.nan

    # Run KMeans on lesion pixels
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    kmeans.fit(lesion_pixels)

    # Count how many pixels belong to each color cluster
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    proportions = counts / np.sum(counts)

    # A higher entropy-like score = more variation
    variation_score = -np.sum(proportions * np.log(proportions + 1e-10))

    return variation_score
