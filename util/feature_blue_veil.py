import cv2
import numpy as np

def blue_veil_percentage(image_path, mask_path):
    """
    Calculates the percentage of lesion area exhibiting blue-white veil characteristics.

    Args:
        image_path (str): Path to the original RGB lesion image.
        mask_path (str): Path to the binary mask image.

    Returns:
        float or None: Percentage of veil-like pixels in the lesion area,
                       or None if images are invalid or lesion area is empty.
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Could not load {image_path} or {mask_path}")
        return None

    mask = (mask > 0).astype(np.uint8)
    lesion_area = np.sum(mask)

    if lesion_area == 0:
        return None

    height, width, _ = image.shape
    count = 0

    for y in range(height):
        for x in range(width):
            if mask[y, x] == 1:
                b, g, r = image[y, x]
                if b > 60 and (r - 46 < g) and (g < r + 15):
                    count += 1

    blue_veil_percent = (count / lesion_area) * 100
    return blue_veil_percent
