import numpy as np
import cv2


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