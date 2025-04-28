import random
import pandas as pd
import cv2
import numpy as np
import os
import shutil

def isbordertouching(mask_path, threshold=0.2):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not load: {mask_path}")
        return False

    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    contour = max(contours, key=cv2.contourArea)
    border_points = contour[:, 0, :]  # (N, 2)

    outside_count = np.sum(
        (border_points[:, 0] <= 1) |
        (border_points[:, 0] >= w - 2) |
        (border_points[:, 1] <= 1) |
        (border_points[:, 1] >= h - 2)
    )
    total_count = len(border_points)
    percent_outside = outside_count / total_count if total_count else 0

    return percent_outside >= threshold

def inspectborders(csv_path, mask_folder, save_folder, threshold=0.2):
    os.makedirs(save_folder, exist_ok=True)
    df = pd.read_csv(csv_path, header=None)
    mask_filenames = df[0].tolist()

    for mask_name in mask_filenames:
        mask_path = os.path.join(mask_folder, mask_name)
        if isbordertouching(mask_path, threshold=threshold):
            save_path = os.path.join(save_folder, mask_name)
            shutil.copy(mask_path, save_path)
            print(f"Saved for manual review: {mask_name}")

def readImageFile(file_path):
    # read image as an 8-bit array
    img_bgr = cv2.imread(file_path)

    # convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # convert the original image to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return img_rgb, img_gray


def saveImageFile(img_rgb, file_path):
    try:
        # convert BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # save the image
        success = cv2.imwrite(file_path, img_bgr)
        if not success:
            print(f"Failed to save the image to {file_path}")
        return success

    except Exception as e:
        print(f"Error saving the image: {e}")
        return False


class ImageDataLoader:
    def __init__(self, csv_path, shuffle=False, transform=None):
        """
        Initializes the ImageDataLoader to load image file names from a CSV file.

        Args:
            csv_path (str): Path to the CSV file containing the 'img_id' column.
            shuffle (bool): Whether to shuffle the file list. Default is False.
            transform (callable, optional): A function/transform to apply to the images. Default is None.
        """
        self.csv_path = csv_path
        self.shuffle = shuffle
        self.transform = transform
        self.file_list = []

        # Load the CSV file
        data_df = pd.read_csv(self.csv_path)

        # Ensure the 'img_id' column exists
        if 'img_id' not in data_df.columns:
            raise ValueError("The CSV file must contain an 'img_id' column.")

        # Extract the file list from the 'img_id' column
        self.file_list = data_df['img_id'].tolist()

        # Raise an error if no files are found
        if not self.file_list:
            raise ValueError("No image files found in the CSV file.")

        # Shuffle the file list if required
        if self.shuffle:
            random.shuffle(self.file_list)

        # Get the total number of batches
        self.num_batches = len(self.file_list)

    def __len__(self):
        """
        Returns the total number of image files.
        """
        return self.num_batches

    def __iter__(self):
        """
        Returns an iterator over the file list.
        """
        for file_name in self.file_list:
            yield file_name
