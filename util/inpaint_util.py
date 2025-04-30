import cv2
import numpy as np
import matplotlib.pyplot as plt

def removeHair(img_rgb, img_gray, kernel_size=13, black_thresh=100, white_thresh=10):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))


    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    _, black_thresh_mask = cv2.threshold(blackhat, black_thresh, 255, cv2.THRESH_BINARY)


    whitehat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    _, white_thresh_mask = cv2.threshold(whitehat, white_thresh, 255, cv2.THRESH_BINARY)


    combined_mask = cv2.bitwise_or(black_thresh_mask, white_thresh_mask)


    img_out = cv2.inpaint(img_rgb, combined_mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    return blackhat, whitehat, combined_mask, img_out


