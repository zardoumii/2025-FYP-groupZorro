import sys
from os.path import join
import os
import cv2
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from util.img_util import readImageFile, saveImageFile, ImageDataLoader, isbordertouching, inspectborders
from util.inpaint_util import removeHair
from util.feature_A import processmaskasymmetry
from util.feature_B import measureborderirregularity
from util.feature_C import colorvariationscore
from util.feature_D import calculatediameter
from os.path import join

import matplotlib.pyplot as plt

"""Adjust paths Bbelow according to your directory structure"""
# Directory where all the images are stored
Imagefolder = '/imgs_part_1'
# Directory where all the masked images are stored
Masksfolder = '/Masked'
# Directory where Features results will be saved
outputA = '/asymmetryscores.csv'
outputB = '/irregularityscores.csv'
outputC = '/colorvariation.csv'
outputD = '/diameterscores.csv'

# inspectborders(
#     csv_path="//dataset.csv",
#     mask_folder="/Masked",
#     save_folder="/manual_inspection",
#     threshold=0.05  # 5% of border touching the image edge
# )

def Asymmetryforall(Masked_path, outputA):
    """
    Processes all mask files in the given folder and calculates their asymmetry scores.
    """
    
    files = [f for f in os.listdir(Masked_path) if f.endswith('.png')]

    results = []
    
    for x in files:
        file_path = join(Masked_path, str(x))
        try:
            asymmetry_score = processmaskasymmetry(file_path)
        except Exception as e:
            asymmetry_score = 'N/A'
            
        results.append({'filename': x, 'asymmetry_score': asymmetry_score})

    resultsdf = pd.DataFrame(results, columns=['filename', 'asymmetry_score'])
    resultsdf.to_csv(outputA, index=False)
    print(f"Asymmetry scores saved to: {outputA}")
    
def IrregularityForAll(masked_path, output_csv):
    """
    Processes all mask files in the given folder and calculates their border irregularity scores.
    Saves the results to a CSV file.
    """
    files = [f for f in os.listdir(masked_path) if f.endswith('.png')]

    results = []

    for filename in files:
        file_path = join(masked_path, filename)
        try:
            score = measureborderirregularity(file_path)
        except Exception as e:
            score = 'N/A'

        results.append({'filename': filename, 'irregularity_score': score})

    df = pd.DataFrame(results, columns=['filename', 'irregularity_score'])
    df.to_csv(output_csv, index=False)
    print(f"Irregularity scores saved to: {output_csv}")
    
    
    
    
def ColorVariationForAll(image_folder, mask_folder, output_csv):
    """
    Computes color variation scores for all image-mask pairs in specified folders and stores results.
    
    """
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]
    results = []

    for mask_filename in mask_files:
       
        base_name = mask_filename.replace('_mask', '')
        image_filename = base_name 

        image_path = os.path.join(image_folder, image_filename)
        mask_path = os.path.join(mask_folder, mask_filename)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            score = 'N/A'
        else:
            try:
                score = colorvariationscore(image_path, mask_path)
            except Exception as e:
                print(f"Error processing {mask_filename}: {e}")
                score = 'N/A'

        results.append({'filename': image_filename, 'color_variation_score': score})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Color variation scores saved to: {output_csv}")
    
def DiameterForAll(mask_folder, output_csv):
    """
    Calculates the maximum diameter for all mask images in a folder and saves the results to a CSV file.
    """
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]
    results = []

    for mask_filename in mask_files:
        mask_path = os.path.join(mask_folder, mask_filename)

        try:
            diameter = calculatediameter(mask_path)
        except Exception as e:
            print(f"Error processing {mask_filename}: {e}")
            diameter = 'N/A'

        results.append({'filename': mask_filename, 'diameter_pixels': diameter})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Diameter scores saved to: {output_csv}")

# Asymmetryforall(Masksfolder, outputA)
# IrregularityForAll(Masksfolder,outputB)
# ColorVariationForAll(Imagefolder, Masksfolder, outputC)
DiameterForAll(Masksfolder,outputD)


# files=ImageDataLoader('/Users/youssefzardoumi/Desktop/dataset.csv')

# for x in files.file_list:
#     file_path = join('/Users/youssefzardoumi/Desktop/imgs_part_1/'+str(x))
#     save_dir = '/Users/youssefzardoumi/Desktop/Masked'

#     # read an image file
#     img_rgb, img_gray = readImageFile(file_path)

#     # apply hair removal
#     blackhat, thresh, img_out = removeHair(img_rgb, img_gray, kernel_size=7, threshold=3)

#     # plot the images
#     plt.figure(figsize=(15, 10))

#     # save the output image
#     save_file_path = join(save_dir, str('hairless_'+str(x)))
#     saveImageFile(img_out, save_file_path)


# def main(csv_path, save_path):
#     # load dataset CSV file
#     data_df = pd.read_csv(csv_path)

#     # select only the baseline features.
#     baseline_feats = [col for col in data_df.columns if col.startswith("feat_")]
#     x_all = data_df[baseline_feats]
#     y_all = data_df["label"]

#     # split the dataset into training and testing sets.
#     x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=42)

#     # train the classifier (using logistic regression as an example)
#     clf = LogisticRegression(max_iter=1000, verbose=1)
#     clf.fit(x_train, y_train)

#     # test the trained classifier
#     y_pred = clf.predict(x_test)
#     acc = accuracy_score(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)
#     print("Test Accuracy:", acc)
#     print("Confusion Matrix:\n", cm)

#     # write test results to CSV.
#     result_df = data_df.loc[x_test.index, ["filename"]].copy()
#     result_df['true_label'] = y_test.values
#     result_df['predicted_label'] = y_pred
#     result_df.to_csv(save_path, index=False)
#     print("Results saved to:", save_path)


# if __name__ == "__main__":
#     csv_path = "./dataset.csv"
#     save_path = "./result/result_baseline.csv"

#     main(csv_path, save_path)
