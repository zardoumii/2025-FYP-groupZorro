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
from util.feature_B import processmaskborderirregularity
from util.feature_C import get_multicolor_rate
from util.feature_D import get_diameter


from os.path import join

import matplotlib.pyplot as plt

"""Adjust paths Bbelow according to your directory structure"""
# Directory where all the masked images are stored
# Masksfolder = '/Users/youssefzardoumi/Desktop/Masked'
# outputA = '/Users/youssefzardoumi/Desktop/ITU/Vscode/ProjectsinData/2025-FYP-Final/result/asymmetryscores.csv'

Masksfolder = r'C:\Users\anial\Desktop\Study 2nd Semester\Projects in Data Scince\project\lesion_masks'
outputA = r'C:\Users\anial\Desktop\Study 2nd Semester\Projects in Data Scince\project\2025-FYP-Final\result\asymmetryscores2.csv'
outputB = r'C:\Users\anial\Desktop\Study 2nd Semester\Projects in Data Scince\project\2025-FYP-Final\result\borderirregularityscores.csv'
outputC = r'C:\Users\anial\Desktop\Study 2nd Semester\Projects in Data Scince\project\2025-FYP-Final\result\colorvariationscores.csv'
outputD = r'C:\Users\anial\Desktop\Study 2nd Semester\Projects in Data Scince\project\2025-FYP-Final\result\diameterscores.csv'


# inspectborders(
#     csv_path="/Users/youssefzardoumi/Desktop/dataset.csv",
#     mask_folder="/Users/youssefzardoumi/Desktop/Masked",
#     save_folder="/Users/youssefzardoumi/Desktop/manual_inspection",
#     threshold=0.05  # 5% of border touching the image edge
# )
'''

def Asymmetryforall(Masked_path, outputA):
    """
    Processes all mask files in the given folder and calculates their asymmetry scores.
    """
    
    files = [f for f in os.listdir(Masked_path) if f.endswith('.png')]

    results = []
    
    for x in files:
        file_path = join(Masked_path, str(x))
        try:
            # Attempt to calculate the asymmetry score
            asymmetry_score = processmaskasymmetry(file_path)
        except Exception as e:
            # If an error occurs, log it and set the score to 'N/A'
            print(f"Error processing file {x}: {e}")
            asymmetry_score = 'N/A'
            
        results.append({'filename': x, 'asymmetry_score': asymmetry_score})

    resultsdf = pd.DataFrame(results, columns=['filename', 'asymmetry_score'])
    resultsdf.to_csv(outputA, index=False)
    print(f"Asymmetry scores saved to: {outputA}")

Asymmetryforall(Masksfolder, outputA)



def BorderIrregularityForAll(Masked_path, output_path):
    """
    Processes all mask files and computes border irregularity features.
    Saves results to a CSV.
    """
    files = [f for f in os.listdir(Masked_path) if f.endswith('.png')]
    results = []

    for x in files:
        file_path = join(Masked_path, str(x))
        try:
            streaks, compactness, convexity = processmaskborderirregularity(file_path)
        except Exception as e:
            print(f"Error processing file {x}: {e}")
            streaks, compactness, convexity = 'N/A', 'N/A', 'N/A'

        results.append({
            'filename': x,
            'streaks': streaks,
            'compactness': compactness,
            'convexity': convexity
        })

    resultsdf = pd.DataFrame(results, columns=['filename', 'streaks', 'compactness', 'convexity'])
    resultsdf.to_csv(output_path, index=False)
    print(f"Border irregularity scores saved to: {output_path}")
    
BorderIrregularityForAll(Masksfolder, outputB)


def ColorVariationForAll(img_folder, mask_folder, output_path, n_clusters=5):
    """
    Calculates color variation scores for all lesions and saves them to CSV.
    """
    files = [f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    results = []

    for file in files:
        img_path = join(img_folder, file)
        mask_path = join(mask_folder, file)  # assumes same filename

        try:
            score = get_multicolor_rate(img_path, mask_path, n_clusters)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            score = 'N/A'

        results.append({'filename': file, 'color_variation_score': score})

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Color variation scores saved to {output_path}")
    
#ColorVariationForAll(img_folder, Masksfolder, outputC)

'''

def DiameterForAll(Masked_path, outputD):
    """
    Processes all mask files in the given folder and calculates their diameter.
    """
    files = [f for f in os.listdir(Masked_path) if f.endswith('.png')]

    results = []

    for x in files:
        file_path = os.path.join(Masked_path, str(x))
        try:
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            diameter = get_diameter(binary_mask)
        except Exception as e:
            print(f"Error processing file {x}: {e}")
            diameter = 'N/A'

        results.append({'filename': x, 'diameter': diameter})

    resultsdf = pd.DataFrame(results, columns=['filename', 'diameter'])
    resultsdf.to_csv(outputD, index=False)
    print(f"Diameter scores saved to: {outputD}")

DiameterForAll(Masksfolder, outputD)



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
