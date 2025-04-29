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
from os.path import join
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from util.img_util import readImageFile, saveImageFile, ImageDataLoader, isbordertouching, inspectborders
from util.inpaint_util import removeHair
from util.feature_A import processmaskasymmetry
from util.feature_A import plot_asymmetry_scores_from_df
from util.feature_B import measureborderirregularity
from util.feature_C import colorvariationscore
from util.feature_D import calculatediameter
from util.feature_blue_veil import BlueWhiteVeilForAll
from util.merge_features import merge_features

"""Adjust paths below according to your directory structure"""
# Directory where all the images are stored
Imagefolder = r'C:\Users\DaraGeorgieva\Documents\zr7vgbcyr2-1\images'
# Directory where all the masked images are stored
Masksfolder = r"C:\Users\DaraGeorgieva\Documents\lesion_masks\lesion_masks"
# Directory where Features results will be saved
outputA = outputA = r'C:\Users\DaraGeorgieva\Documents\2025-FYP-Final\2025-FYP-Final\result\asymmetryscores.csv'
outputB = r'C:\Users\DaraGeorgieva\Documents\2025-FYP-Final\2025-FYP-Final\result\borderscores.csv'
outputC = r'C:\Users\DaraGeorgieva\Documents\2025-FYP-Final\2025-FYP-Final\result\colorvariancescores.csv'
outputBV = r'C:\Users\DaraGeorgieva\Documents\2025-FYP-Final\2025-FYP-Final\result\blue_white_veil.csv'
outputD = r'C:\Users\DaraGeorgieva\Documents\2025-FYP-Final\2025-FYP-Final\result\diameter.csv'



# inspectborders(
#     csv_path="//dataset.csv",
#     mask_folder="/Masked",
#     save_folder="/manual_inspection",
#     threshold=0.05  # 5% of border touching the image edge
# )

def perform_hair_removal_if_needed(Imagefolder, metadata_path):
    save_dir = os.path.join(Imagefolder, 'hairless')

    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"Hairless folder already exists at {save_dir}. Skipping hair removal.")
    else:
        print("Applying hair removal to images...")
        files = ImageDataLoader(metadata_path)
        os.makedirs(save_dir, exist_ok=True)

        for filename in files.file_list:
            img_path = os.path.join(Imagefolder, filename)
            try:
                img_rgb, img_gray = readImageFile(img_path)
                blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)))
                _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
                img_out = cv2.inpaint(img_rgb, thresh, 1, cv2.INPAINT_TELEA)

                save_path = os.path.join(save_dir, filename)
                saveImageFile(img_out, save_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return save_dir

# def Asymmetryforall(Masked_path, outputA):
#     """
#     Processes all mask files in the given folder and calculates their asymmetry scores.
#     """
    
#     files = [f for f in os.listdir(Masked_path) if f.endswith('.png')]

#     results = []
    
#     for x in files:
#         file_path = join(Masked_path, str(x))
#         try:
#             asymmetry_score = processmaskasymmetry(file_path)
#         except Exception as e:
#             asymmetry_score = 'N/A'
            
#         results.append({'filename': x, 'asymmetry_score': asymmetry_score})

#     resultsdf = pd.DataFrame(results, columns=['filename', 'asymmetry_score'])
#     resultsdf.to_csv(outputA, index=False)
#     print(f"Asymmetry scores saved to: {outputA}")

def Asymmetryforall_fast(Masked_path, output_csv, max_workers=4):
    """
    Fast version of processing masks using multiprocessing.
    """
    # Only real mask files (no weird Mac files)
    files = [f for f in os.listdir(Masked_path) if f.endswith('.png') and not f.startswith('._')]
    files = [join(Masked_path, f) for f in files]  # Full path

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for filepath, score in zip(files, executor.map(processmaskasymmetry, files)):
            clean_filename = os.path.basename(filepath).replace('_mask', '')
            results.append({'filename': clean_filename, 'asymmetry_score': score})


    resultsdf = pd.DataFrame(results)
    resultsdf.to_csv(output_csv, index=False)
    print(f"Asymmetry scores saved to: {output_csv}")
    

    
def IrregularityForAll(masked_path, output_csv):
    """
    Processes all mask files in the given folder and calculates their border irregularity scores.
    Saves the results to a CSV file.
    """

    mask_files = [f for f in os.listdir(masked_path) if f.endswith('.png') and not f.startswith('._')]

    results = []

    for filename in mask_files:
        file_path = join(masked_path, filename)
        try:
            score = measureborderirregularity(file_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            score = 'N/A'

        clean_filename = filename.replace('_mask', '')
        results.append({'filename': clean_filename, 'irregularity_score': score})


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


def main(csv_path, save_path):
    """ Main function for feature extraction and demo """

    global Imagefolder  # Add this to modify Imagefolder globally

    # 0. Hair removal
    Imagefolder = perform_hair_removal_if_needed(Imagefolder, metadata_path)


    # 1. Asymmetry feature extraction
    if not os.path.exists(outputA):
        print("Asymmetry CSV not found. Computing asymmetry features...")
        Asymmetryforall_fast(Masksfolder, outputA)
    else:
        print("Asymmetry CSV already exists. Skipping recomputation.")

    # get the cool plot of asymmetry scores
    # df = pd.read_csv(outputA)
    # plot_asymmetry_scores_from_df(df)


    # 2. Border Irregularity extraction
    if not os.path.exists(outputB):
        print("Irregularity CSV not found. Computing irregularity features...")
        IrregularityForAll(Masksfolder, outputB)
    else:
        print("Irregularity CSV already exists. Skipping recomputation.")

    # TO DO: Add a pretty plot of irregularity scores
    # df = pd.read_csv(outputB)
    # plot_irregularity_scores_from_df(df)

    # 3. Color Variation extraction
    if not os.path.exists(outputC):
        print("Color Variation CSV not found. Computing color features...")
        ColorVariationForAll(Imagefolder, Masksfolder, outputC)
    else:
        print("Color Variation CSV already exists. Skipping recomputation.")
    # TO DO: Add a pretty plot of color variation scores
    # df = pd.read_csv(outputC)
    # plot_color_variation_scores_from_df(df)

    # 4. Blue Veil Feature extraction
    if not os.path.exists(outputBV):
        print("Blue Veil CSV not found. Computing blue veil features...")
        BlueWhiteVeilForAll(Imagefolder, Masksfolder, outputBV)
    else:
        print("Blue Veil CSV already exists. Skipping recomputation.")
    # TO DO: Add a pretty plot of blue veil scores maybe with different shades of blue
    # df = pd.read_csv(outputBV)
    # plot_blue_veil_scores_from_df(df)


def merge_metadata(final_dataset_path, metadata_path):
    dataset = pd.read_csv(final_dataset_path)
    metadata = pd.read_csv(metadata_path)

    merged = dataset.merge(metadata[['img_id', 'diagnostic']], left_on='filename', right_on='img_id', how='left')
    merged['label'] = merged['diagnostic'].apply(lambda x: 1 if x == 'MEL' else 0)
    merged = merged.drop(columns=['img_id', 'diagnostic'])
    merged.to_csv(final_dataset_path, index=False)

    print(f"Metadata merged and label column added to {final_dataset_path}")



if __name__ == "__main__":
    output_folder = r'C:\Users\DaraGeorgieva\Documents\2025-FYP-Final\2025-FYP-Final\result'
    final_dataset_path = r'C:\Users\DaraGeorgieva\Documents\2025-FYP-Final\2025-FYP-Final\result\dataset.csv'
    model_result_path = r'C:\Users\DaraGeorgieva\Documents\2025-FYP-Final\2025-FYP-Final\result\result_baseline.csv'
    metadata_path = r'C:\Users\DaraGeorgieva\Documents\zr7vgbcyr2-1\metadata.csv'

    
    # extract Features 
    main(None, None)  

    # merge Features
    merge_features(output_folder, final_dataset_path)

    # merge metadata
    metadata_path = r'C:\Users\DaraGeorgieva\Documents\zr7vgbcyr2-1\metadata.csv'
    merge_metadata(final_dataset_path, metadata_path)