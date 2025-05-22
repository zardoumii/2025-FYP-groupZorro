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
from util.feature_hair import hairratioandremoval
from util.feature_blue_veil import BlueWhiteVeilForAll
from util.merge_features import merge_features







# inspectborders(
#     csv_path="//dataset.csv",
#     mask_folder="/Masked",
#     save_folder="/manual_inspection",
#     threshold=0.05  # 5% of border touching the image edge
# )



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
    output_csv = os.path.join(output_csv, 'asymmetry_scores.csv')
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
    output_csv = os.path.join(output_csv, 'irregularity_scores.csv')
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
    output_csv = os.path.join(output_csv, 'color_variation_scores.csv')
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
    



def main(csv_path, save_path):
    """ Main function for feature extraction and demo """

    global Imagefolder  # Add this to modify Imagefolder globally

    # 0. Hair removal
    Imagefolder = hairratioandremoval(Imagefolder, metadata_path, output_folder)


    # 1. Asymmetry feature extraction
    if not os.path.exists(os.path.join(output_folder, 'asymmetry_scores.csv')):
        print("Asymmetry CSV not found. Computing asymmetry features...")
        Asymmetryforall_fast(Masksfolder, output_folder)
    else:
        print("Asymmetry CSV already exists. Skipping recomputation.")

    # get the cool plot of asymmetry scores
    # df = pd.read_csv(outputA)
    # plot_asymmetry_scores_from_df(df)


    # 2. Border Irregularity extraction
    if not os.path.exists(os.path.join(output_folder, 'irregularity_scores.csv')):
        print("Irregularity CSV not found. Computing irregularity features...")
        IrregularityForAll(Masksfolder, output_folder)
    else:
        print("Irregularity CSV already exists. Skipping recomputation.")

    # TO DO: Add a pretty plot of irregularity scores
    # df = pd.read_csv(outputB)
    # plot_irregularity_scores_from_df(df)

    # 3. Color Variation extraction
    if not os.path.exists(os.path.join(output_folder, 'color_variation_scores.csv')):
        print("Color Variation CSV not found. Computing color features...")
        ColorVariationForAll(Imagefolder, Masksfolder, output_folder)
    else:
        print("Color Variation CSV already exists. Skipping recomputation.")
    # TO DO: Add a pretty plot of color variation scores
    # df = pd.read_csv(outputC)
    # plot_color_variation_scores_from_df(df)

    # 4. Blue Veil Feature extraction
    if not os.path.exists(os.path.join(output_folder, 'blue_veil_scores.csv')):
        print("Blue Veil CSV not found. Computing blue veil features...")
        BlueWhiteVeilForAll(Imagefolder, Masksfolder, output_folder)
    else:
        print("Blue Veil CSV already exists. Skipping recomputation.")
    # TO DO: Add a pretty plot of blue veil scores maybe with different shades of blue
    # df = pd.read_csv(outputBV)
    # plot_blue_veil_scores_from_df(df)


def merge_metadata(final_dataset_path, metadata_path):
    final_dataset_path = os.path.join(final_dataset_path, 'dataset.csv')
    dataset = pd.read_csv(final_dataset_path)
    metadata = pd.read_csv(metadata_path)

    merged = dataset.merge(metadata[['img_id', 'diagnostic']], left_on='filename', right_on='img_id', how='left')
    merged['label'] = merged['diagnostic'].apply(lambda x: 1 if x == 'MEL' else 0)
    merged = merged.drop(columns=['img_id', 'diagnostic'])
    merged.to_csv(final_dataset_path, index=False)

    print(f"Metadata merged and label column added to {final_dataset_path}")



if __name__ == "__main__":
    
    """Adjust paths below according to your directory structure"""
    output_folder = r'/Users/youssefzardoumi/Desktop/ITU/Vscode/ProjectsinData/2025-FYP-Final/result/'
    model_result_path = r'/Users/youssefzardoumi/Desktop/ITU/Vscode/ProjectsinData/2025-FYP-Final/result/result_baseline.csv'
    metadata_path = r'/Users/youssefzardoumi/Desktop/metadata.csv'
    Masksfolder = r"/Users/youssefzardoumi/Desktop/Masked"
    Imagefolder = r'/Users/youssefzardoumi/Desktop/imgs_part_1'
    
    # extract Features 
    main(None, None)  
    
    # merge Features
    merge_features(output_folder, metadata_path)
