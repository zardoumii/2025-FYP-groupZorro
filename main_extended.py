import sys
from os.path import join
import os
import cv2
import pandas as pd
import numpy as np
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
from util.Classifier import run_evaluation, convert_to_serializable, main_train
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import ShuffleSplit
from imblearn.combine import SMOTEENN
from collections import Counter
import json
from sklearn.model_selection import KFold
from sklearn.base import clone






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
    



def extract_features(csv_path, save_path):
    """ Main function for feature extraction and demo """

    global Imagefolder  
    admin = 1
    # 0. Hair removal
    Imagefolder = hairratioandremoval(Imagefolder, filenames, output_folder, admin)


    # 1. Asymmetry feature extraction
    if not os.path.exists(os.path.join(output_folder, 'asymmetry_scores.csv')):
        print("Asymmetry CSV not found. Computing asymmetry features...")
        Asymmetryforall_fast(Masksfolder, output_folder)
    else:
        print("Asymmetry CSV already exists. Skipping recomputation.")




    # 2. Border Irregularity extraction
    if not os.path.exists(os.path.join(output_folder, 'irregularity_scores.csv')):
        print("Irregularity CSV not found. Computing irregularity features...")
        IrregularityForAll(Masksfolder, output_folder)
    else:
        print("Irregularity CSV already exists. Skipping recomputation.")


    # 3. Color Variation extraction
    if not os.path.exists(os.path.join(output_folder, 'color_variation_scores.csv')):
        print("Color Variation CSV not found. Computing color features...")
        ColorVariationForAll(Imagefolder, Masksfolder, output_folder)
    else:
        print("Color Variation CSV already exists. Skipping recomputation.")


    # 4. Blue Veil Feature extraction
    if not os.path.exists(os.path.join(output_folder, 'blue_veil_scores.csv')):
        print("Blue Veil CSV not found. Computing blue veil features...")
        BlueWhiteVeilForAll(Imagefolder, Masksfolder, output_folder)
    else:
        print("Blue Veil CSV already exists. Skipping recomputation.")



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
    
    """Adjust paths below according to your directory structure eg. /Users/youssefzardoumi/Desktop/ProjectsinData/2025-FYP-Final"""
    """"""
    path = 'INSERT HERE/2025-FYP-Final'  # Base directory path - removed trailing slash
    """"""
    # Define all paths relative to the base path - removed extra forward slashes
    output_folder = os.path.join(path, 'result')
    metadata_path = os.path.join(path, 'data', 'metadata.csv')  # Moved to base directory
    Masksfolder = os.path.join(path, 'data', 'Masks')
    Imagefolder = os.path.join(path, 'data', 'images')
    dataset_path = os.path.join(path, 'result', 'dataset.csv')
    filenames = os.path.join(path, 'dataset.csv')
    
    # extract Features 
    extract_features(None, None)  
    
    # merge Features
    merge_features(output_folder, metadata_path)
    
    # train model
    main_train(dataset_path)