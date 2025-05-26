from os.path import join
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.feature_A import Asymmetryforall_fast
from util.feature_B import IrregularityForAll
from util.feature_C import ColorVariationForAll
from util.feature_hair import hairratioandremoval
from util.feature_blue_veil import BlueWhiteVeilForAll
from util.merge_features import merge_features
from util.Classifier import main_train

'''PATH INFO BELOW'''

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




if __name__ == "__main__":
    # Base directory is the current working directory of the script
    path = os.path.dirname(os.path.abspath(__file__))
    
    # Define all paths relative to the base path using os.path.join
    output_folder = os.path.join(path, 'result')
    metadata_path = os.path.join(path, 'data', 'metadata.csv')
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