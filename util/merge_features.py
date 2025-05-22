import pandas as pd
import os

def merge_features(output_folder, metadata_path):
    """
    Merges all feature CSVs and saves a final dataset.
    Skips any feature files that don't exist.
    """
    final_csv_path = os.path.join(output_folder, 'dataset.csv')
    
    # Define paths and their corresponding dataframe variables
    feature_paths = {
        'asymmetry': os.path.join(output_folder, 'asymmetry_scores.csv'),
        'irregularity': os.path.join(output_folder, 'irregularity_scores.csv'),
        'color_variation': os.path.join(output_folder, 'color_variation_scores.csv'),
        'blue_white_veil': os.path.join(output_folder, 'blue_veil_scores.csv'),
        'hair_ratio': os.path.join(output_folder, 'hair_ratio.csv')
    }
    
    # Load available feature files
    feature_dfs = {}
    for feature_name, path in feature_paths.items():
        if os.path.exists(path):
            try:
                feature_dfs[feature_name] = pd.read_csv(path)
                print(f"Loaded {feature_name} features from {path}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        else:
            print(f"Skipping {feature_name} features - file not found: {path}")
    
    if not feature_dfs:
        raise ValueError("No feature files found to merge!")
    
    # Start with the first dataframe
    first_feature = list(feature_dfs.keys())[0]
    merged_df = feature_dfs[first_feature]
    
    # Merge the rest of the dataframes
    for feature_name, df in list(feature_dfs.items())[1:]:
        try:
            merged_df = merged_df.merge(df, on='filename')
            print(f"Merged {feature_name} features successfully")
        except Exception as e:
            print(f"Error merging {feature_name} features: {e}")
    
    # Handle missing values
    merged_df = merged_df.replace('N/A', pd.NA)
    merged_df = merged_df.fillna(0)
    merged_df = merged_df.reset_index(drop=True)
    
    # Save intermediate result
    merged_df.to_csv(final_csv_path, index=False)
    print(f"Saved intermediate merged features to {final_csv_path}")
    
    # Merge with metadata if available
    try:
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            merged = merged_df.merge(metadata[['img_id', 'diagnostic']], left_on='filename', right_on='img_id', how='left')
            merged['label'] = merged['diagnostic'].apply(lambda x: 1 if x == 'MEL' else 0)
            merged = merged.drop(columns=['img_id', 'diagnostic'])
            merged.to_csv(final_csv_path, index=False)
            print(f"Metadata merged and label column added to {final_csv_path}")
        else:
            print(f"Skipping metadata merge - file not found: {metadata_path}")
    except Exception as e:
        print(f"Error merging metadata: {e}")
        print("Using intermediate result without metadata")
    
    print(f"\nFinal dataset saved to: {final_csv_path}")
