import pandas as pd
import os

def merge_features(output_folder, metadata_path):
    """
    Merges all feature CSVs and saves a final dataset.
    """
    final_csv_path = os.path.join(output_folder, 'dataset.csv')
    
    asymmetry_path = os.path.join(output_folder, 'asymmetry_scores.csv')
    irregularity_path = os.path.join(output_folder, 'irregularity_scores.csv')
    colorvariation_path = os.path.join(output_folder, 'color_variation_scores.csv')
    bluewhiteveil_path = os.path.join(output_folder, 'blue_veil_scores.csv')  
    hairratio_path = os.path.join(output_folder, 'hair_ratio.csv')

    df_asym = pd.read_csv(asymmetry_path)
    df_irreg = pd.read_csv(irregularity_path)
    df_color = pd.read_csv(colorvariation_path)
    df_bluewhite = pd.read_csv(bluewhiteveil_path)
    df_hair = pd.read_csv(hairratio_path)

    # merge features
    merged_df = df_asym.merge(df_irreg, on='filename')
    merged_df = merged_df.merge(df_color, on='filename')
    merged_df = merged_df.merge(df_bluewhite, on='filename')
    merged_df = merged_df.merge(df_hair, on='filename')

    merged_df = merged_df.replace('N/A', pd.NA)
    merged_df = merged_df.fillna(0)

    merged_df = merged_df.reset_index(drop=True)
    merged_df.to_csv(final_csv_path, index=False)
    
    
    dataset = pd.read_csv(final_csv_path)
    metadata = pd.read_csv(metadata_path)

    merged = dataset.merge(metadata[['img_id', 'diagnostic']], left_on='filename', right_on='img_id', how='left')
    merged['label'] = merged['diagnostic'].apply(lambda x: 1 if x == 'MEL' else 0)
    merged = merged.drop(columns=['img_id', 'diagnostic'])
    merged.to_csv(final_csv_path, index=False)

    print(f"Metadata merged and label column added to {final_csv_path}")

    print(f"\n Final dataset saved to: {final_csv_path}")
