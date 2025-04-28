import pandas as pd
import os

def merge_features(output_folder, final_csv_path):
    """
    Merges all feature CSVs and saves a final dataset.
    """
    asymmetry_path = os.path.join(output_folder, 'asymmetryscores.csv')
    irregularity_path = os.path.join(output_folder, 'borderscores.csv')
    colorvariation_path = os.path.join(output_folder, 'colorvariancescores.csv')
    bluewhiteveil_path = os.path.join(output_folder, 'blue_white_veil.csv')  

    df_asym = pd.read_csv(asymmetry_path)
    df_irreg = pd.read_csv(irregularity_path)
    df_color = pd.read_csv(colorvariation_path)
    df_bluewhite = pd.read_csv(bluewhiteveil_path)

    # merge features
    merged_df = df_asym.merge(df_irreg, on='filename')
    merged_df = merged_df.merge(df_color, on='filename')
    merged_df = merged_df.merge(df_bluewhite, on='filename')

    merged_df = merged_df.replace('N/A', pd.NA)
    merged_df = merged_df.fillna(0)

    merged_df = merged_df.reset_index(drop=True)
    merged_df.to_csv(final_csv_path, index=False)

    print(f"\n Final dataset saved to: {final_csv_path}")
