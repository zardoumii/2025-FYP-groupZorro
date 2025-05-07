import os
import numpy as np
from util.img_util import readImageFile, saveImageFile, ImageDataLoader
from util.inpaint_util import removeHair


import os
import numpy as np
import csv

def hairratioandremoval(Imagefolder, metadata_path):
    save_dir = os.path.join(Imagefolder, 'hairless')
    csv_path = os.path.join('/Users/youssefzardoumi/Desktop/ITU/Vscode/ProjectsinData/2025-FYP-Final/result/haircoverageratios.csv')

    if os.path.exists(save_dir) and len(os.listdir(save_dir)) and os.path.exists(csv_path) > 0:
        print(f"Hairless folder already exists at {save_dir}. Skipping hair removal.")
    else:
        print("Applying hair removal to images...")
        files = ImageDataLoader(metadata_path)
        os.makedirs(save_dir, exist_ok=True)

        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'hair_coverage_ratio'])

            for filename in files.file_list:
                img_path = os.path.join(Imagefolder, filename)
                try:
                    img_rgb, img_gray = readImageFile(img_path)
                    blackhat, whitehat, combined_mask, img_out = removeHair(img_rgb, img_gray, kernel_size=13, black_thresh=100, white_thresh=10)
                    
                    hair_coverage_ratio = np.sum(combined_mask > 0) / combined_mask.size

                    
                    save_path = os.path.join(save_dir, filename)
                    saveImageFile(img_out, save_path)

                    
                    writer.writerow([filename, hair_coverage_ratio])

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    return save_dir