import os
import re
import math
import time
import glob
import random
import zipfile
import pydicom
import cv2
#import matplotlib.patches as patches
#import collections
import numpy as np
import pandas as pd
#import seaborn as sns
from tqdm import tqdm
from PIL import Image, ImageDraw

current_directory = os.getcwd()
print(current_directory)

patient_list=[i for i in range(1,923)]
len(patient_list)

def convert_dicom_to_jpg_for_patients(mapping_csv_path, dicom_root_dir, output_dir, patient_list):
    """
    Converts DICOM images to JPG for specified patients and saves them in a structured directory.
    Converts grayscale images to RGB by duplicating the grayscale channel three times.
    Always normalizes pixel values and saves JPG images without leading zeros in the filenames.
    Checks if the images are initially RGB; if so, it announces the image is not solely grayscale.

    :param mapping_csv_path: Path to the CSV with DICOM file mappings.
    :param dicom_root_dir: Root directory for DICOM images in classic format.
    :param output_dir: Output directory for converted JPG images.
    :param patient_list: List of patient numbers to process.
    """

    mapping_df = pd.read_csv(mapping_csv_path, low_memory=False)
    pattern = re.compile(r'Breast_MRI_(\d+)/((pre|post_1|post_2|post_3|post_4))/Breast_MRI_\d+_(pre|post_1|post_2|post_3|post_4)_0*(\d+)\.dcm')

    for index, row in tqdm(mapping_df.iterrows(), total=mapping_df.shape[0]):
        original_path = row['original_path_and_filename']
        match = pattern.search(original_path)
        if match:
            patient_number = int(match.group(1))
            if patient_number in patient_list:
                sequence_type = match.group(2).replace("_", "").replace("-", "")
                slice_number = int(match.group(5))
                classic_path = row['classic_path']

                full_path_to_dicom = os.path.join(dicom_root_dir, f"{classic_path[:-7]}{slice_number:02d}.dcm")
                if not os.path.exists(full_path_to_dicom):
                    full_path_to_dicom = os.path.join(dicom_root_dir, f"{classic_path[:-7]}{slice_number:03d}.dcm")

                if not os.path.exists(full_path_to_dicom):
                    print(f"Error: DICOM file not found for patient {patient_number}, slice {slice_number}")
                    continue

                try:
                    dcm_img = pydicom.dcmread(full_path_to_dicom)
                    if 'PixelData' in dcm_img:
                        pixels = dcm_img.pixel_array
                        if dcm_img.PhotometricInterpretation == "MONOCHROME1":
                            pixels = np.invert(pixels)
                        elif dcm_img.PhotometricInterpretation != "MONOCHROME2":
                            continue  # Only proceed if image is grayscale

                        # Normalize pixel values
                        pixels = pixels - np.min(pixels)
                        pixels = pixels / np.max(pixels) * 255 if np.max(pixels) != 0 else pixels
                        pixels = pixels.astype(np.uint8)

                        # Convert grayscale to RGB by duplicating the channel three times
                        pixels_rgb = np.stack((pixels,)*3, axis=-1)

                        img = Image.fromarray(pixels_rgb)
                        patient_output_dir = os.path.join(output_dir, f'patient{patient_number}', sequence_type)
                        os.makedirs(patient_output_dir, exist_ok=True)
                        output_path = os.path.join(patient_output_dir, f'{slice_number}.jpg')  # No leading zeros for output filenames
                        img.save(output_path)
                except Exception as e:
                    print(f"Error processing DICOM for patient {patient_number}: {e}")

mapping_csv_path = 'J:/Breast-Cancer-MRI-filepath_filename-mapping.csv'
dicom_root_dir = 'J:/manifest-1712455059978/'  # DICOM images root directory
output_dir = 'J:/patient_images'  # Output directory for JPG images
convert_dicom_to_jpg_for_patients(mapping_csv_path, dicom_root_dir, output_dir, patient_list)
#J is my external hard drive, which provides ample storage space for the extensive Duke DCE MRI image dataset.
