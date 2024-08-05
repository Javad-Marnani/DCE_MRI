##########################################################################################
#####################################   IMPORTS    #######################################
##########################################################################################
import os
import cv2 as cv
import math
import subprocess
import itertools
import numpy as np
from numpy.linalg import norm
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import zipfile
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.ndimage import zoom
import time
import glob
import re
from datetime import datetime
import random
import subprocess
import sklearn
import zipfile
#import pydicom
#import patoolib
import operator
#import mahotas
import matplotlib.patches as patches
import collections
import shutil
import seaborn as sns
from collections import Counter
from matplotlib.patches import Rectangle
from typing import List, Tuple, Optional
import scipy.io as sio
from scipy.ndimage import zoom
import tensorflow as tf
from scipy.stats import t
from random import choice
from statistics import mode
import zipfile
import pickle
#from pyunpack import Archive
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from matplotlib.colors import ListedColormap
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from keras.models import Sequential
from platform import python_version
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from skimage.io import imsave, imread
from sklearn.impute import KNNImputer
from keras.callbacks import EarlyStopping
from IPython.display import Image, display
from sklearn import datasets, metrics, svm
from collections import Counter, defaultdict
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Activation, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.image as mpimg
from sklearn.metrics import (
    f1_score, make_scorer, confusion_matrix, accuracy_score, classification_report,
    precision_score, recall_score, average_precision_score
)
from sklearn.model_selection import (
    GridSearchCV, validation_curve, train_test_split, KFold, cross_val_score,
    StratifiedKFold
)
import torch
from transformers import SamModel, SamProcessor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from transformers import SamModel, SamProcessor
#from vit import ViTEncoderBlock, MLPBlock
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
#from segment_anything import sam_model_registry
from transformers import SamModel, SamProcessor
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

#pip install scikit-image
#!pip install transformers

def extract_zip(zip_file_path, output_directory):
    """
    Extracts specified zip file into a given directory, creating the directory if it does not already exist.
    
    This function ensures the output directory exists or creates it if necessary. It then extracts the contents of the zip file,
    while maintaining the internal directory structure but ignoring any top-level files. It provides feedback on the process,
    including the creation of the output directory and the completion of extraction.

    Parameters:
    - zip_file_path (str): The file path of the zip archive to extract.
    - output_directory (str): The directory where the zip contents will be extracted.

    Notes:
    - The function will create the necessary subdirectories within the output directory.
    - Only files within subfolders are extracted; top-level files are ignored.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # List of all the paths in the zip file
        zip_contents = zip_ref.namelist()

        # Extract each file, ignoring the first part of the path if it's a directory
        for file in zip_contents:
            # Extract only if there is a subfolder (ignoring any top-level files)
            if '/' in file and not file.endswith('/'):
                file_path = os.path.join(output_directory, '/'.join(file.split('/')[1:]))  # Remove the first folder part
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with zip_ref.open(file) as source, open(file_path, 'wb') as target:
                    target.write(source.read())
        print(f"Extracted {zip_file_path} into {output_directory}")

def show_patient_images(base_path, sequence, image_number=50, num_columns=5, batch_size=50):
    """
    Displays a specified image slice from each patient in a sequence, useful for assessing the orientation
    and quality of the images. This function aids in determining whether the slices of a patient are arranged 
    upward or downward and in checking the overall image quality.

    The function batches the display process to manage large numbers of patient images efficiently. For each batch,
    it configures and fills a subplot grid with the specified images from the patients' data directories.

    Parameters:
    - base_path (str): The base directory path where patient directories are located.
    - sequence (str): The specific folder within each patient's directory where images are stored.
    - image_number (int): The specific image number to display from the sequence directory (default is 50).
    - num_columns (int): Number of columns in the subplot grid (default is 5).
    - batch_size (int): Number of patient images to process in each batch (default is 50).

    Notes:
    - The function assumes a total of 922 patients and adjusts the batch processing and subplot configuration
      accordingly.
    - If an image file does not exist, it displays an empty placeholder in its place.
    """
    # Number of patients
    num_patients = 922
    # Total number of batches
    total_batches = (num_patients + batch_size - 1) // batch_size

    for batch in range(total_batches):
        start_index = batch * batch_size
        end_index = min(start_index + batch_size, num_patients)
        current_batch_size = end_index - start_index

        # Prepare the subplot configuration for current batch
        num_rows = (current_batch_size + num_columns - 1) // num_columns
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(15, num_rows * 3))
        axes = axes.flatten()  # Flatten to make indexing easier

        # Iterate through each patient in the current batch
        for i in range(current_batch_size):
            patient_id = f"patient{start_index + i + 1}"
            image_path = os.path.join(base_path, patient_id, sequence, f"{image_number}.jpg")

            # Load and display the image if it exists
            if os.path.exists(image_path):
                img = mpimg.imread(image_path)
                axes[i].imshow(img)
                axes[i].set_title(f"{patient_id}")
            else:
                axes[i].imshow(np.zeros((10, 10, 3), dtype=int))  # Show an empty image if the file is not found

            axes[i].axis('off')  # Turn off axis numbers and ticks

        # Turn off axes for any unused plots in the last row
        for j in range(current_batch_size, len(axes)):
            axes[j].axis('on')

        plt.tight_layout()
        plt.show()

def get_image_sizes(image_directory):
    """
    Iterates over images in the specified directory, checks if each image is square,
    and returns a list of dictionaries with 'patient_number' and 'image_size', sorted by patient number.

    Parameters:
    - image_directory: The path to the directory containing the images.

    Returns:
    - A sorted list of dictionaries, each containing 'patient_number' and 'image_size'.
    """
    image_sizes = []

    # List all files in the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.png')]

    for image_file in image_files:
        # Extract patient_number from the file name (e.g., "1.png")
        patient_number = int(image_file.split('.')[0])
        try:
            with Image.open(os.path.join(image_directory, image_file)) as img:
                width, height = img.size

                # Check if the image is square
                if width != height:
                    print(f"Warning: Image {image_file} is not square.")
                else:
                    image_sizes.append({
                        'patient_number': patient_number,
                        'image_size': width  # Since the image is square, width = height
                    })

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")

    # Sort the list of dictionaries by 'patient_number'
    sorted_image_sizes = sorted(image_sizes, key=lambda x: x['patient_number'])
    return sorted_image_sizes

def count_patient_slices(base_path, num_patients, trim_percentage=0.20):
    """
    Counts the number of JPG slices for each patient in a given directory, applies a trim percentage to calculate
    a trimmed slice count, and returns a list of dictionaries. Each dictionary contains the patient number, the
    original count of slices, and the trimmed count of slices after applying the trim percentage.

    Parameters:
    - base_path: The base directory containing patient folders named 'patientX',
      where X is the patient number from 1 to num_patients. Each patient folder contains
      a subdirectory 'post2' with JPG images.
    - num_patients: The total number of patients (e.g., 150) to process.
    - trim_percentage: The percentage of slices to be removed equally from the start and end of the slice count (default: 20%).

    Returns:
    - A list of dictionaries, where each dictionary has 'patient_number', 'slice_count', and 'trimmed_slice_count' keys.
    """
    patients_slice_counts = []

    # Iterate over each patient number
    for patient_number in range(1, num_patients + 1):
        if patient_number == 468:
            continue  # Skip processing for patient number 468

        # Construct the path to the 'post3' directory for each patient
        post_dir = os.path.join(base_path, f'patient{patient_number}', 'post2')

        # Count the JPG files in the 'post3' directory
        try:
            slice_count = len([file for file in os.listdir(post_dir) if file.endswith('.jpg')])
        except FileNotFoundError:
            print(f"No 'post2' directory found for patient{patient_number}")
            slice_count = 0

        # Calculate trimmed slice count
        trim_amount = int(slice_count * (trim_percentage / 2)) * 2  # Total slices to be trimmed
        trimmed_slice_count = slice_count - trim_amount  # Subtract the trim amount from the total slice count

        # Append the result as a dictionary to the list
        patients_slice_counts.append({
            'patient_number': patient_number,
            'slice_count': slice_count,
            'trimmed_slice_count': trimmed_slice_count
        })

    return patients_slice_counts

def cube_coordinates(annotation_path, num_patients):
    """
    Retrieve the 3D bounding box coordinates for a given patient from a CSV file specified by the annotation_path.

    Parameters:
    - annotation_path: String representing the full path to the CSV file containing annotation boxes.
    - num_patients: The total number of patients to process from the CSV file.

    Returns:
    - A list of dictionaries, each containing the patient_number and the NumPy array of coordinates for the 3D bounding box. 
      The coordinates are in the order [x_min, y_min, x_max, y_max, z_min, z_max], where x_min and x_max are the 
      starting and ending columns, y_min and y_max are the starting and ending rows, and z_min and z_max are the 
      starting and ending slices, respectively. All coordinates are integers.

    Note:
    - The patient_number is expected to be 1-based indexing (first patient is 1, not 0).
    """
    patients_cube_coords = []
    # Load bounding box data from the specified CSV file
    df_box = pd.read_csv(annotation_path)
    for patient_number in range(1, num_patients+1):
        if patient_number == 468:
            continue  # Skip processing for patient number 468
        # Extract the coordinates for the specified patient
        y_min = df_box.loc[patient_number - 1, 'Start Row']
        y_max = df_box.loc[patient_number - 1, 'End Row']
        x_min = df_box.loc[patient_number - 1, 'Start Column']
        x_max = df_box.loc[patient_number - 1, 'End Column']
        z_min = df_box.loc[patient_number - 1, 'Start Slice']
        z_max = df_box.loc[patient_number - 1, 'End Slice']

        # Create a NumPy array with the coordinates
        cube_coords = np.array([x_min, y_min, x_max, y_max, z_min, z_max], dtype=int)
        patients_cube_coords.append({
            'patient_number': patient_number,
            'cube_coordinates': cube_coords
        })
    return patients_cube_coords

def adjust_z_coordinates(cube_coordinates, patient_slice_counts, percentage=0.20):
    """
    Adjusts the z_min and z_max values within the cube_coordinates for each patient based on a specified percentage,
    aimed at trimming slices from both ends of the patient's scan range. This function ensures z_min remains greater
    than 1 and z_max does not exceed the number of slices remaining after the trim.

    Parameters:
    - cube_coordinates: A list of dictionaries, each containing 'patient_number' and 'cube_coordinates' as an array or list.
    - patient_slice_counts: A list of dictionaries, each containing 'patient_number' and 'slice_count', representing the total
    number of slices for each patient.
    - percentage: The percentage of total slices to be removed evenly from the start and the end of the slice range (default: 20%).

    Returns:
    - A list of dictionaries, akin to cube_coordinates, with 'cube_coordinates' for each patient adjusted so that z_min and z_max
    reflect the trimming based on the specified percentage and constraints.
    """
    adjusted_cube_coords = []

    for patient_coords in cube_coordinates:
        patient_num = patient_coords['patient_number']
        # Extract corresponding patient slice count
        slice_count = next((item['slice_count'] for item in patient_slice_counts if item['patient_number'] == patient_num), None)

        if slice_count is not None:
            # Calculate slices to trim from start and end based on percentage
            trim_amount = int(slice_count * (percentage / 2) )

            # Adjust z_min and z_max within constraints
            z_min_adjusted = int(patient_coords['cube_coordinates'][4] - trim_amount)
            z_max_adjusted = math.ceil(patient_coords['cube_coordinates'][5] - trim_amount)

            # Ensure z_min_adjusted is not greater than z_max_adjusted after adjustment
            if z_min_adjusted <=1:
                z_min_adjusted=1
            if z_max_adjusted>=  slice_count-2*trim_amount:
                z_max_adjusted= slice_count-2*trim_amount

            # Update coordinates with adjusted Z values
            updated_coords = patient_coords['cube_coordinates'].copy()
            updated_coords[4], updated_coords[5] = z_min_adjusted, z_max_adjusted

            adjusted_cube_coords.append({'patient_number': patient_num, 'cube_coordinates': updated_coords})

    return adjusted_cube_coords

def scale_bounding_box_for_rotation(adjusted_z_coordinates, patients_to_rotate, sizes):
    """
    Scales the bounding box coordinates for patients whose images have been rotated by 180 degrees,
    based on the image sizes. It adjusts x_min, y_min, x_max, and y_max accordingly.

    Parameters:
    - adjusted_z_coordinates: A list of dictionaries, each containing 'patient_number' and 'cube_coordinates'.
    - patients_to_rotate: A list of patient numbers whose images have been rotated by 180 degrees.
    - sizes: A list of dictionaries, each containing 'patient_number' and 'image_size'.

    Returns:
    - A list of dictionaries with the format of 'patient_number' and 'scaled_cube_coordinates'.
    """
    scaled_coordinates = []
    size_dict = {item['patient_number']: item['image_size'] for item in sizes}

    for patient in adjusted_z_coordinates:
        patient_number = patient['patient_number']
        if patient_number not in patients_to_rotate:
            scaled_coordinates.append(patient)
            continue

        image_size = size_dict.get(patient_number, None)
        if image_size is None:
            print(f"No image size found for patient {patient_number}. Skipping.")
            continue

        cube_coords = patient['cube_coordinates']
        x_min, y_min, x_max, y_max, z_min, z_max = cube_coords

        # Scale the coordinates for rotation
        new_x_min = image_size - x_max
        new_x_max = image_size - x_min
        new_y_min = image_size - y_max
        new_y_max = image_size - y_min

        scaled_cube_coords = [new_x_min, new_y_min, new_x_max, new_y_max, z_min, z_max]
        scaled_coordinates.append({'patient_number': patient_number, 'cube_coordinates': scaled_cube_coords})

    return scaled_coordinates

def plot_masked_images_with_bbox(image_directory, scaled_cube_coordinates, num_columns=8, show=True, batch_size=100):
    """
    Processes images in the specified directory, applying bounding boxes for visualization and 
    generating binary masks (0s and 1s) based on matching patient numbers and coordinates from scaled_cube_coordinates.
    It batches the output to manage large numbers of images, ensuring all images are processed and shown in batches.
    """
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
    patient_images_info = []
    total_images = len(image_files)
    total_batches = (total_images + batch_size - 1) // batch_size

    for batch in range(total_batches):
        if show:
            plt.figure(figsize=(num_columns * 4, (batch_size // num_columns + 1) * 4))
        
        batch_files = image_files[batch * batch_size: (batch + 1) * batch_size]

        for index, image_file in enumerate(batch_files):
            patient_number = int(image_file.split('.')[0])
            match = next((item for item in scaled_cube_coordinates if item['patient_number'] == patient_number), None)
            if match is None:
                print(f"No matching coordinates found for patient {patient_number}")
                continue
            
            cube_coords = match['cube_coordinates']
            img_path = os.path.join(image_directory, image_file)
            img = Image.open(img_path).convert('RGB')  # Ensure image is RGB

            draw_img = img.copy()

            if show:
                plt.subplot((batch_size // num_columns + 1), num_columns, index + 1)
                plt.imshow(draw_img)
                plt.title(f'Patient {patient_number}\nZ-range: {cube_coords[4]}-{cube_coords[5]}', fontsize=18)
                plt.axis('off')

            # Convert draw_img to a numpy array to store in patient_images_info
            processed_image = np.array(draw_img)

            # Threshold the image to binary values (0 or 1)
            binary_mask = (processed_image > 0).astype(np.uint8)  # Elements greater than 0 set to 1, others remain 0

            patient_images_info.append({
                'patient_number': patient_number,
                'mask': binary_mask,  # Storing the binary mask
                'cube_coordinates': cube_coords
            })

        if show:
            plt.tight_layout()
            plt.show()

    return patient_images_info

def data_loader(base_path, patient_slice_counts, scaled_cube_coordinates, patients_to_rotate, num_patients, patients_to_exclude, seq_type1='pre', seq_type2='post1', seq_type3='post2'):
    """
    Reads all images for patients from 1 to num_patients from specified sequence types, optionally rotates images for specified patients,
    and trims slices based on a percentage. Combines images from three sequences to form RGB volumes. 
    The original images are saved after rotation if required, and the RGB volumes are correctly processed from the rotated originals if rotation is applied.

    Parameters:
    - base_path: The base directory containing patient image folders.
    - patient_slice_counts: A list of dictionaries with each containing 'patient_number', 'slice_count', and 'trimmed_slice_count'.
    - scaled_cube_coordinates: A list of dictionaries with each containing 'patient_number' and 'cube_coordinates'.
    - patients_to_rotate: A list of patient numbers whose images have been rotated by 180 degrees.
    - num_patients: The total number of patients to process.
    - seq_type1, seq_type2, seq_type3: The sequence types to process and combine into RGB volumes.

    Returns:
    - A list of dictionaries for all patients, each containing 'patient_number', 'rgb_volume', and 'cube_coordinates'.
    """
    all_patient_results = []

    # Construct a description for the tqdm progress bar using the function arguments for sequence types
    tqdm_description = f"Constructing RGB images using {seq_type1}, {seq_type2}, and {seq_type3}"
    for patient_number in tqdm(range(1, num_patients + 1), desc=tqdm_description):
        if patient_number in patients_to_exclude:
            continue
        seq_dirs = [os.path.join(base_path, f"patient{patient_number}", seq) for seq in [seq_type1, seq_type2, seq_type3]]

        # Check if any sequence directory is missing
        if not all(os.path.exists(seq) for seq in seq_dirs):
            print(f"One or more sequences not available for patient {patient_number}.")
            continue

        bbox = next((item for item in scaled_cube_coordinates if item['patient_number'] == patient_number), None)
        if bbox is None:
            continue

        slice_data = next((item for item in patient_slice_counts if item['patient_number'] == patient_number), None)
        if slice_data is None:
            continue

        total_slices, trimmed_slice_count = slice_data['slice_count'], slice_data['trimmed_slice_count']
        start_trim, end_trim = (total_slices - trimmed_slice_count) // 2, total_slices - (total_slices - trimmed_slice_count) // 2

        rgb_volume = []
        for i in range(start_trim + 1, end_trim + 1):
            channels = []
            for seq_dir in seq_dirs:
                image_path = os.path.join(seq_dir, f"{i}.jpg")
                image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Error loading image: {image_path}")
                    break  # Skip this slice if any channel is missing

                # Apply rotation if needed
                if patient_number in patients_to_rotate:
                    image = cv.rotate(image, cv.ROTATE_180)

                channels.append(image)

            if len(channels) == 3:  # Ensure all three channels are present
                rgb_image = np.stack(channels, axis=-1)
                rgb_volume.append(rgb_image)

        if len(rgb_volume) > 0:
            all_patient_results.append({
                'patient_number': patient_number,
                'rgb_volume': rgb_volume,
                'cube_coordinates': bbox['cube_coordinates']
            })

    return all_patient_results

def data_loader_subtraction(base_path, patient_slice_counts, scaled_cube_coordinates, patients_to_rotate, num_patients):
    """
    Reads all images for patients from 1 to num_patients from specified sequence types, optionally rotates images for specified patients,
    and trims slices based on a percentage. Combines images from three sequences (with subtraction operation) to form RGB volumes.
    The original images are saved after rotation if required, and the RGB volumes are correctly processed from the rotated originals if rotation is applied.

    Parameters:
    - base_path: The base directory containing patient image folders.
    - patient_slice_counts: A list of dictionaries with each containing 'patient_number', 'slice_count', and 'trimmed_slice_count'.
    - scaled_cube_coordinates: A list of dictionaries with each containing 'patient_number' and 'cube_coordinates'.
    - patients_to_rotate: A list of patient numbers whose images have been rotated by 180 degrees.
    - num_patients: The total number of patients to process.

    Returns:
    - A list of dictionaries for all patients, each containing 'patient_number', 'rgb_volume', and 'cube_coordinates'.
    """
    all_patient_results = []

    for patient_number in tqdm(range(1, num_patients+1)):  # Adjusted to use num_patients
        if patient_number in patients_to_exclude:
            continue
        # Define sequence directories including the fourth potential sequence
        seq_dirs = [os.path.join(base_path, f"patient{patient_number}", seq) for seq in ['pre', 'post1', 'post2', 'post3']]

        # Check if any sequence directory is missing, and adjust sequences accordingly
        seq_availability = [os.path.exists(seq) for seq in seq_dirs]
        if not all(seq_availability):
            if not seq_availability[-1]:  # If post3 is missing
                print(f"Sequence post3 not available for patient {patient_number}, using post2 for the third channel.")
                seq_dirs[-1] = seq_dirs[-2]  # Use post2 for the third channel if post3 is missing
            else:
                print(f"One or more sequences not available for patient {patient_number}.")
                continue

        bbox = next((item for item in scaled_cube_coordinates if item['patient_number'] == patient_number), None)
        if bbox is None:
            continue

        slice_data = next((item for item in patient_slice_counts if item['patient_number'] == patient_number), None)
        if slice_data is None:
            continue

        total_slices, trimmed_slice_count = slice_data['slice_count'], slice_data['trimmed_slice_count']
        start_trim, end_trim = (total_slices - trimmed_slice_count) // 2, total_slices - (total_slices - trimmed_slice_count) // 2

        rgb_volume = []
        for i in range(start_trim + 1, end_trim + 1):
            try:
                # Load pre sequence image
                pre_image_path = os.path.join(seq_dirs[0], f"{i}.jpg")
                pre_image = cv.imread(pre_image_path, cv.IMREAD_GRAYSCALE)
                if pre_image is None:
                    raise Exception(f"Error loading pre sequence image: {pre_image_path}")

                channels = []
                for seq_dir in seq_dirs[1:]:  # Skip pre sequence for subtraction operation
                    post_image_path = os.path.join(seq_dir, f"{i}.jpg")
                    post_image = cv.imread(post_image_path, cv.IMREAD_GRAYSCALE)
                    if post_image is None:
                        raise Exception(f"Error loading post sequence image: {post_image_path}")

                    # Subtract pre from post sequence
                    diff_image = cv.subtract(post_image, pre_image)

                    # Apply rotation if needed
                    if patient_number in patients_to_rotate:
                        diff_image = cv.rotate(diff_image, cv.ROTATE_180)

                    channels.append(diff_image)

                # Ensure all three channels are present for RGB image creation
                if len(channels) == 3:
                    rgb_image = np.stack(channels, axis=-1)
                    rgb_volume.append(rgb_image)

            except Exception as e:
                print(e)
                break  # Skip this slice if any error occurs

        if len(rgb_volume) > 0:
            all_patient_results.append({
                'patient_number': patient_number,
                'rgb_volume': rgb_volume,
                'cube_coordinates': bbox['cube_coordinates']
            })

    return all_patient_results

def apply_mask_to_slices(results, masks):
    """
    Apply masks to RGB volumes of medical imaging slices for each patient in the results list to segment specific regions, such as breasts.

    This function iterates over a list of result dictionaries, each containing patient data and associated RGB volume of images.
    For each patient, it finds a corresponding mask based on the patient number and applies this mask across all images in the RGB volume.
    It handles mismatches in the expected shape of the masks and RGB volumes and logs errors when masks are missing or shape mismatches occur.

    Parameters:
    - results (list of dict): Each dictionary in the list contains patient-specific data, including the patient's number and 
    the RGB volume of images.
    - masks (list of dict): Each dictionary should have a 'patient_number' and a 'mask' key, where 'mask' is an array corresponding
    to the RGB volume shape.

    Returns:
    - list of dict: The input 'results' list with the RGB volumes modified by applying the respective masks.

    Each step of the masking process is accompanied by a print statement to indicate progress or issues such as missing masks
    or shape mismatches. This function also uses tqdm to display a progress bar with descriptions for better monitoring of the process.
    """
    for result in tqdm(results,desc="Applying masks to all slices for each patient to segment the breasts."):
        patient_number = result['patient_number']
        rgb_volume = result['rgb_volume']

        # Find the matching mask for the current patient
        mask_info = next((item for item in masks if item['patient_number'] == patient_number), None)
        if mask_info is None:
            print(f"Mask data not found for patient {patient_number}.")
            continue

        mask = mask_info['mask']

        # Check if the mask has the expected shape
        if mask.shape != rgb_volume[0].shape:
            print(f"Shape mismatch for patient {patient_number}. Expected {rgb_volume[0].shape}, got {mask.shape}")
            continue

        # Apply the mask to each image in the RGB volume
        for i in range(len(rgb_volume)):
            for c in range(3):  # Loop over each color channel
                rgb_volume[i][:, :, c] *= mask[:, :, c]

        # Update the result with the masked RGB volume
        result['rgb_volume'] = rgb_volume

    return results

def resize_volume_full(volume, new_size=(100, 256, 256, 3)):
    """
    Resize the full RGB volume to a new size using bilinear interpolation for all dimensions.
    
    Parameters:
    - volume: 4D numpy array of shape (depth, height, width, channels).
    - new_size: Tuple (new_depth, new_height, new_width, channels), the target size.
    
    Returns:
    - Resized volume as a numpy array.
    """
    # Calculate the scaling factors for each dimension
    scale_factors = [new / old for new, old in zip(new_size, volume.shape)]
    
    # Resizing the volume using zoom with bilinear interpolation (order=1).
    resized_volume = zoom(volume, scale_factors, order=1)
    
    return resized_volume

def resize_rgb_images(results, new_size=(100, 256, 256, 3)):
    """
    Resizes the RGB volumes for each patient to a new specified size and adjusts the associated cube coordinates accordingly.

    Parameters:
    - results : list of dictionaries
        A list where each dictionary contains patient data, including the original RGB volume and its associated cube coordinates.
    - new_size : tuple, optional
        The target size for the RGB volumes as a tuple (depth, height, width, channels). The default is (100, 256, 256, 3).

    Returns:
    - resized_results : list of dictionaries
        A list of dictionaries where each dictionary holds the patient number, the resized RGB volume, and the newly scaled cube coordinates.

    Each RGB volume is converted to a numpy array if not already an array, resized to the new dimensions, and its associated
    cube coordinates are scaled based on the ratio of new dimensions to original dimensions. The function ensures that z-coordinates
    are adjusted within the bounds of the new depth size.
    """
    resized_results = []

    for patient_data in tqdm(results,desc="Resizing RGB volumes along the three axes: x, y, and z."):
        patient_number = patient_data['patient_number']
        # Ensure rgb_volume is an np.array, especially if initially it's a list of images.
        rgb_volume = np.array(patient_data['rgb_volume'])

        # Resize the RGB volume
        resized_volume = resize_volume_full(rgb_volume, new_size)

        # Calculate new cube coordinates after resizing
        cube_coordinates = patient_data['cube_coordinates']
        scale_factors = [new / old for new, old in zip(new_size, rgb_volume.shape)]

        scaled_bbox = [
            round(cube_coordinates[0] * scale_factors[2]),  # x_min scaled
            round(cube_coordinates[1] * scale_factors[1]),  # y_min scaled
            round(cube_coordinates[2] * scale_factors[2]),  # x_max scaled
            round(cube_coordinates[3] * scale_factors[1]),  # y_max scaled
            max(0, round(cube_coordinates[4] * scale_factors[0])),  # z_min scaled, ensure it's >= 0
            min(new_size[0] - 1, round(cube_coordinates[5] * scale_factors[0]))  # z_max scaled, ensure within new depth
        ]

        resized_results.append({
            'patient_number': patient_number,
            'rgb_volume': resized_volume,
            'scaled_cube_coordinates': scaled_bbox
        })

    return resized_results

def plot_resized_patient_volumes(resized_results, patient_to_plot, num_columns=5):
    """
    Plots the RGB volume of images for a specified patient, highlighting a region of interest with a rectangle based on cube coordinates.

    Parameters:
    - resized_results (list of dicts): Contains patient data including 'patient_number', 'rgb_volume', and 'scaled_cube_coordinates'.
    - patient_to_plot (int): The patient number whose images are to be displayed.
    - num_columns (int): Number of columns in the subplot grid (default is 5).

    This function filters the results for the specified patient, arranges the images in a grid, and highlights specific regions in the images. 
    If no data is found for the patient, it returns early with a message.
    """
    # Filter for the specified patient
    patient_data = next((item for item in resized_results if item['patient_number'] == patient_to_plot), None)

    if not patient_data:
        print(f"No data found for patient {patient_to_plot}.")
        return

    rgb_volume = patient_data['rgb_volume']
    cube_coords = patient_data['scaled_cube_coordinates']

    print(f"Cube coordinates for patient {patient_to_plot}: {cube_coords}")

    num_images = len(rgb_volume)
    num_rows = (num_images + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 3, num_rows * 3))
    axes = axes.flatten()

    for i, img in enumerate(rgb_volume):
        ax = axes[i]
        ax.imshow(img)

        # Title with slice number
        ax.set_title(f"Slice {i + 1}")

        # Draw rectangle if within z_min and z_max
        if cube_coords[4]-1 <= i < cube_coords[5]-1:
            x_min, y_min, x_max, y_max = cube_coords[:4]
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        ax.axis('on')

    # Hide any unused axes if the number of images is not a perfect fit for the grid
    for j in range(i + 1, num_columns * num_rows):
        axes[j].axis('on')

    plt.tight_layout()
    plt.show()

def split_patient_data(volumes, test_percentage=0.2, start_point=0):
    """
    Splits the data into train and test sets in a patient-wise manner, starting the selection
    of test patients from a specified start point.

    Parameters:
    - resized_volumes: A list of dictionaries, where each dictionary contains data for one patient.
    - test_percentage: The percentage of patients to include in the test set.
    - start_point: The starting index for selecting test patients systematically.

    Returns:
    - train_set: A list of dictionaries for the training set.
    - test_set: A list of dictionaries for the test set.
    """
    # Calculate the number of patients to include in the test set
    total_patients = len(volumes)
    num_test_patients = int(total_patients * test_percentage)

    # Sort results by patient number to ensure systematic selection
    sorted_results = sorted(volumes, key=lambda x: x['patient_number'])

    # Determine the step for systematic selection based on the desired test set size
    step = total_patients // num_test_patients

    # Adjust start_point if it's outside the range [0, step)
    start_point = start_point % step

    # Select test patients starting from start_point, stepping through the patients
    test_indices = list(range(start_point, total_patients, step))

    test_set = [sorted_results[i] for i in test_indices]
    train_set = [sorted_results[i] for i in range(total_patients) if i not in test_indices]

    return train_set, test_set

def save_images_and_labels(data, base_dir):
    """
    Saves images and corresponding labels in YOLO format. Positive slices (with tumors)
    have their labels saved with bounding box coordinates. Negative slices (without tumors)
    have an empty label file generated.

    Parameters:
    - data (list of dicts): Dataset containing image data, bounding boxes, and patient IDs.
      Each dictionary must include keys for 'rgb_image', 'cube_coordinates', and 'patient_number',
      indicating the image data, bounding box coordinates, and patient ID, respectively.
    - base_dir (str): Base directory path for saving the dataset. This directory will be
      cleared and recreated with 'images' and 'labels' subdirectories on each call.
    """

    if 'train' in base_dir:
        dataset_type = 'train'
    elif 'test' in base_dir:
        dataset_type = 'test'
    else:
        raise ValueError("Base directory should include either 'train' or 'test'.")

    images_dir = os.path.join(base_dir, dataset_type, 'images')
    labels_dir = os.path.join(base_dir, dataset_type, 'labels')

    # Clear previous data specific to dataset type ('train' or 'test')
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for item in data:
        images = item['rgb_volume']
        bbox = item['scaled_cube_coordinates']  # Format: [x_min, y_min, x_max, y_max, z_min, z_max]
        patient_id = item['patient_number']

        z_min, z_max = bbox[4], bbox[5]

        for idx, image in enumerate(images):
            slice_number = idx + 1  # Slice numbering starts from 1
            # Ensure z_min is inclusive and z_max is exclusive
            class_label = 1 if z_min <= slice_number < z_max else 0  # Use 0 to denote negative slices

            file_name = f"{str(patient_id).zfill(3)}_{str(slice_number).zfill(3)}"
            image_path = os.path.join(images_dir, f'{file_name}.jpg')
            label_path = os.path.join(labels_dir, f'{file_name}.txt')

            # Save the RGB image
            cv.imwrite(image_path, cv.cvtColor(image, cv.COLOR_RGB2BGR))  # Ensure correct color conversion
            # Handle label file creation based on slice positivity
            if class_label == 1:
                x_center = ((bbox[0] + bbox[2]) / 2) / image.shape[0]
                y_center = ((bbox[1] + bbox[3]) / 2) / image.shape[1]
                width = (bbox[2] - bbox[0]) / image.shape[0]
                height = (bbox[3] - bbox[1]) / image.shape[1]
                with open(label_path, 'w') as f:
                    f.write(f'{class_label-1} {x_center} {y_center} {width} {height}\n')  # class_label-1 for zero-indexing
            else:
                # Create an empty label file for negative slices
                open(label_path, 'a').close()

def plot_loss(results_path):
    """
    Plot and save the training and validation loss from a YOLOv5 training session.

    Parameters:
    - results_path: String, path to the results.csv file containing training and validation loss data.
    """
    # Load the data
    df = pd.read_csv(results_path)

    # Clean up column names
    df.columns = df.columns.str.strip()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
    plt.plot(df['epoch'], df['val/box_loss'], label='Validation Box Loss')
    plt.plot(df['epoch'], df['train/obj_loss'], label='Train Object Loss')
    plt.plot(df['epoch'], df['val/obj_loss'], label='Validation Object Loss')
    plt.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', linestyle='dotted')
    plt.plot(df['epoch'], df['val/cls_loss'], label='Validation Class Loss', linestyle='dotted')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)

    # Save the plot in the same directory as the CSV
    # Create a unique filename with a timestamp
    filename = f"loss_plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    save_path = '/'.join(results_path.split('/')[:-1]) + '/' + filename
    plt.savefig(save_path)
    plt.close()  # Close the figure after saving to free up memory

    return save_path  # Optional, return the path where the file was saved

def process_detections(label_dir, output_dir):
    """
    Processes detection labels by selecting the bounding box with the highest confidence
    when multiple detections are present, and saves the refined labels to a new directory.

    This function iterates through all label files in a specified directory, each containing
    potential multiple detected bounding boxes for an image. For each label file, if multiple
    detections are found, only the detection with the highest confidence score is kept. Label
    files with no detections are skipped. The selected or unchanged detection for each image
    is then saved to a new label file in an output directory, ensuring that each image is
    represented by at most one bounding box with the highest confidence.

    Parameters:
    - label_dir (str): The directory containing the original label files with detections.
    - output_dir (str): The directory where the processed label files will be saved.

    Each label file should be in the format:
    <class_id> <x_center> <y_center> <width> <height> <confidence>
    where <x_center>, <y_center>, <width>, and <height> are relative to the image size.
    """
    for label_file in os.listdir(label_dir):
        filepath = os.path.join(label_dir, label_file)
        with open(filepath, 'r') as file:
            lines = file.readlines()
            # Select the detection with the highest confidence, if multiple
            if len(lines) > 1:
                highest_confidence = 0
                best_line = None
                for line in lines:
                    parts = line.split()
                    confidence = float(parts[5])
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_line = line
                lines = [best_line]
            # If no detections, you might choose to handle this differently
            # For now, we'll skip images with no detections
            elif len(lines) == 0:
                continue  # Or handle no-detection scenario

        # Save the processed label
        output_path = os.path.join(output_dir, label_file)
        with open(output_path, 'w') as file:
            file.writelines(lines)

def get_patient_and_slice_number(filename):
    """
    Extracts and returns the patient number and slice number from a filename.

    Parameters:
    - filename (str): The filename formatted as 'patientNumber_sliceNumber.txt'.

    Returns:
    - tuple (int, int): A tuple containing the patient number and slice number.
    """
    parts = filename.split('_')
    patient_number = int(parts[0])
    slice_number = int(parts[1].split('.')[0])  # Removing '.txt' and converting to int
    return patient_number, slice_number

def convert_to_corners(boxes):
    """Convert [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max]."""
    new_boxes = []
    for box in boxes:
        x_center, y_center, width, height = box
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        x_max = x_center + (width / 2)
        y_max = y_center + (height / 2)
        new_boxes.append([x_min, y_min, x_max, y_max])
    return new_boxes

def iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two 2D bounding boxes.

    Parameters:
    - boxA (tuple of floats): Coordinates of the first bounding box (x_min, y_min, x_max, y_max).
    - boxB (tuple of floats): Coordinates of the second bounding box in the same format as boxA.

    Returns:
    - float: The IoU ratio, ranging from 0 (no overlap) to 1 (perfect overlap). This is calculated as the area of the intersection
    divided by the area of the union of the two boxes.

    Method:
    1. Determine the (x, y) coordinates of the intersection rectangle.
    2. Compute the area of the intersection rectangle.
    3. Calculate the area of each bounding box.
    4. Calculate the union of the areas of boxA and boxB, subtracting the intersection area to avoid double counting.
    5. Compute the IoU as the ratio of the intersection area to the union area, ensuring division by float to prevent integer division issues.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def visualize_patient_data(data, patient_number, pred_label_dir, num_columns=5, use_processed_predictions=False, show=False):
    """
    Visualizes and returns the patient data including actual and predicted bounding boxes for slices.

    Parameters:
    - data (list): A list of dictionaries containing patient data.
    - patient_number (int or str): The patient number to visualize.
    - pred_label_dir (str): Directory path where predicted labels are stored.
    - num_columns (int): Number of columns in the visualization grid (default is 5).
    - use_processed_predictions (bool): If True, use processed predictions from a modified directory path.
    - show (bool): If True, displays the visualizations.

    Returns:
    - list: A list of results with slice details and bounding box coordinates.
    - str: Error message if no data is found.

    The function displays RGB images and their respective bounding boxes, processing each slice individually.
    """
    patient_number = int(patient_number)
    patient_data = next((item for item in data if item['patient_number'] == patient_number), None)
    if patient_data is None:
        print(f"No data found for patient number {patient_number}.")
        return [], "No data found."

    cube_coordinates = patient_data['scaled_cube_coordinates']
    print(f"Cube coordinates for patient {patient_number}: {cube_coordinates}")

    rgb_images = patient_data['rgb_volume']
    num_images = len(rgb_images)
    num_rows = (num_images + num_columns - 1) // num_columns

    results = []

    if show:
        plt.figure(figsize=(num_columns * 5, num_rows * 5))

    for i, image in enumerate(rgb_images, start=1):
        if show:
            ax = plt.subplot(num_rows, num_columns, i)
            if image.ndim == 4 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            ax.imshow(image)

        filename = f"{str(patient_number).zfill(3)}_{str(i).zfill(3)}.jpg"
        pred_label_path = os.path.join(pred_label_dir, filename.replace('.jpg', '.txt'))
        if use_processed_predictions:
            pred_label_path = pred_label_path.replace('/labels', '/processed_labels')

        predicted_bbox = None  # Default value if no predicted bounding box is found

        # Draw actual bounding box for images between z_min and z_max, include z_min and z_max in the actual_bbox
        actual_bbox = None
        if cube_coordinates[4] <= i <= cube_coordinates[5]:
            x_min, y_min, x_max, y_max = cube_coordinates[:4]
            width = x_max - x_min
            height = y_max - y_min
            actual_bbox = [x_min, y_min, x_max, y_max, cube_coordinates[4], cube_coordinates[5]]
            if show:
                ax.add_patch(Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='g', facecolor='none', label='Actual'))

        # Draw predicted bounding box if exists
        if os.path.exists(pred_label_path):
            with open(pred_label_path, 'r') as file:
                for line in file:
                    class_id, x_center, y_center, width, height, conf = map(float, line.split())
                    x1 = (x_center - width / 2) * image.shape[1]
                    y1 = (y_center - height / 2) * image.shape[0]
                    box_width = width * image.shape[1]
                    box_height = height * image.shape[0]
                    predicted_bbox = [x1, y1, x1 + box_width, y1 + box_height]
                    if show:
                        ax.add_patch(Rectangle((x1, y1), box_width, box_height, linewidth=2, edgecolor='r', facecolor='none', label='Predicted'))

        if show:
            plt.axis('on')
            plt.title(f'Slice: {i}')

        # Collect results
        results.append({
            'patient_number': patient_number,
            'slice_number': i,
            'actual_cube_coordinates': cube_coordinates,  # Now includes z_min and z_max
            'predicted_bbox': predicted_bbox
        })

    if show:
        plt.tight_layout()
        plt.show()

    return results

def do_overlap(box1, box2):
    """
    Check if two bounding boxes overlap.
    Each box is specified by a list of four coordinates [x1, y1, x2, y2].
    """
    # Unpack the coordinates
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Check if there's no overlap
    if (x1_box1 > x2_box2 or x2_box1 < x1_box2 or
        y1_box1 > y2_box2 or y2_box1 < y1_box2):
        return False

    return True

def calculate_overlap_stats_with_true_false(patient_results):
    """
    Calculate the overlap statistics for bounding boxes per patient, distinguishing between
    true overlaps (within z_min and z_max) and false overlaps (outside that range), as well
    as counting slices with no predicted bounding box.

    Parameters:
    - patient_results: A list of dictionaries containing patient number, slice number,
      actual cube coordinates, and predicted bounding boxes.

    Returns:
    A dictionary with patient numbers as keys, each associated with a dictionary containing
    counts of true overlaps, false overlaps, non-overlaps, and no predictions.
    """
    overlap_stats = {}

    for result in patient_results:
        patient_number = result['patient_number']
        actual_cube = result['actual_cube_coordinates'][:4]  # Ignore z_min and z_max for 2D overlap check
        z_min, z_max = result['actual_cube_coordinates'][4], result['actual_cube_coordinates'][5]
        predicted_bbox = result.get('predicted_bbox')
        slice_number = result['slice_number']

        # Initialize patient stats if not already present
        if patient_number not in overlap_stats:
            overlap_stats[patient_number] = {
                'true_overlaps': 0,
                'false_overlaps': 0,
                'non_overlaps': 0,
                'no_prediction': 0
            }

        # Check and count the overlaps, non-overlaps, and no predictions
        if predicted_bbox:
            if do_overlap(actual_cube, predicted_bbox):
                # Check if it's a true overlap based on the slice number
                if z_min <= slice_number < z_max:
                    overlap_stats[patient_number]['true_overlaps'] += 1
                else:
                    overlap_stats[patient_number]['false_overlaps'] += 1
            else:
                overlap_stats[patient_number]['non_overlaps'] += 1
        else:
            # Increment no prediction count
            overlap_stats[patient_number]['no_prediction'] += 1

    return overlap_stats

def find_similar_bboxes_center_distance(patient_results, confidence_results, max_distance=10, min_group_size=3, comparison_range=5):
    """
    Analyzes bounding boxes predicted by YOLOv5 to identify and group similar detections based on the Euclidean distance
    between their center points. This grouping helps in aggregating detections that might belong to the same object across
    multiple slices or frames within a defined proximity.

    Parameters:
    - patient_results (list of dicts): A list containing dictionaries where each dict represents detection results for a patient,
    including patient number, slice number, and predicted bounding box.
    - confidence_results (list of dicts): A list of dictionaries containing the confidence scores for each detection,
    including patient number, slice number, and confidence score.
    - max_distance (int, optional): The maximum allowed distance between centers of two bounding boxes to consider them as similar. Defaults to 10.
    - min_group_size (int, optional): The minimum number of bounding boxes required to form a group. Defaults to 3.
    - comparison_range (int, optional): The maximum number of consecutive slices to be considered for grouping bounding boxes. Defaults to 5.

    Returns:
    - list of dicts: Returns a sorted list of dictionaries, each representing a group of similar bounding boxes.
    Each group dictionary contains patient number, average predicted bounding box, start and end slice numbers, average confidence,
    and maximum confidence of the detections within the group.

    Algorithm Steps:
    1. Define helper functions for calculating center points, distances, average bounding boxes, and confidence metrics.
    2. Iterate over unique patients and their respective slices.
    3. For each slice, compare its bounding box with the next slices within the comparison range using the distance function.
    4. Group slices whose bounding boxes are within the maximum distance.
    5. If a group meets the minimum size requirement, calculate its average bounding box and confidence metrics.
    6. Append the group details to the results list if it meets the criteria.
    7. Sort and return the list of similar bounding box groups.
    """
    def center(bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def distance(center1, center2):
        return math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    def average_bbox(bboxes):
        avg = [0, 0, 0, 0]
        for bbox in bboxes:
            for i in range(4):
                avg[i] += bbox[i]
        return [x / len(bboxes) for x in avg]

    def average_confidence(slice_numbers, patient_number):
        confidences = [item['confidence'] for item in confidence_results if item['slice_number'] in slice_numbers and item['patient_number'] == patient_number]
        return sum(confidences) / len(confidences) if confidences else 0

    def max_confidence(slice_numbers, patient_number):
        confidences = [item['confidence'] for item in confidence_results if item['slice_number'] in slice_numbers and item['patient_number'] == patient_number]
        return max(confidences) if confidences else 0

    similar_groups = []

    for patient_number in set(item['patient_number'] for item in patient_results):
        patient_slices = sorted([item for item in patient_results if item['patient_number'] == patient_number and item['predicted_bbox'] is not None], key=lambda x: x['slice_number'])

        i = 0
        while i < len(patient_slices):
            current_slice = patient_slices[i]
            current_group = [current_slice]
            slice_numbers = [current_slice['slice_number']]
            j = i + 1

            while j < len(patient_slices) and j <= i + comparison_range:
                next_slice = patient_slices[j]
                if distance(center(current_slice['predicted_bbox']), center(next_slice['predicted_bbox'])) <= max_distance:
                    current_group.append(next_slice)
                    slice_numbers.append(next_slice['slice_number'])
                    i = j  # Move i to the last slice added to the group to prevent overlap
                j += 1

            if len(current_group) >= min_group_size:
                avg_bbox = average_bbox([slice['predicted_bbox'] for slice in current_group])
                avg_conf = average_confidence(slice_numbers, patient_number)
                max_conf = max_confidence(slice_numbers, patient_number)
                similar_groups.append({
                    'patient_number': patient_number,
                    'average_predicted_bbox': avg_bbox,
                    'start_slice': current_group[0]['slice_number'],
                    'end_slice': current_group[-1]['slice_number'],
                    'avg_conf': avg_conf,
                    'max_conf': max_conf
                })

            i += 1  # Move to the next slice if no group was formed or continue from the end of the group

    similar_groups.sort(key=lambda x: (x['patient_number'], x['start_slice']))
    return similar_groups

def visualize_aggregated_images_with_bboxes(test_data, similar_bboxes):
    """
    Visualizes aggregated RGB images for each patient group with actual and predicted bounding boxes,
    displaying the average and maximum confidence of the predicted boxes as multiline captions under each image.

    Parameters:
    - test_data (list of dicts): A list where each element is a dictionary containing the patient's data, including
      the patient number, RGB volume of slices, and scaled cube coordinates representing the actual bounding box.
    - similar_bboxes (list of dicts): A list where each element is a dictionary containing information about a group
      of similar slices for a patient. This includes the patient number, the start and end slices for the group,
      and the average predicted bounding box.
    """
    grouped_bboxes = {}
    for bbox_info in similar_bboxes:
        patient_number = bbox_info['patient_number']
        if patient_number not in grouped_bboxes:
            grouped_bboxes[patient_number] = []
        grouped_bboxes[patient_number].append(bbox_info)

    for patient_number, bboxes in grouped_bboxes.items():
        patient_data = next((item for item in test_data if item['patient_number'] == patient_number), None)
        if patient_data is None:
            print(f"Patient number {patient_number} not found in test data.")
            continue

        rgb_volume = patient_data['rgb_volume']
        actual_cube_coords = patient_data['scaled_cube_coordinates'][:4]  # x_min, y_min, x_max, y_max
        z_min, z_max = patient_data['scaled_cube_coordinates'][4:]  # z_min, z_max

        plt.figure(figsize=(3 * len(bboxes), 4))  # Adjusted figure height for multiline captions
        for i, bbox_info in enumerate(bboxes):
            start_slice = bbox_info['start_slice']
            end_slice = bbox_info['end_slice']
            avg_pred_bbox = bbox_info['average_predicted_bbox']
            avg_conf = bbox_info['avg_conf']
            max_conf = bbox_info['max_conf']

            aggregated_image = np.mean(rgb_volume[start_slice-1:end_slice], axis=0).astype(np.uint8)

            cv.rectangle(aggregated_image, (int(actual_cube_coords[0]), int(actual_cube_coords[1])),
                         (int(actual_cube_coords[2]), int(actual_cube_coords[3])), (0, 255, 0), 1)
            cv.rectangle(aggregated_image, (int(avg_pred_bbox[0]), int(avg_pred_bbox[1])),
                         (int(avg_pred_bbox[2]), int(avg_pred_bbox[3])), (255, 0, 0), 1)

            ax = plt.subplot(1, len(bboxes), i+1)
            plt.imshow(aggregated_image)
            plt.title(f"Slices {start_slice}-{end_slice}, Z: {z_min}-{z_max}")
            plt.axis('off')  # Turn off axis for a cleaner look
            # Place the text directly using plt.text
            plt.text(0.5, -0.2, f"Avg. Conf.: {avg_conf:.3f}", fontsize=12, ha='center', transform=ax.transAxes)
            plt.text(0.5, -0.3, f"Max Conf.: {max_conf:.3f}", fontsize=12, ha='center', transform=ax.transAxes)

        plt.suptitle(f"Patient {patient_number}")
        plt.tight_layout()
        plt.show()

def calculate_iou_3d(pred_box, actual_box):
    """
    Calculates the Intersection over Union (IoU) for two 3D bounding boxes.

    Parameters:
    - pred_box (tuple of floats): A tuple representing the predicted bounding box in the format (x_min, y_min, x_max, y_max, z_min, z_max),
    where `x_min`, `y_min`, `z_min` are the coordinates of the lower corner, and `x_max`, `y_max`, `z_max` are the coordinates of the upper corner.
    - actual_box (tuple of floats): A tuple representing the actual bounding box in the same format as pred_box.

    Returns:
    - float: The IoU ratio, a value between 0 and 1, where 0 means no overlap and 1 means perfect overlap.
    The IoU is calculated as the volume of the intersection divided by the volume of the union of the two boxes.

    The function calculates overlap in the x, y, and z dimensions, determines the volume of the intersection of the two bounding boxes,
    computes the volumes of each individual box, and calculates the union of the volumes to finally compute the IoU.
    If the union of the boxes is zero, the function returns 0 to avoid division by zero.
    """
    # Unpack the predicted and actual box coordinates
    x1_min, y1_min, x1_max, y1_max, z1_min, z1_max = pred_box
    x2_min, y2_min, x2_max, y2_max, z2_min, z2_max = actual_box

    # Calculate overlap in each dimension
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))

    # Calculate volume of intersection
    intersection = x_overlap * y_overlap * z_overlap

    # Calculate volumes of each box
    volume1 = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
    volume2 = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)

    # Calculate union
    union = volume1 + volume2 - intersection

    # Compute IoU
    iou = intersection / union if union != 0 else 0
    return iou
    
def calculate_absolute_errors(z1, z2):
    """
    Calculates the absolute differences between corresponding elements of two tuples.

    Parameters:
    - z1 (tuple): A tuple containing two numeric values (z1_min, z1_max).
    - z2 (tuple): Another tuple containing two numeric values, corresponding to z1.

    Returns:
    - tuple: A tuple of absolute differences between the elements of z1 and z2.
    """
    # Calculate absolute errors
    return abs(z1[0] - z2[0]), abs(z1[1] - z2[1])  # z1_min, z1_max

def visualize_aggregated_images_with_highest_avg_conf(test_data, similar_bboxes, confidence_threshold=0):
    """
    Visualizes aggregated images for each patient from the test data, selecting the bounding box with the highest average confidence.
    Additionally, it checks if the selected bounding box's average confidence exceeds a specified threshold and discards those
    that do not meet this criterion.

    Parameters:
        test_data (list of dicts): A list containing patient data. Each entry should have patient-specific metrics and image data.
        similar_bboxes (list of dicts): A list of bounding boxes with confidence scores across different patients.
        confidence_threshold (float, optional): The minimum average confidence required to consider a bounding box valid. Defaults to 0 (no limitation).

    Returns:
        None: This function directly prints the results and visualizations without returning any value.

    Discarded patients (due to not meeting the confidence threshold) and remaining patients are tracked and reported at the end.
    Metrics for each patient, such as IoU (Intersection over Union) for volume and area, as well as absolute errors for the Z dimensions,
    are calculated and printed.
    Each patient's aggregated RGB volume is visualized with actual and predicted bounding boxes.
    """

    volume_ious = []
    area_ious = []
    z_min_errors = []
    z_max_errors = []
    discarded_patients = 0
    remaining_patients = 0

    for patient_data in test_data:
        patient_number = patient_data['patient_number']
        bboxes = [bbox for bbox in similar_bboxes if bbox['patient_number'] == patient_number]

        if not bboxes:
            continue

        # Select the bbox with the highest avg_conf
        selected_bbox = max(bboxes, key=lambda x: x['avg_conf'])
        avg_conf = selected_bbox['avg_conf']

        # Check if the selected bbox's avg_conf meets the threshold
        if avg_conf < confidence_threshold:
            discarded_patients += 1
            continue  # Skip this patient if avg_conf is below threshold

        remaining_patients += 1  # Count this patient as remaining

        # Proceed with remaining processing...
        start_slice = selected_bbox['start_slice']
        end_slice = selected_bbox['end_slice']
        avg_pred_bbox = selected_bbox['average_predicted_bbox']

        # Unpack the actual bounding box and calculate the IoU
        actual_cube_coords = patient_data['scaled_cube_coordinates']
        predicted_cube_coords = avg_pred_bbox + [start_slice, end_slice]
        iou_volume = calculate_iou_3d(predicted_cube_coords, actual_cube_coords)
        iou_area = iou(avg_pred_bbox, actual_cube_coords[:4])
        z_min_error, z_max_error = calculate_absolute_errors(actual_cube_coords[4:], [start_slice, end_slice])

        volume_ious.append(iou_volume)
        area_ious.append(iou_area)
        z_min_errors.append(z_min_error)
        z_max_errors.append(z_max_error)

        rgb_volume = patient_data['rgb_volume']
        aggregated_image = np.mean(rgb_volume[start_slice-1:end_slice], axis=0).astype(np.uint8)

        cv.rectangle(aggregated_image, (int(actual_cube_coords[0]), int(actual_cube_coords[1])),
                     (int(actual_cube_coords[2]), int(actual_cube_coords[3])), (0, 255, 0), 2)
        cv.rectangle(aggregated_image, (int(avg_pred_bbox[0]), int(avg_pred_bbox[1])),
                     (int(avg_pred_bbox[2]), int(avg_pred_bbox[3])), (255, 0, 0), 2)

        print(f"Patient {patient_number} - IoU Volume: {iou_volume:.3f}, IoU Area: {iou_area:.3f}, "
              f"Z-min Error: {z_min_error}, Z-max Error: {z_max_error}")

        plt.figure(figsize=(5, 4))
        plt.imshow(aggregated_image)
        plt.title(f'Patient {patient_number}, IoU Volume: {iou_volume:.3f}')
        plt.xlabel(f'Slices: {start_slice}-{end_slice}, Avg. Conf.: {avg_conf:.3f}')
        plt.axis('on')
        plt.show()

    print(f"Average IoU Volume: {np.mean(volume_ious):.3f}")
    print(f"Average IoU Area: {np.mean(area_ious):.3f}")
    print(f"Average Z-min Error: {np.mean(z_min_errors):.3f}")
    print(f"Average Z-max Error: {np.mean(z_max_errors):.3f}")
    print(f"{discarded_patients} patients were discarded due to the threshold of {confidence_threshold} not being met.")
    print(f"{remaining_patients} patients were considered after applying the threshold.")
