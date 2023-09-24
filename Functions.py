# ##########################################################################################
# #####################################   IMPORTS    #######################################
# ########################
# # ##################################################################
import os
import re
import math
import time
import glob
import random
import sklearn
import pyfeats
import pydicom
import patoolib
import operator
import mahotas
import cv2 as cv
import collections
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import scipy.io as sio
#import tensorflow as tf
from scipy.stats import t
from random import choice
from statistics import mode
from pyunpack import Archive
import matplotlib.pyplot as plt
#from keras.models import Sequential
from platform import python_version
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from skimage.io import imsave, imread
from sklearn.impute import KNNImputer
#from keras.callbacks import EarlyStopping
from IPython.display import Image, display
from sklearn import datasets, metrics, svm
from collections import Counter, defaultdict
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
#from tensorflow.keras.utils import to_categorical
#from keras.layers import Dense, Activation, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    f1_score, make_scorer, confusion_matrix, accuracy_score, classification_report,
    precision_score, recall_score, average_precision_score
)
from sklearn.model_selection import (
    GridSearchCV, validation_curve, train_test_split, KFold, cross_val_score,
    StratifiedKFold
)
from pyfeats import (
    fos, glcm_features, glds_features, ngtdm_features, sfm_features, lte_measures, fdta, glrlm_features,
    fps, shape_parameters, glszm_features, hos_features, lbp_features, grayscale_morphology_features,
    multilevel_binary_morphology_features, histogram, multiregion_histogram, amfm_features,
    dwt_features, gt_features, zernikes_moments, hu_moments, hog_features
)

##########################################################################################
################################   DIRECTORY HANDLER    ##################################
##########################################################################################
# current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.getcwd()
print("Current path is: ", current_path)
bc_mri_path = current_path + r'\BC_MRI'
dataset_path = bc_mri_path + r'\dataset'
xlsx_csv_files_path = bc_mri_path + r'\xlsx_csv_files'
samples_path = dataset_path + r'\Duke-Breast-Cancer-MRI'
types = ['pre', 'post_1', 'post_2', 'post_3']
clinical_file_path = xlsx_csv_files_path + r'\Clinical_and_Other_Features.csv'
mapping_path = xlsx_csv_files_path + r'\Breast-Cancer-MRI-filepath_filename-mapping.csv'
boxes_path = xlsx_csv_files_path + r'\Annotation_Boxes.csv'
radiomics_clinical_path=bc_mri_path+r'\extracted_features\radiomics_clinical_features_data.csv'
features_by_saha=xlsx_csv_files_path + r'\Imaging_Features.csv'
def random_sample_for_each_cancer_type(path, N0=2, N1=2, N2=2, N3=2,
                                      exclude=[103,164,253,258,282,700,728,801,893],
                                      random_seed =42, show_patients=False):
    """
    Generate random samples for each cancer type using the labels obtained from clinical features.

    Args:
        path (str): The path to the clinical features file.
        N0 (int): The maximum number of samples to select for cancer type 0. Default is 50.
        N1 (int): The maximum number of samples to select for cancer type 1. Default is 50.
        N2 (int): The maximum number of samples to select for cancer type 2. Default is 50.
        N3 (int): The maximum number of samples to select for cancer type 3. Default is 50.
        exclude (list): List of patients to exclude initially. By default, it includes a list
        of 9 patients who do not have the pre-contrast 3 sequence. Additional patients with
        poor image quality can be included in this list for exclusion.
        random_seed (int or None): Seed value for controlling the randomness of the function. Default is 42.
        show_patients (bool): Whether to print the list of patients for each cancer type. Default is False.

    Returns:
        tuple: Four lists containing the random samples for each cancer type.

    """
    clinical_features = pd.read_csv(path)
    label = clinical_features.iloc[2:, 26]
    label = label.astype(int)
    label = label.reset_index(drop=True)
    list0 = []
    list1 = []
    list2 = []
    list3 = []

    for i in range(0, 4):
        patients = (np.where(label == i)[0]) + 1

        if show_patients:
            print(f"\n\nThere are {len(patients)} patients diagnosed with cancer type {i}.")
            print(f"The following patients have been identified for cancer type {i}:\n{patients}")

        if i == 0:
            N = min(N0, (label == 0).sum())
        elif i == 1:
            N = min(N1, (label == 1).sum())
        elif i == 2:
            N = min(N2, (label == 2).sum())
        else:
            N = min(N3, (label == 3).sum())

        mask_ = np.isin(patients, exclude)
        filtered_patients = patients[~mask_]
        np.random.seed(random_seed)
        random_patients = np.random.choice(filtered_patients, size=N, replace=False)
        ls_ = (random_patients.tolist())
        ls_ = sorted(ls_)

        if i == 0:
            list0.extend(ls_)
        elif i == 1:
            list1.extend(ls_)
        elif i == 2:
            list2.extend(ls_)
        else:
            list3.extend(ls_)

    return list0, list1, list2, list3

def filter_mapping_df(mapping_path, list_,seq_type):
    """
    Filter the mapping DataFrame based on the provided list of patients.

    Args:
        mapping_path (str): Path to the CSV file containing the mapping data.
        list_ (list): List of patient numbers to filter.
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'.

    Returns:
        pandas.DataFrame, one filtered DataFrames based on sequence type ('pre', 'post_1', 'post_2', 'post_3').
    """
    mapping_df = pd.read_csv(mapping_path)
    mapping_df_seq = pd.DataFrame()

    filtered_df = mapping_df[mapping_df['original_path_and_filename'].str.contains(seq_type)]
    crossref_pattern = '|'.join(["DICOM_Images/Breast_MRI_{:03d}".format(s) for s in list_])
    filtered_df = filtered_df[filtered_df['original_path_and_filename'].str.contains(crossref_pattern)]
    mapping_df_seq=filtered_df
    return mapping_df_seq

def save_dcm_slice(dcm_fname, label, vol_idx, seq_type):
    """
    Save a DICOM slice as a PNG file, separating positive and negative slices into separate folders.

    Args:
        dcm_fname (str): Filepath of the DICOM file.
        label (int): Target label. 1 for positive, 0 for negative.
        vol_idx (int): Volume index.
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'.

    Returns:
        None
    """
    # Create a path to save the slice .png file, based on the original DICOM filename, target label, and sequence type
    png_path = dcm_fname.split('/')[-1].replace('.dcm', '-{}-{}.png'.format(seq_type, vol_idx))
    label_dir = 'pos' if label == 1 else 'neg'
    target_png_dir = dataset_path+'/target_{}'.format(seq_type)
    png_path = os.path.join(target_png_dir, label_dir, png_path)

    if not os.path.exists(os.path.join(target_png_dir, label_dir)):
        os.makedirs(os.path.join(target_png_dir, label_dir))
    if not os.path.exists(png_path):
        # Load DICOM file with pydicom library
        try:
            dcm = pydicom.dcmread(dcm_fname)
        except FileNotFoundError:
            # Fix possible errors in filename from list
            dcm_fname_split = dcm_fname.split('/')
            dcm_fname_end = dcm_fname_split[-1]
            assert dcm_fname_end.split('-')[1][0] == '0'

            dcm_fname_end_split = dcm_fname_end.split('-')
            dcm_fname_end = '-'.join([dcm_fname_end_split[0], dcm_fname_end_split[1][1:]])

            dcm_fname_split[-1] = dcm_fname_end
            dcm_fname = '/'.join(dcm_fname_split)
            dcm = pydicom.dcmread(dcm_fname)

        # Convert DICOM into a numerical numpy array of pixel intensity values
        img = dcm.pixel_array

        # Convert uint16 datatype to float, scaled properly for uint8
        img = img.astype(np.float32) * 255. / img.max()
        # Convert from float -> uint8
        img = img.astype(np.uint8)
        # Invert image if necessary, according to DICOM metadata
        img_type = dcm.PhotometricInterpretation
        if img_type == "MONOCHROME1":
            img = np.invert(img)

        # Save the final .png
        imsave(png_path, img)

def process_mapping_df(data_path, mapping_path,boxes_path, mapping_df, seq_type,N_class=12000):
    """
    Process the mapping dataframe and save DICOM slices as PNG files based on the specified criteria.

    Args:
        data_path (str): Path to the DICOM data.
        mapping_path (str): Path to the CSV file containing the mapping data.
        boxes_path (str): Path to the CSV file containing tumor bounding box information.
        mapping_df: (pandas.DataFrame) filtered DataFrames based on sequence type.
        N_class (int): Number of examples for each class.
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'.

    Returns:
        None
    """
    boxes_df = pd.read_csv(boxes_path)
    # Counts of examples extracted from each class
    ct_negative = 0
    ct_positive = 0

    # Initialize iteration index of each patient volume
    vol_idx = -1
    for row_idx, row in tqdm(mapping_df.iterrows(), total=N_class * 2):
        # Indices start at 1 here
        new_vol_idx = int((row['original_path_and_filename'].split('/')[1]).split('_')[-1])
        slice_idx = int(((row['original_path_and_filename'].split('/')[-1]).split('_')[-1]).replace('.dcm', ''))

        # New volume: get tumor bounding box
        if new_vol_idx != vol_idx:
            box_row = boxes_df.iloc[[new_vol_idx - 1]]
            start_slice = int(box_row['Start Slice'])
            end_slice = int(box_row['End Slice'])
            assert end_slice >= start_slice
        vol_idx = new_vol_idx

        # Get DICOM filename
        dcm_fname = str(row['classic_path'])
        dcm_fname = os.path.join(data_path, dcm_fname)

        # Determine slice label:
        # (1) If within 3D box, save as positive
        if slice_idx >= start_slice and slice_idx < end_slice:
            if ct_positive >= N_class:
                continue
            save_dcm_slice(dcm_fname, 1, vol_idx, seq_type)
            ct_positive += 1

        # (2) If outside 3D box by >5 slices, save as negative
        elif (slice_idx + 5) <= start_slice or (slice_idx - 5) > end_slice:
            if ct_negative >= N_class:

                continue
            save_dcm_slice(dcm_fname, 0, vol_idx, seq_type)
            ct_negative += 1

def image_filenames_plot_one_at_random(seq_type='pre', label='pos', show=False):
    """
    Plot a random PNG image based on the specified label and sequence type.

    Args:
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        label (str, 'pos' or 'neg'): Label for the image directory. Defaults to 'pos'.
        show (bool): Whether to display the image. Defaults to False.

    Returns:
        list: List of image filenames in the specified directory.
    """
    target_png_dir = dataset_path+'/target_{}'.format(seq_type)
    image_dir = os.path.join(target_png_dir, label)
    image_filenames = os.listdir(image_dir)
    sample_image_path = os.path.join(image_dir, choice(image_filenames))
    if show:
        display(Image(filename=sample_image_path))
        print(sample_image_path)
    return image_filenames

def patients_number(seq_type='pre', label='pos'):
    """
    Find the patient number associated with each slice.

    Args:
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        label (str, 'pos' or 'neg'): Label for the image directory. Defaults to 'pos'.

    Returns:
        list: containing the patient numbers for each slice.
    """
    image_filenames = image_filenames_plot_one_at_random(seq_type, label)
    n = len(image_filenames)
    patients = []
    for i in range(n):
        pattern = r"{}-(\d+)".format(seq_type)
        # Extract the patient number before .png
        match = re.search(pattern, image_filenames[i])
        if match:
            patient_number = int(match.group(1))
            patients.append(patient_number)
        else:
            print("Number not found")
    return patients

def subtype_frequency(subtype, seq_type='pre', label='pos'):
    """
    Calculate the frequency of cancer-containing slices for each subtype.

    Args:
        subtype (str): Subtype of cancer. Possible values: 'Luminal A', 'Luminal B', 'HER2+', 'TN'.
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        label (str, 'pos' or 'neg'): Label for the image directory. Defaults to 'pos'.

    Returns:
        tuple: A tuple containing the frequencies of cancer-containing slices for each subtype.
    """
    patients = patients_number(seq_type, label)
    list0,list1,list2,list3=random_sample_for_each_cancer_type(clinical_file_path)
    n0 = 0
    n1 = 0
    n2 = 0
    n3 = 0
    for i in range(len(patients)):
        if patients[i] in list0:
            n0 += 1
        elif patients[i] in list1:
            n1 += 1
        elif patients[i] in list2:
            n2 += 1
        elif patients[i] in list3:
            n3 += 1
    if subtype == 'Luminal A':
        return n0
    elif subtype == 'Luminal B':
        return n1
    elif subtype == 'HER2+':
        return n2
    elif subtype == 'TN':
        return n3

def count_patient_slices(seq_type='pre', label='pos'):
    """
    Count the number of patients and the corresponding number of cancer-containing slices.

    Args:
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        label (str, 'pos' or 'neg'): Label for the image directory. Defaults to 'pos'.

    Returns:
        collections.OrderedDict: An ordered dictionary containing the count of cancer-containing slices for each patient,
                                 sorted by patient numbers.
    """
    patients = patients_number(seq_type, label)
    sorted_patients = sorted(patients)
    elements_count = collections.Counter(patients)

    ordered_dict = collections.OrderedDict()

    for patient in sorted_patients:
        ordered_dict[patient] = elements_count[patient]

    return ordered_dict

def process_paths(seq_type='pre', label='pos', show=False):
    """
    Process the paths of patient slices and generate relevant information.

    This function takes sequence type, label, and show as inputs and processes the paths
    to generate patient numbers, frequencies of slices,
    slice numbers, and path information. Optionally, it can display the images based on
    the specified show parameter.

    Args:
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        label (str, 'pos' or 'neg'): Label for the image directory. Defaults to 'pos'.
        show (bool or int, optional): Specifies whether to display images. If an integer is provided,
        it limits the number of images to be displayed. Defaults to False.

    Returns:
        tuple: A tuple containing the patient numbers, frequencies of slices, slice numbers, and path information.
    """
    list0,list1,list2,list3=random_sample_for_each_cancer_type(clinical_file_path)
    p = []
    freq = []
    slices = []
    path = []

    counter = 1
    patient_slices=count_patient_slices(seq_type, label)
    target_png_dir = dataset_path + '/target_{}'.format(seq_type)
    image_dir = os.path.join(target_png_dir, label)
    image_filenames = os.listdir(image_dir)
    patients=patients_number(seq_type, label)
    for i in  list0 +  list1 + list2 +  list3:
        if i in patient_slices:
            value = patient_slices[i]
            p.append(i)
            freq.append(value)

        n = len(image_filenames)
        for j in range(n):
            if patients[j] == i:
                sample_image_path = os.path.join(image_dir, image_filenames[j])
                # Extract the slice number before the name of sequence
                pattern = fr'\d+-(\d+)-{seq_type}'
                match = re.search(pattern, sample_image_path)
                if match:
                    slice_number = int(match.group(1))
                else:
                    raise ValueError("Slice number not found")
                if show and (isinstance(show, int) and counter <= show):
                    print("Image Counter:", counter)
                    print("This is slice",slice_number,"derived from patient", i)
                    if i in  list0:
                        print("Patient", i, "of type Luminal A has", value, "cancer-containing slices")
                    elif i in  list1:
                        print("Patient", i, "of type Luminal B has", value, "cancer-containing slices")
                    elif i in  list2:
                        print("Patient", i, "of type HER2+ has", value, "cancer-containing slices")
                    elif i in  list3:
                        print("Patient", i, "of type TN has", value, "cancer-containing slices")
                    else:
                        print("There is something wrong!")
                    print(sample_image_path)
                    display(Image(filename=sample_image_path))
                    counter += 1


                slices.append(slice_number)
                path.append(sample_image_path)
                #print(sample_image_path)

    return p, freq, slices, path

def sort_slices(seq_type='pre', label='pos'):
    """
    Sort the slices based on slice number.

    Args:
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        label (str, 'pos' or 'neg'): Label for the image directory. Defaults to 'pos'.

    Returns:
        list: A list containing the sorted slices.
    """
    list0,list1,list2,list3=random_sample_for_each_cancer_type(clinical_file_path)
    p, freq, slices, path=process_paths(seq_type, label)
    sorted_slices = slices.copy()  # Initialize sorted_Slices with a copy of Slices
    sum = 0
    list_= list0+ list1+ list2+ list3
    for i in range(len(list_)):
        if i == 0:
            l = int(freq[0])
            sorted_slices[0:l] = sorted(slices[0:l])
        else:
            sum = sum + freq[(i-1)]
            m = sum + freq[i]
            sorted_slices[sum:m] = sorted(slices[sum:m])
    padded_numbers = [str(number).zfill(3) for number in sorted_slices]
    return padded_numbers

def sort_paths(seq_type='pre', label='pos'):
    """
    Sort the paths based on sorted slices.
    This function takes a list of sorted slices as input and modifies the paths accordingly.
    Each path in the original path list is replaced with a new path that includes the corresponding
    sorted slice number. The modified paths are appended to a new list and returned.

    Args:
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        label (str, 'pos' or 'neg'): Label for the image directory. Defaults to 'pos'.
    Returns:
        list: A list of modified paths with sorted slice numbers.
    """
    sorted_slices=sort_slices(seq_type, label)
    p, freq, slices, path=process_paths(seq_type, label)
    sorted_path = []
    for i in range(len(sorted_slices)):
        path_ = path[i]
        sorted_slice_ =sorted_slices[i]
        pattern = fr'\d+-(\d+)-{seq_type}'
        match = re.search(pattern, path[i])
        if match:
            slice_number = match.group(1)
            slice_number=slice_number.zfill(3)

        else:
            raise ValueError("Slice number not found")

        p = path_.replace(slice_number, sorted_slice_, 1)
        sorted_path.append(p)
    return sorted_path

from PIL import Image
def plot_cropped_images(boxes_path,seq_type='pre', label='pos', show=False):
    """
    Plot cropped images with bounding boxes.

    This function takes a list of sorted paths as input and plots the corresponding cropped images
    with bounding boxes. The bounding box coordinates are obtained from a DataFrame called 'boxes_df'
    using the patient number extracted from the path.

    Args:
        boxes_path (str): Path to the CSV file containing tumor bounding box information.
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        label (str, 'pos' or 'neg'): Label for the image directory. Defaults to 'pos'.
        show (bool, optional): Whether to show the plotted images. Defaults to False.

    Returns:
        None
    """
    boxes_df = pd.read_csv(boxes_path)
    sorted_paths=sort_paths(seq_type, label)
    for i in range (len(sorted_paths)):
        sorted_path=sorted_paths[i]
        pattern = r"{}-(\d+)".format(seq_type)
        # Extract the patient number before .png
        match = re.search(pattern,sorted_path)
        if match:
            patient_num = int(match.group(1))
        else:
            print("Number not found")


        left = boxes_df.iloc[(patient_num - 1)][3]
        right = boxes_df.iloc[(patient_num - 1)][4]
        top = boxes_df.iloc[(patient_num - 1)][1]
        bottom = boxes_df.iloc[(patient_num - 1)][2]
        width = right - left
        height = bottom - top
        img=Image.open(sorted_path)
        fig, ax = plt.subplots()
        title = f"{i+1}: {sorted_path} {width}*{height}"
        rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(title, fontsize=10)
        plt.imshow(img, cmap="gray")
        if show and (isinstance(show, int) and i < show):
            plt.show()
        else:
            plt.close()
            break

def crop_and_save_images(boxes_path,seq_type='pre',crop='original', label='pos', show=False):
    """
    This function crops the images based on the crop argument and also bounding box coordinates
    obtained from a DataFrame called 'boxes_df'. The cropped images are then saved in the
    'resized_images'/+seq_type+'_'+crop directory. Optionally, the function can display the cropped images
    if the 'show' parameter is set to True or an integer value.

    Args:
        boxes_path (str): Path to the CSV file containing tumor bounding box information.
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        label (str, 'pos' or 'neg'): Label for the image directory. Defaults to 'pos'.
        crop (Union[str, int]): The method and size to crop. It can be either 'original' or an optional
        integer such as 32 or 64. Defaults to 'original'.
        show (bool, optional): Whether to show the plotted images. Defaults to False.

    Returns:
        None
    """
    boxes_df = pd.read_csv(boxes_path)
    sorted_paths=sort_paths(seq_type, label)
    print("crop:",crop,"  ","sequence:",seq_type)
    for i in range (len(sorted_paths)):
        sorted_path=sorted_paths[i]
        pattern = r"{}-(\d+)".format(seq_type)
        # Extract the patient number before .png
        match = re.search(pattern,sorted_path)
        if match:
            patient_num = int(match.group(1))
        else:
            print("Number not found")
        img=Image.open(sorted_path)
        left = boxes_df.iloc[(patient_num - 1)][3]
        right = boxes_df.iloc[(patient_num - 1)][4]
        top = boxes_df.iloc[(patient_num - 1)][1]
        bottom = boxes_df.iloc[(patient_num - 1)][2]
        w = right - left
        h = bottom - top
        mid_w=(right+left)/2
        mid_h= (bottom+top)/2
        if crop=='original':
            img_cropped = img.crop((left, top, right, bottom))
            img_cropped= img_cropped.resize((64,64))
        elif type(crop)==int:
            img_cropped = img.crop(((mid_w-(crop/2)),(mid_h-(crop/2)), (mid_w+(crop/2)), (mid_h+(crop/2))))

        fig, ax = plt.subplots()
        title = f"{i+1}: {sorted_path} {w}*{h}"
        ax.set_title(title, fontsize=10)
        plt.imshow(img_cropped, cmap="gray")

        if type(crop)==int:
            crop=str(crop)
        output_directory=bc_mri_path+'/resized_images/'+seq_type+'_'+crop
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        if crop!='original':
            crop=int(crop)

        filename = f"img_{seq_type}_{i:04d}.png"
        output_path = os.path.join(output_directory, filename)
        # Check if the folder already exists
        if not os.path.exists(output_path):
        # If the folder does not exist, save it
          img_cropped.save(output_path, 'png')
        if show and (isinstance(show, int) and i < show):
            plt.show()
        else:
            plt.close()

    print("Cropping and saving completed.")


def extract_pixels(seq_type='pre',crop='original'):
    """
    Extract pixels and store them in a matrix.

    This function takes a sequence type and crop as inputs to extracts the pixels from images located at the specified path.

    Args:
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        crop (Union[str, int]): The method and size to crop. It can be either 'original' or an optional
        integer such as 32 or 64. Defaults to 'original'.
    Returns:
        numpy.ndarray: Matrix containing the extracted pixel values.
    """
    sorted_paths=sort_paths(seq_type,label='pos')
    n = len(sorted_paths)
    if crop=='original':
        img_size=64
    else:
        img_size=crop
    pixel = np.zeros((n, img_size, img_size), dtype=np.int16, order="F")
    for i in range(n):
        # Construct the path using the crop and seq_type arguments
        path = f"{bc_mri_path}/resized_images/{seq_type}_{crop}/img_{seq_type}_{str(i).zfill(4)}.png"
        pixel[i, :, :] = cv.imread(path, 0)
    return pixel

def plot_images(seq_type='pre',crop='original', num_images=100):
    """
    Plot images from a pixel matrix.

    This function takes a sequence type and crop as inputs to plot up to the num_images. The axes, ticks, and grid lines are
    turned off to provide a clean visualization. The grayscale colormap is used to display the images. The
    title of each subplot indicates the image index.

    Args:
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        crop (Union[str, int]): The method and size to crop. It can be either 'original' or an optional
        integer such as 32 or 64. Defaults to 'original'.
        num_images (int): Number of images to plot.

    Returns:
        None
    """
    pixel = extract_pixels(seq_type,crop)
    plt.figure(figsize=(60, 60))

    for i in range(num_images):
        plt.subplot(20, 20, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(pixel[i, :, :], cmap='gray')
        plt.title(f"Image {i+1}")

    plt.show()

def Include(seq_type='pre', label='pos', s=3):
    """
    Narrow down to the middle s slices containing cancer.

    This function takes sequence type and label to obtain the frequency of cancer-containing slices per patient.
    This function also needs an optional parameter 's' as input. The function creates an array called 'include'
    with the length of n (toatal number of cancer-containing slices) and initializes all elements to zero.
    It then iterates over the frequencies and determines whether to include or exclude slices based on the value
    of 's'. If the frequency is less than or equal to 's', all the slices for that patient will be included.
    If the frequency is greater than 's', the function excludes additional slices from each side to have only
    the s middle cancer-containing slices.
    Args:
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        label (str, 'pos' or 'neg'): Label for the image directory. Defaults to 'pos'.
        s (int, optional): Number of middle slices to include. Defaults to 3.

    Returns:
        numpy.ndarray: Array indicating whether to include (1) or exclude (0) each slice.
    """
    p, freq, slices, path=process_paths(seq_type, label)
    sum = 0
    n = len(path)
    include = np.zeros(n, dtype=int)

    for i in freq:
        if i <= s:
            include[sum:(sum+i)] = 1
        else:
            r = i - s
            t = r / 2
            t = math.ceil(t)
            include[(sum+t):(sum+t+s)] = 1

        sum = sum + i

    return include

def threshold_segmentation(seq_type='pre',crop='original', threshold=25, show=False):
    """
    Perform threshold segmentation on a pixel array.

    This function applies threshold segmentation to a pixel array. It takes a sequence type and crop
    as inputs to obtain pixel along with an optional 'threshold' parameter, which defaults to 30.
    The 'show' parameter, if set to True, enables the visualization of intermediate results for a specified
    number of images. The function initializes a mask array of the same shape as 'pixel' with zeros.
    It also initializes an array 'T' to store the counts of pixels above the threshold for each image.
    The function then iterates over the images in the 'pixel' array and applies threshold segmentation.
    If 'show' is True and the 'show' parameter is an integer value and the current image index is less than
    'show', the original image and its corresponding mask are displayed side by side.

    Within the threshold segmentation loop, the function checks each pixel value in the current image. If the
    pixel value is greater than or equal to the threshold, the corresponding pixel in the mask is set to 1, and
    the count 'T' for the current image is incremented.

    After processing all the images, the elapsed time is printed, and the mask array is returned.

    Args:
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        crop (Union[str, int]): The method and size to crop. It can be either 'original' or an optional
        threshold (int, optional): Threshold value for segmentation. Defaults to 25.
        show (bool or int, optional): Flag indicating whether to display intermediate results. If set to True,
            all images will be displayed. If set to an integer value, only the specified number of images will be
            displayed. Defaults to False.

    Returns:
        numpy.ndarray: Mask array representing the segmented images.

    """
    pixel=extract_pixels(seq_type,crop)
    n = pixel.shape[0]
    if crop=='original':
        img_size=64
    else:
        img_size=crop
    mask = np.zeros((n, img_size, img_size), dtype=int)
    T = np.zeros(n, dtype=int)
    start = time.time()

    for i in tqdm(range(n)):
        image = pixel[i, :, :]

        if show and (isinstance(show, int) and i < show):
            fig, axes = plt.subplots(1, 2, figsize=(8, 8))
            ax = axes.flatten()
            ax[0].imshow(image, cmap="gray")
            ax[0].set_axis_off()
            ax[0].set_title(str(i+1)+ " "+ "Image (left) & Mask (right) ", fontsize=10)

        for j in range(img_size):
            for k in range(img_size):
                if image[j, k] >= threshold:
                    mask[i, j, k] = 1
                    T[i] += 1

        if show and (isinstance(show, int) and i < show):
            ax[1].imshow(mask[i, :, :], cmap="gray")
            ax[1].set_axis_off()
            title = "greaterthan"+" "+str(threshold)+"=" +str(T[i]) + "  " + "mean="+str(image.mean())
            ax[1].set_title(title, fontsize=10)
            plt.show()

    end = time.time()
    print("The elapsed time is:", end - start)

    return mask

def feature_extraction(seq_type='pre',crop='original', threshold=25,s=3,label='pos'):
    """
    Extracts 369 radiomics features from each slice using its corresponding mask and include array to
    determine selection.The function requires a sequence type and crop as inputs to obtain pixel. It also
    requires threshold argument to obtain mask Radiomics features are extracted using the Pyfeats library and
    21 feature classes.
    The selected 21 feature classes are:
        - First Order Statistics (FOS)
        - Gray Level Co-occurrence Matrix (GLCM)
        - Gray Level Difference Statistics (GLDS)
        - Neighborhood Gray Tone Difference Matrix (NGTDM)
        - Statistical Feature Matrix (SFM)
        - Law's Texture Energy Measures (LTE/TEM)
        - Fractal Dimension Texture Analysis (FDTA)
        - Gray Level Run Length Matrix (GLRLM)
        - Fourier Power Spectrum (FPS)
        - Shape Parameters
        - Gray Level Size Zone Matrix (GLSZM)
        - Higher Order Spectra (HOS)
        - Local Binary Pattern (LBP)
        - Gray-Scale Morphological Analysis
        - Histogram
        - Multi-Region Histogram
        - Amplitude Modulation - Frequency Modulation (AM-FM)
        - Discrete Wavelet Transform (DWT)
        - Gabor Transform (GT)
        - Zernike's Moments
        - Hu's Moments

    Args:
        seq_type (str): Sequence type. 'pre', 'post_1', 'post_2', or 'post_3'. Defaults to 'pre'.
        crop (Union[str, int]): The method and size to crop. It can be either 'original' or an optional
        threshold (int, optional): Threshold value for segmentation. Defaults to 25.
        s (int, optional): Number of middle slices to include. Defaults to 3.
        label (str, 'pos' or 'neg'): Label for the image directory. Defaults to 'pos'.

    Returns:
        numpy.ndarray: Array containing the extracted radiomics features for each image.
    Remark:
    Please note that the execution time of the process may vary and could take several hours depending on
    the processing power of your system.

    """
    start=time.time()
    pixel=extract_pixels(seq_type,crop)
    mask=threshold_segmentation(seq_type,crop, threshold,show=False)
    include=Include(seq_type, label,s)
    n=pixel.shape[0]
    data=np.zeros((n,369))
    for i in tqdm(range(n)):
        if (include[i]==1):
            img=pixel[i,:,:]
            mask_=mask[i,:,:]
            features1, labels = fos(img,mask_)
            l1=len(features1)
            data[i,0:l1]=features1
            features_mean2, features_range3, labels_mean, labels_range = glcm_features(img, ignore_zeros=True)
            l2=len(features_mean2)
            l3=len(features_range3)
            data[i,l1:l1+l2]=features_mean2
            data[i,l1+l2:l1+l2+l3]=features_range3
            features4, labels = glds_features(img, mask_, Dx=[0,1,1,1], Dy=[1,1,0,-1])
            l4=len(features4)
            data[i,l1+l2+l3:l1+l2+l3+l4]=features4
            features5, labels = ngtdm_features(img, mask_, d=1)
            l5=len(features5)
            data[i,l1+l2+l3+l4:l1+l2+l3+l4+l5]=features5
            features6, labels = sfm_features(img, mask_, Lr=4, Lc=4)
            l6=len(features6)
            data[i,l1+l2+l3+l4+l5:l1+l2+l3+l4+l5+l6]=features6
            features7, labels = lte_measures(img, mask_, l=7)
            l7=len(features7)
            data[i,l1+l2+l3+l4+l5+l6:l1+l2+l3+l4+l5+l6+l7]=features7
            h8, labels = fdta(img, mask_, s=2)
            l8=len(h8)
            data[i,l1+l2+l3+l4+l5+l6+l7:l1+l2+l3+l4+l5+l6+l7+l8]=h8
            features9, labels = glrlm_features(img, mask_, Ng=256)
            l9=len(features9)
            data[i,l1+l2+l3+l4+l5+l6+l7+l8:l1+l2+l3+l4+l5+l6+l7+l8+l9]=features9
            features10, labels = fps(img, mask_)
            l10=len(features10)
            S10=l1+l2+l3+l4+l5+l6+l7+l8+l9+l10
            data[i,l1+l2+l3+l4+l5+l6+l7+l8+l9:S10]=features10
            features11, labels = shape_parameters(img, mask_, perimeter=4, pixels_per_mm2=1)
            l11=len(features11)
            data[i,S10:S10+l11]=features11
            features12, labels = glszm_features(img, mask_)
            l12=len(features12)
            data[i,S10+l11:S10+l11+l12]=features12
            features13, labels = hos_features(img, th=[135,140])
            l13=len(features13)
            data[i,S10+l11+l12:S10+l11+l12+l13]=features13
            features14, labels = lbp_features(img, mask_, P=[8,16,24], R=[1,2,3])
            l14=len(features14)
            data[i,S10+l11+l12+l13:S10+l11+l12+l13+l14]=features14
            pdf15, cdf16 = grayscale_morphology_features(img,30)
            l15=len(pdf15)
            l16=len(cdf16)
            data[i,S10+l11+l12+l13+l14:S10+l11+l12+l13+l14+l15]=pdf15
            data[i,S10+l11+l12+l13+l14+l15:S10+l11+l12+l13+l14+l15+l16]=cdf16
            H17, labels = histogram(img, mask_, bins=5)
            l17=len(H17)
            data[i,S10+l11+l12+l13+l14+l15+l16:S10+l11+l12+l13+l14+l15+l16+l17]=H17
            features18, labels = multiregion_histogram(img, mask_, bins=5, num_eros=3, square_size=3)
            l18=len(features18)
            data[i,S10+l11+l12+l13+l14+l15+l16+l17:S10+l11+l12+l13+l14+l15+l16+l17+l18]=features18
            features19, labels = amfm_features(img, bins=32)
            l19=len(features19)
            data[i,S10+l11+l12+l13+l14+l15+l16+l17+l18:S10+l11+l12+l13+l14+l15+l16+l17+l18+l19]=features19
            features20, labels = dwt_features(img, mask_, wavelet='bior3.3', levels=1)
            l20=len(features20)
            S20=S10+l11+l12+l13+l14+l15+l16+l17+l18+l19+l20
            data[i,S10+l11+l12+l13+l14+l15+l16+l17+l18+l19:S20]=features20
            features21, labels = gt_features(img, mask_, deg=4, freq=[0.05, 0.4])
            l21=len(features21)
            data[i,S20:S20+l21]=features21
            features22, labels = zernikes_moments(img, radius=9)
            l22=len(features22)
            data[i,S20+l21:S20+l21+l22]=features22
            features23, labels = hu_moments(img)
            l23=len(features23)
            data[i,S20+l21+l22:S20+l21+l22+l23]=features23
            #fd24, labels = hog_features(img, ppc=10, cpb=3)
            #l24=len(fd24)
            #data[i,S20+l21+l22+l23:S20+l21+l22+l23+l24]=fd24
    end=time.time()
    print("The elapsed time is:",end-start)
    return data

def take_average(df):
    """
    Combine every 3 samples of the input data by taking the average.

    Args:
        df (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: The combined data with every 3 samples averaged.
    """
    data = np.zeros((int(df.shape[0]/3), df.shape[1]))
    k = 0

    for i in range(int(df.shape[0]/3)):
        avg = df.iloc[k:k+3, :].mean()
        data[i, :] = avg
        k += 3

    data = pd.DataFrame(data=data)
    return data

def initial_feature_selection_var(df, std=0.01, percentile=0.05, feature_stat=False):
    """
    Perform initial feature selection based on variance and variety.

    Parameters:
        df (DataFrame): Input DataFrame containing the features.
        std (float): Threshold value to determine low variance. Features with a standard
            deviation below this threshold will be considered to have low variance.
            Default is 0.01.
        percentile (float): Percentile value used to determine low variety. Features with
            a range (difference between the pth and (1-p)th percentiles) less than or equal
            to this percentile will be considered to have low variety. Default is 0.05.
        feature_stat (bool): Flag to determine whether to display the statistics (min, mean,
            max, and standard deviation) for each feature. If True, the statistics will be
            printed for each feature. Default is False.

    Returns:
        small_var (list): List of column indices corresponding to features with low variance.
        low_variety (list): List of column indices corresponding to features with low variety.

    """
    small_var = []
    low_variety = []

    for i in range(df.shape[1]):
        des = df.iloc[:, i].describe()
        min_val = des.iloc[3]
        mean = des.iloc[1]
        max_val = des.iloc[7]
        std_ = des.iloc[2]
        p1 = df.iloc[:, i].quantile(percentile)
        p2 = df.iloc[:, i].quantile(1 - percentile)
        q3 = des.iloc[6]

        if std_ < std:
            small_var.append(i)
        if p1 == p2:
            low_variety.append(i)
        if feature_stat:
            print(i, "min:", min_val, "mean:", mean, "max:", max_val, "std:", std)

    return small_var, low_variety

def find_zero_variance_features(df):
    """
    Find features in the given data that have zero variance (i.e constant features).

    Args:
        df (pd.DataFrame): Input data.

    Returns:
        list: List of column indices with zero variance.
    """
    zero_var = []
    l = 0

    for i in range(df.shape[1]):
        if df.iloc[:, i].min() == df.iloc[:, i].max():
            zero_var.append(i)
            l += 1

    print("The number of features with variance 0 is:", l)
    return zero_var

def initial_feature_selection_corr(df, corr=0.98, show=False):
    """
    Find pairs of features in the given data that have a strong correlation (greater than or equal to corr).

    Parameters:
        df (pd.DataFrame): Input data as a pandas DataFrame.
        corr (float): Threshold value for defining strong correlation. Features with a correlation
            greater than or equal to this value will be considered strongly correlated. Default is 0.98.
        show (bool): Flag to determine whether to display the correlated pairs. If True, the correlated
            pairs will be printed. Default is False.

    Returns:
        list: List of column indices representing features with strong correlations.

    """
    Cor = df.corr()
    cnt = 0
    str_cor = []

    for i in range(0, df.shape[1]):
        for j in range(i + 1, df.shape[1]):
            if Cor.iloc[i, j] >= corr:
                str_cor.append(j)
                cnt += 1
                if show:
                    print(cnt, i, j, Cor.iloc[i, j])
    # Get unique column indices with strong correlations
    ind_cor = [*set(str_cor)]
    return ind_cor

def process_clinical_features_extract_labels(path):
    """
    Process clinical features data and extract labels.

    Args:
        path (str): Path to the clinical features data file.

    Returns:
        tuple: A tuple containing the processed DataFrame (including the selected clinical features) and
        corresponding labels.

    """
    list0,list1,list2,list3=random_sample_for_each_cancer_type(clinical_file_path)
    list_ = list0 + list1 + list2 + list3
    patinets_indices_in_sample = [x - 1 for x in list_]
    clinical_features = pd.read_csv(path)
    label = clinical_features.iloc[2:, 26]
    label = label.astype(int)
    label = label.reset_index(drop=True)
    # In most cases, these columns primarily or entirely consist of string values, and their meaning or significance is unknown to me.
    columns_to_drop_redundant = [0, 7, 9, 15, 16, 23, 24, 25, 26, 27, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                 58, 59, 61, 62, 63, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                                 87, 92, 93, 94, 95, 96, 97]

    # Drop some redundant features at the very first
    df = clinical_features.drop(clinical_features.columns[columns_to_drop_redundant], axis=1)

    # Convert the side of the cancer into numerical values ('L': -1, 'R': +1, 'not given': 0)
    for i in range(2, df.shape[0]):
        if df.iloc[i, 24] == 'L':
            df.iloc[i, 24] = -1
        elif df.iloc[i, 24] == 'R':
            df.iloc[i, 24] = +1
        else:
            df.iloc[i, 24] = 0

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df1 = df.iloc[2:, :]

    # Drop features associated with treatments
    cols_to_drop_treatments = [32, 33, 34, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    df2 = df1.drop(df1.columns[cols_to_drop_treatments], axis=1, inplace=False)

    # Drop features associated with tumor grades
    cols_to_drop_tumor_grades = [18, 19, 20, 21, 22, 23]
    df3 = df2.drop(df2.columns[cols_to_drop_tumor_grades], axis=1, inplace=False)

    df4 = df3.iloc[patinets_indices_in_sample, :]
    labels_in_sample = label[patinets_indices_in_sample]

    return df4, labels_in_sample

def confidence_interval(vec, percent=0.90):
    """
    Analyze a vector by calculating the mean, standard deviation, and constructing a confidence interval.

    Parameters:
        vec (array-like): The vector to be analyzed.
        percent (float): The percentage (between 0 and 100) to contract the confidence interval.

    Returns:
        None: Prints the mean, standard deviation, and confidence interval.

    """
    n = len(vec)
    mean = np.mean(vec)
    std = np.std(vec, ddof=1)
    #print(vec)
    print("Mean:", mean)
    print("Standard deviation:", std)
    t_critical = t.ppf((1-((1-percent)/2)), n - 1)
    margin_of_error = t_critical * std / np.sqrt(n)
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    print("{:.1f}% Confidence interval for the mean:".format(percent * 100), confidence_interval)

def calculate_average_or_mode(data):
    """
    Calculate the average for numerical values and the mode for non-numerical values in a list of dictionaries.

    Parameters:
        data (list): A list of dictionaries containing key-value pairs.

    Returns:
        dict: A dictionary with keys representing the keys in the input dictionaries and values representing the calculated average or mode.

    """
    # initialize dictionaries to store numerical and non-numerical elements
    numeric_values = {}
    non_numeric_values = {}

    # iterate over each dictionary in the list
    for d in data:
        # iterate over each key-value pair in the dictionary
        for key, value in d.items():
            # check if the value is numerical
            if isinstance(value, (int, float)):
                # if the key is not already in the dictionary, initialize it
                if key not in numeric_values:
                    numeric_values[key] = []
                # append the value to the list associated with the key
                numeric_values[key].append(value)
            else:
                # if the key is not already in the dictionary, initialize it
                if key not in non_numeric_values:
                    non_numeric_values[key] = []
                # append the value to the list associated with the key
                non_numeric_values[key].append(value)

    # calculate the average of numerical values and mode of non-numerical values
    result = {}
    for key, value in numeric_values.items():
        result[key] = sum(value) / len(value)
    for key, value in non_numeric_values.items():
        result[key] = mode(value)

    return result

def convert_label_one_vs_the_rest(data, subtype):
    """
    Convert the original labels into binary format (one vs. the rest) based on the specified subtype.

    Parameters:
         data (pandas.DataFrame): The input data including features, patient numbers, and the labels.
        subtype (str or int): The subtype to convert into binary format. Can be a string ('Luminal A', 'Luminal B', 'HER2+', 'TN')
                              or the corresponding integers (0, 1, 2, 3).

    Returns:
        tuple: Tuple containing the original feature matrix (X) and target vector (Y)
              for the specified one vs. the rest classification.

    """
    f_num=data.shape[1]-2
    X_=data.iloc[:,0:f_num]
    Y_=data.iloc[:,-1]
    n_ = len(Y_)
    binary_label = np.zeros(n_, dtype=int)

    if isinstance(subtype, str):
        subtype = ['Luminal A', 'Luminal B', 'HER2+', 'TN'].index(subtype)

    for i in range(n_):
        if Y_[i] == subtype:
            binary_label[i] = 1

    return X_,binary_label

def convert_label_one_vs_one(data, subtype_1, subtype_2):
    """
    Process the data based on the specified subtypes.

    Parameters:
        The input data including features, patient numbers, and the labels.
        subtype_1 (int or str): First subtype, specified as either an integer or a string.
        subtype_2 (int or str): Second subtype, specified as either an integer or a string.
        ('Luminal A', 'Luminal B', 'HER2+', 'TN') or the corresponding integers (0, 1, 2, 3).

    Returns:
        tuple: Tuple containing processed feature matrix (X) and target vector (Y)
              for the specified pair of subtypes.

    """
    subtype_values = {
        'Luminal A': 0,
        'Luminal B': 1,
        'HER2+': 2,
        'TN': 3
    }

    f_num = data.shape[1] - 2

    if isinstance(subtype_1, str):
        subtype_1 = subtype_values[subtype_1]
    if isinstance(subtype_2, str):
        subtype_2 = subtype_values[subtype_2]

    data_i = data[data.iloc[:, f_num + 1].isin([subtype_1, subtype_2])]
    data_i = data_i.reset_index(drop=True)
    X_i = data_i.iloc[:, 0:f_num]
    Y_i = data_i.iloc[:, -1]

    return X_i, Y_i

def anova_feature_selection(X_train,y_train,X_test,n_features):
    """
    Perform ANOVA feature selection on the given datasets.

    Args:
        X_train (pd.DataFrame): Training dataset features.
        y_train (pd.Series): Training dataset labels.
        X_test (pd.DataFrame): Test dataset features.
        n_features (int): Number of top features to select.

    Returns:
        tuple: A tuple containing the transformed training and test datasets.
            - X_train_anov (pd.DataFrame): Transformed training dataset with the selected features.
            - X_test_anov (pd.DataFrame): Transformed test dataset with the selected features.
    """
    best = SelectKBest(k=n_features)
    fit_train = best.fit(X_train, y_train)
    X_train_anov = best.transform(X_train)
    X_test_anov = best.transform(X_test)
    return X_train_anov, X_test_anov

def evaluate_classifier(X, y, k_fold_cv=10, random_search_cv=5, n_iter=200,
                        max_features=150, classifier='None',n_neighbors_impute=10,n_neighbors_LOF=10,
                        hyperparameters=None,random_state=42):
    """
    Evaluate a classifier's performance on given data.

    Args:
        X (array-like): The feature matrix.
        y (array-like): The target vector.
        k_fold_cv (int, optional): The number of cross-validation folds. Defaults to 10.
        random_search_cv (int, optional): The number of iterations for randomized search. Defaults to 5.
        n_iter (int, optional): The number of iterations for randomized search. Defaults to 500.
        max_features (int, optional): The maximum number of features to consider for feature selection. Defaults to 150.
        classifier (str, optional): The classifier type. Must be 'svm' or 'rf'. Defaults to 'None'.
        n_neighbors_impute (int, optional): The number of neighbors for KNN imputation. Defaults to 5.
        n_neighbors_LOF (int, optional): The number of neighbors for Local Outlier Factor. Defaults to 5.
        hyperparameters (dict, optional): Hyperparameters for the classifier. Defaults to None.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing the maximum test score, optimal features, optimal number of features,
               and optimal parameters.

    """
    max_test_score = []
    optimal_features = []
    optimal_num_features = []
    optimal_param = []

    # Perform k-fold cross-validation
    kf = KFold(n_splits=k_fold_cv, shuffle=True, random_state=random_state)
    for train_index, test_index in tqdm(kf.split(X)):
        start = time.time()
        test_score = []

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        imputer = KNNImputer(n_neighbors=n_neighbors_impute)
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_imputed = imputer.fit_transform(X_train_scaled)
        X_test_scaled = scaler.transform(X_test)
        X_test_imputed = imputer.transform(X_test_scaled)

        # Loop for ANOVA feature selection
        for i in tqdm(range(1, max_features+1)):
            # Call the anova feature selection function
            X_train_anov,X_test_anov=anova_feature_selection(X_train_imputed,y_train,X_test_imputed,n_features=i)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors_LOF, contamination='auto')
            y_pred_train = lof.fit_predict(X_train_anov)
            X_train_inliers = X_train_anov[y_pred_train == 1]
            y_train_inliers = y_train[y_pred_train == 1]

            print("Training data shape:", X_train_anov.shape)
            print("Cleaned training data shape:", X_train_inliers.shape)

            if classifier == 'svm':
                from sklearn import svm
                classifier_obj = svm.SVC()
            elif classifier == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                classifier_obj = RandomForestClassifier()
            f1 = make_scorer(f1_score, average='macro')
            Randomsearch = RandomizedSearchCV(classifier_obj, hyperparameters, cv=random_search_cv, n_iter=n_iter,
                                              scoring=f1,verbose=1, n_jobs=-1, random_state=42)
            Randomsearch.fit(X_train_inliers, y_train_inliers)
            test_score.append(Randomsearch.score(X_test_anov, y_test))
            del Randomsearch

        end = time.time()
        print("The elapsed time is:", end - start)
        print("\nThe random state is:", random)
        print(np.sort(test_score))

        max_test_score.append(np.max(test_score))
        best = SelectKBest(k=np.argmax(test_score) + 1)
        fit_train = best.fit(X_train_imputed, y_train)
        optimal_features.append(fit_train.get_support(indices=True))
        X_train_anov = best.transform(X_train_imputed)
        X_test_anov = best.transform(X_test_imputed)

        lof = LocalOutlierFactor(n_neighbors=n_neighbors_LOF, contamination='auto')
        y_pred_train = lof.fit_predict(X_train_anov)
        X_train_inliers = X_train_anov[y_pred_train == 1]
        y_train_inliers = y_train[y_pred_train == 1]

        if classifier == 'svm':
            from sklearn import svm
            classifier_obj = svm.SVC()
        elif classifier == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            classifier_obj = RandomForestClassifier()

        Randomsearch = RandomizedSearchCV(classifier_obj,hyperparameters,cv=random_search_cv,n_iter=n_iter,
                                          scoring=f1,verbose=1,n_jobs=-1,random_state=42)
        Randomsearch.fit(X_train_inliers,y_train_inliers)
        print("\n The number of feature is:",np.argmax(test_score)+1)
        optimal_num_features.append(np.argmax(test_score)+1)
        optimal_estimator=Randomsearch.best_estimator_
        optimal_estimator.fit(X_train_inliers,y_train_inliers)
        test_prediction =optimal_estimator.predict(X_test_anov)
        print("The optimal parameters are:", Randomsearch.best_params_)
        optimal_param.append(Randomsearch.best_params_)
        print("test score is:\t", (Randomsearch.score(X_test_anov,y_test)))
        print("train score is:\t",(Randomsearch.score(X_train_inliers,y_train_inliers)))
        print(confusion_matrix(y_test, test_prediction))
        print("Report: \n",classification_report(y_test, test_prediction))
        train_prediction =optimal_estimator.predict(X_train_inliers)
        print(confusion_matrix(y_train_inliers, train_prediction))
        print("Report: \n",classification_report(y_train_inliers, train_prediction))
    return max_test_score, optimal_features, optimal_num_features, optimal_param


def one_vs_the_rest_classification(data, subtype, k_fold_cv=2, random_search_cv=2, n_iter=5,
                        max_features=5, classifier='None',n_neighbors_impute=1,n_neighbors_LOF=1,
                        hyperparameters=None,random_state=42):
    """
    Perform one-vs-the-rest classification using the provided functions.

    Args:
        data (pandas.DataFrame): The input data including features, patient numbers, and labels.
        subtype (str or int): The subtype to determine the case of one-vs-the-rest classification. Can be a string ('Luminal A', 'Luminal B', 'HER2+', 'TN') or the corresponding integers (0, 1, 2, 3).
        k_fold_cv (int, optional): The number of cross-validation folds. Defaults to 2.
        random_search_cv (int, optional): The number of iterations for randomized search. Defaults to 2.
        n_iter (int, optional): The number of iterations for randomized search. Defaults to 10.
        max_features (int, optional): The maximum number of features to consider for feature selection. Defaults to 5.
        classifier (str, optional): The classifier type. Must be 'svm' or 'rf'. Defaults to 'None'.
        n_neighbors_impute (int, optional): The number of neighbors for KNN imputation. Defaults to 1.
        n_neighbors_LOF (int, optional): The number of neighbors for Local Outlier Factor. Defaults to 1.
        hyperparameters (dict, optional): Hyperparameters for the classifier. Defaults to None.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        None. Prints information regarding the chosen criteria, optimal hyperparameters, and number of features.
    """
    X, Y = convert_label_one_vs_the_rest(data, subtype)
    max_test_score, optimal_features, optimal_num_features, optimal_param = evaluate_classifier(X, Y, k_fold_cv, random_search_cv, n_iter,
                        max_features, classifier,n_neighbors_impute,n_neighbors_LOF,
                        hyperparameters,random_state)

    print('Max test scores:\n', max_test_score)
    print('Optimal features:\n', optimal_features)
    print('Optimal number of features:\n', optimal_num_features)
    print('Optimal parameters:\n', optimal_param)

    avg_mode = calculate_average_or_mode(optimal_param)
    print("Average of numerical hyperparameters and mode of string hyperparameters across different runs:", avg_mode)

    print("Confidence Interval for the mean of max_test_scores:\n")
    confidence_interval(max_test_score)

    print("Confidence Interval for the mean of optimal_num_features:\n")
    confidence_interval(optimal_num_features)

def one_vs_one_classification(data, subtype_1, subtype_2, k_fold_cv=2, random_search_cv=2, n_iter=5,
                        max_features=5, classifier='None',n_neighbors_impute=1,n_neighbors_LOF=1,
                        hyperparameters=None,random_state=42):
    """
    Perform one-vs-one classification using the provided functions.

    Args:
        data (pandas.DataFrame): The input data including features, patient numbers, and labels.
        subtype_1 (int or str): First subtype, specified as either an integer or a string.
        subtype_2 (int or str): Second subtype, specified as either an integer or a string.
        ('Luminal A', 'Luminal B', 'HER2+', 'TN') or the corresponding integers (0, 1, 2, 3).
        k_fold_cv (int, optional): The number of cross-validation folds. Defaults to 2.
        random_search_cv (int, optional): The number of iterations for randomized search. Defaults to 2.
        n_iter (int, optional): The number of iterations for randomized search. Defaults to 10.
        max_features (int, optional): The maximum number of features to consider for feature selection. Defaults to 5.
        classifier (str, optional): The classifier type. Must be 'svm' or 'rf'. Defaults to 'None'.
        n_neighbors_impute (int, optional): The number of neighbors for KNN imputation. Defaults to 1.
        n_neighbors_LOF (int, optional): The number of neighbors for Local Outlier Factor. Defaults to 1.
        hyperparameters (dict, optional): Hyperparameters for the classifier. Defaults to None.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        None. Prints information regarding the chosen criteria, optimal hyperparameters, and number of features.
    """
    X, Y = convert_label_one_vs_one(data,subtype_1,subtype_2 )
    max_test_score, optimal_features, optimal_num_features, optimal_param = evaluate_classifier(X, Y, k_fold_cv, random_search_cv, n_iter,
                        max_features, classifier,n_neighbors_impute,n_neighbors_LOF,
                        hyperparameters,random_state)

    print('Max test scores:\n', max_test_score)
    print('Optimal features:\n', optimal_features)
    print('Optimal number of features:\n', optimal_num_features)
    print('Optimal parameters:\n', optimal_param)

    avg_mode = calculate_average_or_mode(optimal_param)
    print("Average of numerical hyperparameters and mode of string hyperparameters across different runs:", avg_mode)

    print("Confidence Interval for the mean of max_test_scores:\n")
    confidence_interval(max_test_score)

    print("Confidence Interval for the mean of optimal_num_features:\n")
    confidence_interval(optimal_num_features)