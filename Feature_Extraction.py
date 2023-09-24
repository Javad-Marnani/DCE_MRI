#pip install -r requirements.txt
#!pip install pydicom
#!pip install patool
#!pip install pyunpack
#!pip install pyfeats
#!pip install openpyxl
##########################################################################################
#####################################   IMPORTS    #######################################
########################
# ##################################################################
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
from PIL import Image
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
####################################   SETTINGS    #######################################
##########################################################################################
_GPU = False

##########################################################################################
######################################   TEMP    #########################################
##########################################################################################
pd.set_option('display.max_columns', None)

##########################################################################################
#################################   SETTINGS EXEC    #####################################
##########################################################################################
'''Choose GPU 0 or 1 if they are available for processing.'''
if _GPU:
	physical_devices = tf.config.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[1], True)
	tf.config.set_visible_devices(physical_devices[1], 'GPU')
	visible_devices = tf.config.get_visible_devices('GPU')
	print(visible_devices)


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

directories_to_check = ["dataset", "extracted_features", "resized_images", "xlsx_csv_files"]
for folder in directories_to_check:
    folder_path = os.path.join(bc_mri_path, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

print("Please ensure that you have placed four XLSX files into the 'xlsx_csv_files' directory before running the process.")
# List of XLSX file paths
xlsx_file_paths = [xlsx_csv_files_path+ r'\Breast-Cancer-MRI-filepath_filename-mapping.xlsx',
                   xlsx_csv_files_path+ r'\Annotation_Boxes.xlsx',
                   xlsx_csv_files_path+ r'\Clinical_and_Other_Features.xlsx',
                   xlsx_csv_files_path+ r'\Imaging_Features.xlsx'
]
# Convert XLSX file paths to CSV file paths
csv_file_paths = [path[:-4] + "csv" for path in xlsx_file_paths]
# Iterate over the XLSX and CSV file paths
for xlsx_file, csv_file in zip(xlsx_file_paths, csv_file_paths):
    # Check if the CSV file already exists
    if os.path.exists(csv_file):
        # If the CSV file exists, skip this iteration
        continue

    # Read the XLSX file into a pandas DataFrame
    data_frame = pd.read_excel(xlsx_file)

    # Convert DataFrame to CSV format
    csv_data = data_frame.to_csv(index=False)

    # Write CSV data and save it in the target directory
    with open(csv_file, "w") as file:
        file.write(csv_data)
clinical_file_path = xlsx_csv_files_path + r'\Clinical_and_Other_Features.csv'
mapping_path = xlsx_csv_files_path + r'\Breast-Cancer-MRI-filepath_filename-mapping.csv'
boxes_path = xlsx_csv_files_path + r'\Annotation_Boxes.csv'
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

# Set the path to the clinical features file
path =clinical_file_path
# Call the function to generate random samples for each cancer type
list0, list1, list2, list3 = random_sample_for_each_cancer_type(path)

# Combine the first elements from each list into a single list
list_ = list0 + list1 + list2 + list3

# Print the length of the combined list
print("The total sample size is:", len(list_))

# Print the selected patients
print("Please download the following patient folders and put the folder Duke-Breast-Cancer-MRI in dataset directory:")
for folder in sorted(list_):
    print(folder)

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

for seq_type in types:
    print("Processing", seq_type, "contrast sequence ")
    mapping_df=filter_mapping_df(mapping_path, list_, seq_type)
    process_mapping_df(dataset_path, mapping_path, boxes_path, mapping_df, seq_type)
   

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

# Number of all cancer-containing slices derived from the selected patients
n = len(image_filenames_plot_one_at_random(seq_type='pre', label='pos', show=False))
print("Number of all cancer-containing slices derived from the selected patients is:",n)

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

#Frequencies of cancer-containing slices for each subtype
n0=subtype_frequency(subtype='Luminal A',seq_type='pre', label='pos')
n1=subtype_frequency(subtype='Luminal B',seq_type='pre', label='pos')
n2=subtype_frequency(subtype='HER2+',seq_type='pre', label='pos')
n3=subtype_frequency(subtype='TN',seq_type='pre', label='pos')
print("Frequencies of cancer-containing slices for each subtype are as follow:")
print('Luminal A:',n0,'\nLuminal B:',n1,'\nHER2+:',n2,'\nTN:',n3)

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

print("Dictionary containing the count of cancer-containing slices for each patient:")
print(count_patient_slices(seq_type='post_3', label='pos'))

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
    for i in list0 + list1 + list2 + list3:
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
                    if i in list0:
                        print("Patient", i, "of type Luminal A has", value, "cancer-containing slices")
                    elif i in list1:
                        print("Patient", i, "of type Luminal B has", value, "cancer-containing slices")
                    elif i in list2:
                        print("Patient", i, "of type HER2+ has", value, "cancer-containing slices")
                    elif i in list3:
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
    p, freq, slices, path=process_paths(seq_type, label)
    sorted_slices = slices.copy()  # Initialize sorted_Slices with a copy of Slices
    sum = 0
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
from PIL import Image

plot_cropped_images(boxes_path,seq_type='pre', label='pos', show=False)

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

for seq_type in tqdm(['pre', 'post_1', 'post_2', 'post_3']):
    for crop in  tqdm(['original',32,64]):
        crop_and_save_images(boxes_path,seq_type, crop,label='pos', show=False)

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

# Extract pixels from images and store them in a matrix
pixel = extract_pixels(seq_type='post_2',crop='original')
# Print the shape of the pixel matrix
print(np.shape(pixel))


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

# Plot the images from the 'pixel' matrix
plot_images(seq_type='pre',crop='original', num_images=False)

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

# Calculate the number of included slices in the 'include' array (s=3)
include=Include(seq_type='pre', label='pos', s=3)
u, c = np.unique(include, return_counts=True)
print("Number of included (1) and excluded (0) slices in the include array (s=3):\n",pd.Series(c, index=u))

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

# Perform threshold segmentation on the 'pixel' array with a threshold of 50.
mask = threshold_segmentation(seq_type='post_1',crop=64, threshold=50, show=False)
# Print the shape of the 'mask' array
print("The shape of generated masks:\n",mask.shape)

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
for seq_type in tqdm(['pre', 'post_1', 'post_2', 'post_3']):
      for crop in tqdm(['original', 32, 64]):
          print(f"Generating CSV file associated with sequence {seq_type} and crop {crop}")
          # Check if the CSV file already exists
          filename = f"{bc_mri_path}/extracted_features/{seq_type.capitalize()}_{str(crop).capitalize()}.csv"
          if os.path.exists(filename):
              # If the CSV file exists, skip this iteration
              continue
          # Obtain the data using the radiomics features
          data = feature_extraction(seq_type, crop, threshold=25, s=3, label='pos')
          D = pd.DataFrame(data=data)
          # Select rows where all columns are zero (related to the slices that were not selected)
          all_zeros = (D == 0).all(axis=1)
          # Drop the rows where all columns are zero
          D = D[~all_zeros]
          print("Total number of missing values:", D.isnull().sum().sum())
          # Write the data and save it as a CSV file
          D.to_csv(filename, header=True)
#Load the CSV file 'Pre_Original.csv' into a pandas DataFrame
D = pd.read_csv(bc_mri_path+'/extracted_features/Pre_Original.csv')
#Remove the first column from the DataFrame
D = D.iloc[:, 1:]
#Display the updated DataFrame
display(D)