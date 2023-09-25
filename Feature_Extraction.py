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
from Images_Functions import random_sample_for_each_cancer_type, filter_mapping_df,save_dcm_slice,process_mapping_df,image_filenames_plot_one_at_random,patients_number
from Images_Functions import subtype_frequency,count_patient_slices,process_paths,sort_slices,sort_paths,plot_cropped_images,crop_and_save_images,extract_pixels
from Images_Functions import plot_images,Include,threshold_segmentation,feature_extraction
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
directories_to_check = ["dataset", "extracted_features", "resized_images", "xlsx_csv_files"]
for folder in directories_to_check:
    folder_path = os.path.join(bc_mri_path, folder)
    if not os.path.exists(folder_path):
       os.makedirs(folder_path)
types = ['pre', 'post_1', 'post_2', 'post_3']
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
radiomics_clinical_path=bc_mri_path+r'\extracted_features\radiomics_clinical_features_data.csv'
features_by_saha=xlsx_csv_files_path + r'\Imaging_Features.csv'
def Feature_Extraction():
  # Call the function to generate random samples for each cancer type
  global list0, list1, list2, list3
  list0, list1, list2, list3 = random_sample_for_each_cancer_type(clinical_file_path)
  # Combine the first elements from each list into a single list
  global list_
  list_ = list0 + list1 + list2 + list3
  # Print the length of the combined list
  print("The total sample size is:", len(list_))
  # Print the selected patients
  print("Please download the following patient folders and put the folder Duke-Breast-Cancer-MRI in dataset directory:")
  for folder in sorted(list_):
      print(folder)
  print("Separating the cancer-containing slices (pos) from the non-cancer-containing slices (neg) and saving them in different directories")
  for seq_type in types:
      print("Processing", seq_type, "contrast sequence ")
      mapping_df=filter_mapping_df(mapping_path, list_, seq_type)
      process_mapping_df(dataset_path, mapping_path, boxes_path, mapping_df, seq_type)
  #Number of all cancer-containing slices derived from the selected patients
  n = len(image_filenames_plot_one_at_random(seq_type='pre', label='pos', show=False))
  print("Number of all cancer-containing slices derived from the selected patients is:",n)
  #Frequencies of cancer-containing slices for each subtype
  n0=subtype_frequency(subtype='Luminal A',seq_type='pre', label='pos')
  n1=subtype_frequency(subtype='Luminal B',seq_type='pre', label='pos')
  n2=subtype_frequency(subtype='HER2+',seq_type='pre', label='pos')
  n3=subtype_frequency(subtype='TN',seq_type='pre', label='pos')
  print("Frequencies of cancer-containing slices for each subtype are as follow:")
  print('Luminal A:',n0,'\nLuminal B:',n1,'\nHER2+:',n2,'\nTN:',n3)
  print("Dictionary containing the count of cancer-containing slices for each patient:")
  print(count_patient_slices(seq_type='post_3', label='pos'))
  plot_cropped_images(boxes_path,seq_type='pre', label='pos', show=False)
  print("Cropping and saving the images if they don't already exist in the 'resized_images' directory")
  for seq_type in tqdm(['pre', 'post_1', 'post_2', 'post_3']):
      for crop in  tqdm(['original',32,64]):
          crop_and_save_images(boxes_path,seq_type, crop,label='pos', show=False)
  #Extract pixels from images and store them in a matrix
  pixel = extract_pixels(seq_type='post_2',crop='original')
  # Print the shape of the pixel matrix
  print("shape of the pixel matrix is:", np.shape(pixel))
  # Plot the images from the 'pixel' matrix
  plot_images(seq_type='pre',crop='original', num_images=False)
  # Calculate the number of included slices in the 'include' array (s=3)
  include=Include(seq_type='pre', label='pos', s=3)
  u, c = np.unique(include, return_counts=True)
  print("Number of included (1) and excluded (0) slices in the include array (s=3):\n",pd.Series(c, index=u))
  # Perform threshold segmentation on the 'pixel' array with a threshold of 50.
  mask = threshold_segmentation(seq_type='post_1',crop=64, threshold=50, show=False)
  # Print the shape of the 'mask' array
  print("The shape of generated masks:\n",mask.shape)
  print("==============================================")
  print("||   FEATURE EXTRACTION IN PROGRESS...       ||")
  print("==============================================")
  # Iterate over seq_type and crop values
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
  #print("The data frame for pre contrast sequences and original crop is:")
  #display(D)
  print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
  print('- - - - - - - - - - - - - - - - FEATURE EXTRACTION DONE  - - - - - - - - - - - - - - - -')
  print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
  
# if __name__ == '__Feature_Extraction__':
Feature_Extraction()