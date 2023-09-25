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
from Data_Preprocessing_Functions import take_average,initial_feature_selection_var,find_zero_variance_features,initial_feature_selection_corr,process_clinical_features_extract_labels
from Classification_Functions import confidence_interval,calculate_average_or_mode,convert_label_one_vs_the_rest,convert_label_one_vs_one
from Classification_Functions import anova_feature_selection,evaluate_classifier,one_vs_the_rest_classification,one_vs_one_classification
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
def main():
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
  # Reading 12 datasets
  #Pre
  D1=pd.read_csv(bc_mri_path+r'\extracted_features\Pre_Original.csv')
  D1=D1.iloc[:,1:]
  D2=pd.read_csv(bc_mri_path+r'\extracted_features\Pre_64.csv')
  D2=D2.iloc[:,1:]
  D3=pd.read_csv(bc_mri_path+r'\extracted_features\Pre_32.csv')
  D3=D3.iloc[:,1:]
  #Post1
  D4=pd.read_csv(bc_mri_path+r'\extracted_features\Post_1_Original.csv')
  D4=D4.iloc[:,1:]
  D5=pd.read_csv(bc_mri_path+r'\extracted_features\Post_1_64.csv')
  D5=D5.iloc[:,1:]
  D6=pd.read_csv(bc_mri_path+r'\extracted_features\Post_1_32.csv')
  D6=D6.iloc[:,1:]
  #Post2
  D7=pd.read_csv(bc_mri_path+r'\extracted_features\Post_2_Original.csv')
  D7=D7.iloc[:,1:]
  D8=pd.read_csv(bc_mri_path+r'\extracted_features\Post_2_64.csv')
  D8=D8.iloc[:,1:]
  D9=pd.read_csv(bc_mri_path+r'\extracted_features\Post_2_32.csv')
  D9=D9.iloc[:,1:]
  #Post3
  D10=pd.read_csv(bc_mri_path+r'\extracted_features\Post_3_Original.csv')
  D10=D10.iloc[:,1:]
  D11=pd.read_csv(bc_mri_path+r'\extracted_features\Post_3_64.csv')
  D11=D11.iloc[:,1:]
  D12=pd.read_csv(bc_mri_path+r'\extracted_features\Post_3_32.csv')
  D12=D12.iloc[:,1:]
  # Merging 12 datasets
  frames=[D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12]
  data= pd.concat(frames, axis=1,join='inner',ignore_index=True)
  #display(data)
  #missing values
  print("The number of missing values in 12 combintations of extracted features is:",data.isnull().sum().sum())
  #infinite values
  print("The number of infinite values in 12 combintations of extracted features is:",np.isinf(data).values.sum())
  #columns including missing values
  cols_with_nan_data=list(np.where(data.isna().any(axis=0))[0])
  print("Columns with missing values in the extracted features are:\n ",cols_with_nan_data)
  #columns including inf values
  cols_with_inf_data=list(data.columns[np.isinf(data).any()])
  print("Columns with infinite values in the extracted features are:\n ",cols_with_inf_data)
  # Combine every 3 samples by taking the average
  dataa=take_average(data)
  #missing values
  print("The number of missing values in 12 combintations of extracted features after taking average is:",dataa.isnull().sum().sum())
  print("Performing initial feature selection based on variance for our radiomics data")
  # Perform initial feature selection based on variance
  small_var, low_variety = initial_feature_selection_var(dataa)
  # Print the column indices with small variance
  print("Columns with small variance in the extracted features are:\n",small_var)
  print("The number of columns with small variance in the extracted features is:",len(small_var))
  # Print the column indices with low variance
  print("Columns with low variety in the extracted features are:\n",low_variety)
  print("The number of columns with low variety in the extracted features is:",len(low_variety))
  print("Performing initial feature selection based on correlation for our radiomics data")
  # Perform initial feature selection based on correlation
  high_corr = initial_feature_selection_corr(dataa)
  # Combine lists of features with low variety, small variance, and high correlations
  red_features = low_variety + small_var + high_corr
  print("The total number of redundant extracted features is:", len(red_features))
  # Get unique redundant features
  redun_features = [*set(red_features)]
  print("The number of unique redundant extracted features is:", len(redun_features))
  # drop some redundant features
  radiomics_data=dataa.drop(dataa.columns[redun_features], axis=1, inplace=False)
  #print(radiomics_data)
  #missing values
  print("The number of missing values in radiomics data after removing the redundant features is:",radiomics_data.isnull().sum().sum())
  #infinite values
  print("The number of infinite values in radiomics data after removing the redundant features is:",np.isinf(radiomics_data).values.sum())
  selected_clinical_features, labels_in_sample= process_clinical_features_extract_labels(clinical_file_path)
  print("Performing initial feature selection based on variance for the clinical features")
  # Perform initial feature selection based on variance
  small_var_, low_variety_ = initial_feature_selection_var(selected_clinical_features)
  # Print the column indices with small variance
  print("Columns with small variance in the clinical features are:\n",small_var_)
  print("The number of columns with small variance in the clinical features is:",len(small_var_))
  # Print the column indices with low variance
  print("Columns with low variety in the clinical features are:\n",low_variety_)
  print("The number of columns with low variety in the clinical features is:",len(low_variety_))
  print("Performing initial feature selection based on correlation for the clinical features")
  # Perform initial feature selection based on correlation
  high_corr_ = initial_feature_selection_corr(selected_clinical_features)
  high_corr_
  # Combine lists of features with low variety, small variance, and high correlations
  red_features_ = low_variety_ + small_var_ + high_corr_
  print("The total number of redundant clinical features is:", len(red_features_))
  # Get unique redundant features
  redun_features_ = [*set(red_features_)]
  print("The number of unique redundant clinical features is:", len(redun_features_))
  clinical_features_data = selected_clinical_features.drop(selected_clinical_features.columns[redun_features_], axis=1, inplace=False).reset_index(drop=True)
  #print(clinical_features_data)
  # Drop redundant features among the selected clinical features
  clinical_features_data=selected_clinical_features.drop(selected_clinical_features.columns[redun_features_], axis=1, inplace=False)
  clinical_features_data
  # Create a DataFrame from the radiomics_data, clinical_features_data, list_, and labels_in_sample
  df_1 = pd.DataFrame(data=radiomics_data)
  df_2=clinical_features_data.reset_index(drop=True, inplace=False)
  df_3 = pd.DataFrame(data=list_)
  df_4 = labels_in_sample.reset_index(drop=True, inplace=False)
  frames = [df_1, df_2, df_3,df_4]
  # Concatenate the DataFrames horizontally with inner join
  D = pd.concat(frames, axis=1, join='inner', ignore_index=True)
  # Display the resulting DataFrame D
  # Save the DataFrame D to a CSV file named radiomics_clinical_featues_data
  D.to_csv(bc_mri_path+'/extracted_features/radiomics_clinical_features_data.csv', header=True)
  print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
  print('- - - - - - - - - - - - - - - - DATA PREPROCESSING DONE  - - - - - - - - - - - - - - - -')
  print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
  D = pd.read_csv(bc_mri_path+'/extracted_features/radiomics_clinical_features_data.csv')
  #Remove the first column from the DataFrame
  D = D.iloc[:, 1:]
  #Display the updated DataFrame
  #print("The data consisting of the extracted radiomics features and added clinical features after initial feature selection and adding the patient numbers and labels is:")
  #display(D)
  #Distribution of y
  u, c = np.unique(D.iloc[:,-1], return_counts=True)
  print("Distribution of y is:\n",pd.Series(c, index=u))
  # Read the CSV file 'radiomics_clinical_features_data.csv' and store the data in the 'data' variable
  data = pd.read_csv(radiomics_clinical_path)
  # Select all rows and columns starting from index 1 (excluding the first column)
  data = data.iloc[:, 1:]
  #missing values
  print("missing values:", data.isnull().sum().sum())
  #inf values
  print("inf values: ", np.isinf(data).values.sum())
  #Number of features
  f_num=data.shape[1]-2
  print("The number of features is:",f_num)
  X_=data.iloc[:,0:f_num]
  X_
  Y_=data.iloc[:,-1]
  Y_
  #Generate a dictionary for hyperparameters. Please feel free to modify it based on the sample size and the available resources you have.
  rf_parameters = {'criterion': ['gini'],  # , 'entropy'],
                   'n_estimators': [5, 10, 20, 50, 70, 200, 500],
                   # 'max_depth':[5,7,9,15,20,30],
                   # 'min_samples_split':[2,3,4,5,6,7],
                   # 'min_samples_leaf':[1,2,3,5],
                   # 'min_weight_fraction_leaf':[0,0.50],
                   # 'bootstrap':[True,False]
                   }
  svm_parameters = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                   'C': [0.0001, 0.001, 0.01, 0.05, 0.25, 0.5, 1, 5, 10, 20, 30, 45, 55, 60, 80, 100],
                   # 'degree':[1,2],
                   # 'gamma':['scale','auto',0.001,0.005,0.01,0.03,0.10,0.30,0.50,0.60,0.75,1]
                   }
  print("==============================================")
  print("||      BINARY CLASSIFICATION USING          ||")
  print("||           OUR RADIOMICS DATA              ||")
  print("==============================================")
  # Call the function
  one_vs_the_rest_classification(data, subtype='TN', classifier='rf', hyperparameters=rf_parameters)
  # one_vs_the_rest_classification(data, subtype=1, classifier='rf', hyperparameters=rf_parameters)
  #Call the function (Ensure you have enough samples before calling the function to avoid potential errors.)
  #one_vs_one_classification(data, subtype_1=0,subtype_2=2, classifier='rf', hyperparameters=rf_parameters)
  print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
  print('- - - - - - - - - - - - - - - - BINARY CLASSIFICATION DONE  - - - - - - - - - - - - - - - -')
  print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
  d=pd.read_csv(features_by_saha)
  d=d.iloc[:,1:]
  #display(d)
  #missing values
  print("The number of missing values in the extracted features by Saha et al. is:",d.isnull().sum().sum())
  # inf values
  print("The number of infinite values in the extracted features by Saha et al. is:",np.isinf(d).values.sum())
  #total number of features
  f_total= d.shape[1]
  print("The total number of features in the extracted features by Saha et al. is:",f_total)
  X=d.iloc[:,0:f_total]
  #print(X)
  print("Performing initial feature selection based on variance for Saha's radiomics data")
  # Perform initial feature selection based on variance
  small_var, low_variety = initial_feature_selection_var(X)
  # Print the column indices with small variance
  print("Columns with small variance in the extracted features by Saha et al. are:\n",small_var)
  print("The number of columns with small variance in the extracted features by Saha et al. is:",len(small_var))
  # Print the column indices with low variance
  print("Columns with low variety in the extracted features by Saha et al. are:\n",low_variety)
  print("The number of columns with low variety in the extracted features by Saha et al. is:",len(low_variety))
  print("Performing initial feature selection based on correlation for Saha's radiomics data")
  high_corr = initial_feature_selection_corr(X)
  # Combine lists of features with low variety, small variance, and high correlations
  red_features = low_variety + small_var + high_corr
  print("The total number of redundant features extracted by Saha et al. is:", len(red_features))
  # Get unique redundant features
  redun_features = [*set(red_features)]
  print("The number of unique redundant features extracted by Saha et al. is:", len(redun_features))
  # drop some redundant features
  radiomics_data=X.drop(X.columns[redun_features], axis=1, inplace=False)
  radiomics_data
  #missing values
  print("Then number of missing values in the extracted features by Saha et al. after initial fs is: ",radiomics_data.isnull().sum().sum())
  #Using clinical_features file to extract the labels
  clinical_features=pd.read_csv(clinical_file_path)
  label=clinical_features.iloc[2:,26]
  label=label.astype(int)
  label=label.reset_index(drop=True)
  #print("label:",label)
  print("data shape:",radiomics_data.shape)
  print("label shape:",label.shape)
  patients_number=list(range(1, 923))
  patients_number=pd.DataFrame(patients_number)
  patients_number
  df_1=pd.DataFrame(data=radiomics_data)
  df_2=pd.DataFrame(data=patients_number)
  df_3=pd.DataFrame(data=label)
  frames=[df_1,df_2,df_3]
  data= pd.concat(frames, axis=1,join='inner',ignore_index=True)
  #display(data)
  #Number of features
  f_num=data.shape[1]-2
  print("The number of features extracted by Saha et al. after initial fs is:",f_num)
  X_=data.iloc[:,0:f_num]
  X_
  Y_=data.iloc[:,-1]
  Y_
  #Distribution of the data
  u_lab, c_lab = np.unique(Y_, return_counts=True)
  print("Distribution of y in the extracted by Saha et al. is:\n",pd.Series(c_lab, index=u_lab))
  #Generate a dictionary for hyperparameters. Please feel free to modify it based on the available resources you have.
  rf_parameters = {'criterion': ['gini'],  # , 'entropy'],
                   'n_estimators': [5, 10, 20, 50, 70, 200, 500],
                   # 'max_depth':[5,7,9,15,20,30],
                   # 'min_samples_split':[2,3,4,5,6,7],
                   # 'min_samples_leaf':[1,2,3,5],
                   # 'min_weight_fraction_leaf':[0,0.50],
                   # 'bootstrap':[True,False]
                   }
  svm_parameters = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                   'C': [0.0001, 0.001, 0.01, 0.05, 0.25, 0.5, 1, 5, 10, 20, 30, 45, 55, 60, 80, 100],
                   # 'degree':[1,2],
                   # 'gamma':['scale','auto',0.001,0.005,0.01,0.03,0.10,0.30,0.50,0.60,0.75,1]
                   }
  print("==============================================")
  print("||      BINARY CLASSIFICATION USING          ||")
  print("||           SAHA'S RADIOMICS DATA           ||")
  print("==============================================")
  # Call the function
  one_vs_the_rest_classification(data, subtype='TN', classifier='rf', hyperparameters=rf_parameters)
  one_vs_the_rest_classification(data, subtype=1, classifier='svm', hyperparameters=svm_parameters)
  #Call the function
  one_vs_one_classification(data, subtype_1=0,subtype_2=3, classifier='rf', hyperparameters=rf_parameters)
  one_vs_one_classification(data, subtype_1='Luminal B',subtype_2='HER2+', classifier='svm', hyperparameters=svm_parameters)
  print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
  print('- - - - - - - - - - - - -BINARY CLASSIFICATION USING RADIOMICS FEATURES BY SAHA DONE  - - - - - - - - - - - -')
  print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
if __name__ == '__main__':
    main()