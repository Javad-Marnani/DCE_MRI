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
from Images_Functions import random_sample_for_each_cancer_type
from Data_Preprocessing_Functions import take_average,initial_feature_selection_var,find_zero_variance_features,initial_feature_selection_corr,process_clinical_features_extract_labels
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
clinical_file_path = xlsx_csv_files_path + r'\Clinical_and_Other_Features.csv'
mapping_path = xlsx_csv_files_path + r'\Breast-Cancer-MRI-filepath_filename-mapping.csv'
boxes_path = xlsx_csv_files_path + r'\Annotation_Boxes.csv'
radiomics_clinical_path=bc_mri_path+r'\extracted_features\radiomics_clinical_features_data.csv'
features_by_saha=xlsx_csv_files_path + r'\Imaging_Features.csv'
types = ['pre', 'post_1', 'post_2', 'post_3']
def Data_Preprocessing():
  # Call the function to generate random samples for each cancer type
  global list0, list1, list2, list3
  list0, list1, list2, list3 = random_sample_for_each_cancer_type(clinical_file_path)
  # Combine the first elements from each list into a single list
  global list_
  list_ = list0 + list1 + list2 + list3
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

Data_Preprocessing()