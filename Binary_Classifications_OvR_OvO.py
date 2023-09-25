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
clinical_file_path = xlsx_csv_files_path + r'\Clinical_and_Other_Features.csv'
mapping_path = xlsx_csv_files_path + r'\Breast-Cancer-MRI-filepath_filename-mapping.csv'
boxes_path = xlsx_csv_files_path + r'\Annotation_Boxes.csv'
radiomics_clinical_path=bc_mri_path+r'\extracted_features\radiomics_clinical_features_data.csv'
features_by_saha=xlsx_csv_files_path + r'\Imaging_Features.csv'
types = ['pre', 'post_1', 'post_2', 'post_3']
def Binary_Classifications_OvR_OvO ():
  # Call the function to generate random samples for each cancer type
  global list0, list1, list2, list3
  list0, list1, list2, list3 = random_sample_for_each_cancer_type(clinical_file_path)
  # Combine the first elements from each list into a single list
  global list_
  list_ = list0 + list1 + list2 + list3
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

Binary_Classifications_OvR_OvO()