##########################################################################################
#####################################   IMPORTS    #######################################
##########################################################################################
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
#import tensorflow as tf
from Images_Functions import random_sample_for_each_cancer_type
from Classification_Functions import confidence_interval,calculate_average_or_mode,convert_label_one_vs_the_rest,convert_label_one_vs_one
from Classification_Functions import anova_feature_selection,evaluate_classifier,one_vs_the_rest_classification,one_vs_one_classification
from Path_Functions import path_provider,covert_xlsx_to_csv
##########################################################################################
####################################   SETTINGS    #######################################
##########################################################################################
_GPU = False

##########################################################################################
######################################   TEMP    #########################################
##########################################################################################
pd.set_option('display.max_columns', None)

##########################################################################################
#################################   GPU CONFIGURATION AND SELECTION    #####################################
##########################################################################################
'''Choose GPU 0 or 1 if they are available for processing.'''
if _GPU:
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[1], True)
  tf.config.set_visible_devices(physical_devices[1], 'GPU')
  visible_devices = tf.config.get_visible_devices('GPU')
  print(visible_devices)

##########################################################################################
#################################   Binary_Classifications_OvR_OvO    #####################################
##########################################################################################

def Binary_Classifications_OvR_OvO():
  current_path,bc_mri_path,dataset_path,xlsx_csv_files_path,samples_path,clinical_file_path,mapping_path,boxes_path,radiomics_clinical_path,features_by_saha=path_provider()
  types = ['pre', 'post_1', 'post_2', 'post_3']
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

if __name__ == '__main__':
  Binary_Classifications_OvR_OvO()