##########################################################################################
#####################################   IMPORTS    #######################################
##########################################################################################
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
#import tensorflow as tf
from Images_Functions import random_sample_for_each_cancer_type
from Data_Preprocessing_Functions import take_average,initial_feature_selection_var,find_zero_variance_features,initial_feature_selection_corr,process_clinical_features_extract_labels
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
################################   Radiomics_Features_Saha    ##################################
##########################################################################################
def Radiomics_Features_Saha():
  current_path,bc_mri_path,dataset_path,xlsx_csv_files_path,samples_path,clinical_file_path,mapping_path,boxes_path,radiomics_clinical_path,features_by_saha=path_provider()
  types = ['pre', 'post_1', 'post_2', 'post_3']
  # Call the function to generate random samples for each cancer type
  global list0, list1, list2, list3
  list0, list1, list2, list3 = random_sample_for_each_cancer_type(clinical_file_path)
  # Combine the first elements from each list into a single list
  global list_
  list_ = list0 + list1 + list2 + list3
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
  print("Performing initial feature selection based on variance for Saha's radiomics data.")
  # Perform initial feature selection based on variance
  small_var, low_variety = initial_feature_selection_var(X)
  # Print the column indices with small variance
  print("Columns with small variance in the extracted features by Saha et al. are:\n",small_var)
  print("The number of columns with small variance in the extracted features by Saha et al. is:",len(small_var))
  # Print the column indices with low variance
  print("Columns with low variety in the extracted features by Saha et al. are:\n",low_variety)
  print("The number of columns with low variety in the extracted features by Saha et al. is:",len(low_variety))
  print("Performing initial feature selection based on correlation for Saha's radiomics data.")
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
  Radiomics_Features_Saha()