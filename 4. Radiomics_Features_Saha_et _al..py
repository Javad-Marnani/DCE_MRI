import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import f1_score, make_scorer, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.impute import KNNImputer
from statistics import mode
from tensorflow.keras.utils import to_categorical
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from scipy.stats import t
from statistics import mode
from sklearn.decomposition import PCA
import glob
import cv2 as cv
from PIL import Image
from platform import python_version
from sklearn.model_selection import StratifiedKFold
import os
import random
import pydicom
from skimage.io import imsave, imread
import math
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, classification_report
import time

print(python_version())

current_path = os.getcwd()
current_path

'''Navigate to the directory where all the codes and data are stored'''
os.chdir('.../.../...')

'''Choose GPU 0 or 1 if they are available for processing.'''
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[1], True)
tf.config.set_visible_devices(physical_devices[1], 'GPU')
visible_devices = tf.config.get_visible_devices('GPU')
print(visible_devices)

d=pd.read_csv('DATA/Imaging_Features.csv') 
d=d.iloc[:,1:]
d

#missing values
print(d.isnull().sum().sum())
# inf values
print(np.isinf(d).values.sum())

#total number of features
f_total= d.shape[1]
f_total

X=d.iloc[:,0:f_total]
X

# Importing functions for initial feature selection
from 2. Data_Preprocessing import initial_feature_selection_var,find_zero_variance_features, initial_feature_selection_corr

# Perform initial feature selection based on variance
small_var, low_variety = initial_feature_selection_var(X,feature_stat=True)  
# Print the column indices with small variance
print(small_var)  
print(len(small_var))  
# Print the column indices with low variance
print(low_variety)  
print(len(low_variety))  

# Perform initial feature selection based on correlation
high_corr = initial_feature_selection_corr(X)  

# Combine lists of features with low variety, small variance, and high correlations
red_features = low_variety + small_var + high_corr
print("The total number of redundant features is:", len(red_features))

# Get unique redundant features
redun_features = [*set(red_features)]
print("The number of unique redundant features is:", len(redun_features))

# drop some redundant features
radiomics_data=X.drop(X.columns[redun_features], axis=1, inplace=False)
radiomics_data

#missing values
print(radiomics_data.isnull().sum().sum())
#infinit values
print(np.isinf(radiomics_data).values.sum())

#Using clinical_features file to extract the labels
clinical_features=pd.read_csv('DATA/Clinical_and_Other_Features.csv') 
clinical_features

label=clinical_features.iloc[2:,26]
label=label.astype(int)
label=label.reset_index(drop=True) 
label

print(radiomics_data.shape)
print(label.shape)

patients_number=list(range(1, 923))
patients_number=pd.DataFrame(patients_number)
patients_number

df_1=pd.DataFrame(data=radiomics_data)
df_2=pd.DataFrame(data=patients_number)
df_3=pd.DataFrame(data=label)
frames=[df_1,df_2,df_3]
data= pd.concat(frames, axis=1,join='inner',ignore_index=True)
display(data)

#Number of features
f_num=data.shape[1]-2
print("The number of features is:",f_num)

X_=data.iloc[:,0:f_num]
X_

Y_=data.iloc[:,-1]
Y_

#Distribution of the data
u_lab, c_lab = np.unique(Y_, return_counts=True)
pd.Series(c_lab, index=u_lab)

# Importing functions for classification
from 3. Binary_Classifications_OvR_OvO import confidence_interval, calculate_average_or_mode,convert_label_one_vs_the_rest
from 3. Binary_Classifications_OvR_OvO import convert_label_one_vs_one, evaluate_classifier

# One vs. the rest classifications using svm
classifier='svm'
hyperparameters={'kernel':['linear','rbf','poly','sigmoid'],'C':[0.0001,0.001,0.01,0.05,0.25,0.5,1,5,10,
                  20,30,45,55,60,80,100],'degree':[1,2],
               'gamma':['scale','auto',0.001,0.005,0.01,0.03,0.10,0.30,0.50,0.60,0.75,1]}
for i in tqdm(range(0,4)):
    if i==0:
        subtype='Luminal A'
    elif i==1:
        subtype='Luminal B'
    elif i==2:
        subtype='HER2+'
    else:
        subtype="TN"
    print(subtype, "vs. the rest classification using",classifier)   
    X,Y= convert_label_one_vs_the_rest(data, i)
    max_test_score,optimal_features,optimal_num_features,optimal_param =evaluate_classifier(X,Y,n_iter=500,
                        max_features=150,k_fold_cv=10,classifier=classifier, hyperparameters=hyperparameters)
    print('max_test_scores:\n',max_test_score)
    print('optimal_features:\n',optimal_features)
    print('optimal_num_features:\n',optimal_num_features)
    print('optimal_param:\n',optimal_param)
    #calling the functions
    avg_mode = calculate_average_or_mode(optimal_param)
    print("Average of numerical hyperparameters and mode of string hyperparameters across different runs:", avg_mode)
    print("C.I for the mean of max_test_scores:\n")
    confidence_interval(max_test_score)
    print("C.I for the mean of optimal_num_features:\n")
    confidence_interval(optimal_num_features)

# One vs. the rest classifications using rf
classifier='rf'
hyperparameters={'criterion':['gini'],#, 'entropy'],
                  'n_estimators':[5,10,20,50,70,200,500],
                  'max_depth':[5,7,9,15,20,30],
                  'min_samples_split':[2,3,4,5,6,7],
                  'min_samples_leaf':[1,2,3,5],
                  #'min_weight_fraction_leaf':[0,0.50],
                  #'bootstrap':[True,False]
                  }
for i in tqdm(range(0,4)):
    if i==0:
        subtype='Luminal A'
    elif i==1:
        subtype='Luminal B'
    elif i==2:
        subtype='HER2+'
    else:
        subtype="TN"
    print(subtype, "vs. the rest classification using",classifier)   
    X,Y= convert_label_one_vs_the_rest(data, i)
    max_test_score,optimal_features,optimal_num_features,optimal_param =evaluate_classifier(X,Y,n_iter=500,
                        max_features=150,k_fold_cv=10,classifier=classifier, hyperparameters=hyperparameters)
    print('max_test_scores:\n',max_test_score)
    print('optimal_features:\n',optimal_features)
    print('optimal_num_features:\n',optimal_num_features)
    print('optimal_param:\n',optimal_param)
    #calling the functions
    avg_mode = calculate_average_or_mode(optimal_param)
    print("Average of numerical hyperparameters and mode of string hyperparameters across different runs:", avg_mode)
    print("C.I for the mean of max_test_scores:\n")
    confidence_interval(max_test_score)
    print("C.I for the mean of optimal_num_features:\n")
    confidence_interval(optimal_num_features)

# One vs. one classifications using svm
classifier='svm'
hyperparameters={'kernel':['linear','rbf','poly','sigmoid'],'C':[0.0001,0.001,0.01,0.05,0.25,0.5,1,5,10,
                  20,30,45,55,60,80,100],'degree':[1,2],
               'gamma':['scale','auto',0.001,0.005,0.01,0.03,0.10,0.30,0.50,0.60,0.75,1]}
classification_cases = [('Luminal A', 'Luminal B'),
                        ('Luminal A', 'HER2+'),
                        ('Luminal A', 'TN'),
                        ('Luminal B', 'HER2+'),
                        ('Luminal B', 'TN'),
                        ('HER2+', 'TN')]
for subtype_1, subtype_2 in classification_cases:
    print(subtype_1, "vs.",subtype_2,"classification using",classifier)   
    X,Y= convert_label_one_vs_one(data,subtype_1,subtype_2 )
    max_test_score,optimal_features,optimal_num_features,optimal_param =evaluate_classifier(X,Y,n_iter=500,
                        max_features=150,k_fold_cv=10,classifier=classifier, hyperparameters=hyperparameters)
    print('max_test_scores:\n',max_test_score)
    print('optimal_features:\n',optimal_features)
    print('optimal_num_features:\n',optimal_num_features)
    print('optimal_param:\n',optimal_param)
    #calling the functions
    avg_mode = calculate_average_or_mode(optimal_param)
    print("Average of numerical hyperparameters and mode of string hyperparameters across different runs:", avg_mode)
    print("C.I for the mean of max_test_scores:\n")
    confidence_interval(max_test_score)
    print("C.I for the mean of optimal_num_features:\n")
    confidence_interval(optimal_num_features)

# One vs. one classifications using rf
classifier='rf'
hyperparameters={'criterion':['gini'],#, 'entropy'],
                  'n_estimators':[5,10,20,50,70,200,500],
                  'max_depth':[5,7,9,15,20,30],
                  'min_samples_split':[2,3,4,5,6,7],
                  'min_samples_leaf':[1,2,3,5],
                  #'min_weight_fraction_leaf':[0,0.50],
                  #'bootstrap':[True,False]
                  }
classification_cases = [('Luminal A', 'Luminal B'),
                        ('Luminal A', 'HER2+'),
                        ('Luminal A', 'TN'),
                        ('Luminal B', 'HER2+'),
                        ('Luminal B', 'TN'),
                        ('HER2+', 'TN')]
for subtype_1, subtype_2 in classification_cases:
    print(subtype_1, "vs.",subtype_2,"classification using",classifier)   
    X,Y= convert_label_one_vs_one(data,subtype_1,subtype_2 )
    max_test_score,optimal_features,optimal_num_features,optimal_param =evaluate_classifier(X,Y,n_iter=500,
                        max_features=150,k_fold_cv=10,classifier=classifier, hyperparameters=hyperparameters)
    print('max_test_scores:\n',max_test_score)
    print('optimal_features:\n',optimal_features)
    print('optimal_num_features:\n',optimal_num_features)
    print('optimal_param:\n',optimal_param)
    #calling the functions
    avg_mode = calculate_average_or_mode(optimal_param)
    print("Average of numerical hyperparameters and mode of string hyperparameters across different runs:", avg_mode)
    print("C.I for the mean of max_test_scores:\n")
    confidence_interval(max_test_score)
    print("C.I for the mean of optimal_num_features:\n")
    confidence_interval(optimal_num_features)