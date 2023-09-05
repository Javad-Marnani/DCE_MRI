import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import f1_score, make_scorer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.impute import KNNImputer
from tensorflow.keras.utils import to_categorical                            
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score,confusion_matrix, average_precision_score
import time
import random
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob
import cv2 as cv
from PIL import Image
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import pydicom
from platform import python_version
from tqdm import tqdm
from skimage.io import imsave
from skimage.io import imread

print(python_version())

current_path = os.getcwd()
current_path

'''Navigate to the directory where all the codes and data are stored'''
os.chdir('/users/fs2/jaghadavoodm/BC')

'''Choose GPU 0 or 1 if they are available for processing.'''
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[1], True)
tf.config.set_visible_devices(physical_devices[1], 'GPU')
visible_devices = tf.config.get_visible_devices('GPU')
print(visible_devices)

def random_sample_for_each_cancer_type(path, N0=50, N1=50, N2=50, N3=50,
                                      exclude0=None, exclude1=None, exclude2=None, exclude3=None,
                                      random_seed =42, show_patients=False):
    """
    Generate random samples for each cancer type using the labels obtained from clinical features.

    Args:
        path (str): The path to the clinical features file.
        N0 (int): The maximum number of samples to select for cancer type 0. Default is 50.
        N1 (int): The maximum number of samples to select for cancer type 1. Default is 50.
        N2 (int): The maximum number of samples to select for cancer type 2. Default is 50.
        N3 (int): The maximum number of samples to select for cancer type 3. Default is 50.
        exclude0 (list): List of patients to exclude for cancer type 0. Default is None.
        exclude1 (list): List of patients to exclude for cancer type 1. Default is None.
        exclude2 (list): List of patients to exclude for cancer type 2. Default is None.
        exclude3 (list): List of patients to exclude for cancer type 3. Default is None.
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
            exclude = exclude0
            N = min(N0, (label == 0).sum())
        elif i == 1:
            exclude = exclude1
            N = min(N1, (label == 1).sum())
        elif i == 2:
            exclude = exclude2
            N = min(N2, (label == 2).sum())
        else:
            exclude = exclude3
            N = min(N3, (label == 3).sum())
        
        mask_ = np.isin(patients, exclude) 
        filtered_patients = patients[~mask_]
        np.random.seed(random_seed)
        random_patients = np.random.choice(filtered_patients, size=N, replace=False)
        ls_ = (random_patients.tolist())
        ls_ = sorted(ls_)

        if i == 0:
            list0.append(ls_)
        elif i == 1:
            list1.append(ls_)
        elif i == 2:
            list2.append(ls_)
        else:
            list3.append(ls_)

    return list0, list1, list2, list3

# Set the path to the clinical features file
path = 'DATA/Clinical_and_Other_Features.csv'

# Call the function to generate random samples for each cancer type
# note that the random_seed should be the same as the one you already used
list0, list1, list2, list3 = random_sample_for_each_cancer_type(path)

# Combine the first elements from each list into a single list
list_ = list0[0] + list1[0] + list2[0] + list3[0]

# Print the length of the combined list
print("The total sample size is:", len(list_))

# Print the selected patients
list_

# Reading 12 datasets
#Pre
D1=pd.read_csv('Extracted Features/Pre_Original.csv') 
D1=D1.iloc[:,1:]
D2=pd.read_csv('Extracted Features/Pre_64.csv') 
D2=D2.iloc[:,1:]
D3=pd.read_csv('Extracted Features/Pre_32.csv') 
D3=D3.iloc[:,1:]
#Post1
D4=pd.read_csv('Extracted Features/Post1_Original.csv') 
D4=D4.iloc[:,1:]
D5=pd.read_csv('Extracted Features/Post1_64.csv') 
D5=D5.iloc[:,1:]
D6=pd.read_csv('Extracted Features/Post1_32.csv') 
D6=D6.iloc[:,1:]
#Post2
D7=pd.read_csv('Extracted Features/Post2_Original.csv') 
D7=D7.iloc[:,1:]
D8=pd.read_csv('Extracted Features/Post2_64.csv') 
D8=D8.iloc[:,1:]
D9=pd.read_csv('Extracted Features/Post2_32.csv') 
D9=D9.iloc[:,1:]
#Post3
D10=pd.read_csv('Extracted Features/Post3_Original.csv') 
D10=D10.iloc[:,1:]
D11=pd.read_csv('Extracted Features/Post3_64.csv') 
D11=D11.iloc[:,1:]
D12=pd.read_csv('Extracted Features/Post3_32.csv') 
D12=D12.iloc[:,1:]

# Merging 12 datasets
frames=[D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12]
data= pd.concat(frames, axis=1,join='inner',ignore_index=True)
display(data)

#missing values
data.isnull().sum().sum()

#infinit values
np.isinf(data).values.sum()

#columns including missing values
cols_with_nan_data=list(np.where(data.isna().any(axis=0))[0])
print(cols_with_nan_data)
#columns including inf values
cols_with_inf_data=list(data.columns[np.isinf(data).any()])
print(cols_with_inf_data)

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

# Combine every 3 samples by taking the average
dataa=take_average(data)
dataa

#missing values
dataa.isnull().sum().sum()

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
        min_val = des[3]
        mean = des[1]
        max_val = des[7]
        std_ = des[2]
        p1 = df.iloc[:, i].quantile(percentile)
        p2 = df.iloc[:, i].quantile(1 - percentile)
        q3 = des[6]

        if std_ < std:
            small_var.append(i)
        if p1 == p2:
            low_variety.append(i)
        if feature_stat:
            print(i, "min:", min_val, "mean:", mean, "max:", max_val, "std:", std)

    return small_var, low_variety

# Perform initial feature selection based on variance
small_var, low_variety = initial_feature_selection_var(dataa,feature_stat=True)  
# Print the column indices with small variance
print(small_var)  
print(len(small_var))  
# Print the column indices with low variance
print(low_variety)  
print(len(low_variety))  

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

# Perform initial feature selection based on correlation
high_corr = initial_feature_selection_corr(dataa)  

# Combine lists of features with low variety, small variance, and high correlations
red_features = low_variety + small_var + high_corr
print("The total number of redundant features is:", len(red_features))

# Get unique redundant features
redun_features = [*set(red_features)]
print("The number of unique redundant features is:", len(redun_features))

# drop some redundant features
radiomics_data=dataa.drop(dataa.columns[redun_features], axis=1, inplace=False)
radiomics_data

#missing values
radiomics_data.isnull().sum().sum()

#infinit values
np.isinf(radiomics_data).values.sum()

def process_clinical_features_extract_labels(path):
    """
    Process clinical features data and extract labels.

    Args:
        path (str): Path to the clinical features data file.

    Returns:
        tuple: A tuple containing the processed DataFrame (including the selected clinical features) and
        corresponding labels.

    """

    list0, list1, list2, list3 = random_sample_for_each_cancer_type(path)
    list_ = list0[0] + list1[0] + list2[0] + list3[0]
    patinets_indices_in_sample = [x - 1 for x in list_]
    clinical_features = pd.read_csv(path) 
    label = clinical_features.iloc[2:, 26]
    label = label.astype(int)
    label = label.reset_index(drop=True) 

    columns_to_drop_redundant = [0, 7, 9, 15, 16, 23, 24, 25, 26, 27, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                 58, 59, 61, 62, 63, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                                 87, 92, 93, 94, 95, 96, 97, 98]

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

selected_clinical_features, labels_in_sample= process_clinical_features_extract_labels('DATA/Clinical_and_Other_Features.csv')

# Perform initial feature selection based on variance
small_var_, low_variety_ = initial_feature_selection_var(selected_clinical_features)  
# Print the column indices with small variance
print(small_var_)  
print(len(small_var_))  
# Print the column indices with low variance
print(low_variety_)  
print(len(low_variety_)) 

# Perform initial feature selection based on correlation
high_corr_ = initial_feature_selection_corr(selected_clinical_features)  
high_corr_

# Combine lists of features with low variety, small variance, and high correlations
red_features_ = low_variety_ + small_var_ + high_corr_
print("The total number of redundant features is:", len(red_features_))

# Get unique redundant features
redun_features_ = [*set(red_features_)]
print("The number of unique redundant features is:", len(redun_features_))

clinical_features_data = selected_clinical_features.drop(selected_clinical_features.columns[redun_features_], axis=1, inplace=False).reset_index(drop=True)
clinical_features_data    

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
display(D)  
# Save the DataFrame D to a CSV file named radiomics_clinical_featues_data
D.to_csv('Extracted Features/radiomics_clinical_features_data.csv', header=True)  

u, c = np.unique(D.iloc[:,-1], return_counts=True)
pd.Series(c, index=u)