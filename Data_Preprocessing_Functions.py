##########################################################################################
#####################################   IMPORTS    #######################################
##########################################################################################
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
from Images_Functions import random_sample_for_each_cancer_type
from Path_Functions import path_provider,covert_xlsx_to_csv

current_path,bc_mri_path,dataset_path,xlsx_csv_files_path,samples_path,clinical_file_path,mapping_path,boxes_path,radiomics_clinical_path,features_by_saha=path_provider()
types = ['pre', 'post_1', 'post_2', 'post_3']

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
