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
from Path_Functions import path_provider,covert_xlsx_to_csv

current_path,bc_mri_path,dataset_path,xlsx_csv_files_path,samples_path,clinical_file_path,mapping_path,boxes_path,radiomics_clinical_path,features_by_saha=path_provider()
types = ['pre', 'post_1', 'post_2', 'post_3']

def confidence_interval(vec, percent=0.90):
    """
    Analyze a vector by calculating the mean, standard deviation, and constructing a confidence interval.

    Parameters:
        vec (array-like): The vector to be analyzed.
        percent (float): The percentage (between 0 and 100) to contract the confidence interval.

    Returns:
        None: Prints the mean, standard deviation, and confidence interval.

    """
    n = len(vec)
    mean = np.mean(vec)
    std = np.std(vec, ddof=1)
    #print(vec)
    print("Mean:", mean)
    print("Standard deviation:", std)
    t_critical = t.ppf((1-((1-percent)/2)), n - 1)
    margin_of_error = t_critical * std / np.sqrt(n)
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    print("{:.1f}% Confidence interval for the mean:".format(percent * 100), confidence_interval)

def calculate_average_or_mode(data):
    """
    Calculate the average for numerical values and the mode for non-numerical values in a list of dictionaries.

    Parameters:
        data (list): A list of dictionaries containing key-value pairs.

    Returns:
        dict: A dictionary with keys representing the keys in the input dictionaries and values representing the calculated average or mode.

    """
    # initialize dictionaries to store numerical and non-numerical elements
    numeric_values = {}
    non_numeric_values = {}

    # iterate over each dictionary in the list
    for d in data:
        # iterate over each key-value pair in the dictionary
        for key, value in d.items():
            # check if the value is numerical
            if isinstance(value, (int, float)):
                # if the key is not already in the dictionary, initialize it
                if key not in numeric_values:
                    numeric_values[key] = []
                # append the value to the list associated with the key
                numeric_values[key].append(value)
            else:
                # if the key is not already in the dictionary, initialize it
                if key not in non_numeric_values:
                    non_numeric_values[key] = []
                # append the value to the list associated with the key
                non_numeric_values[key].append(value)

    # calculate the average of numerical values and mode of non-numerical values
    result = {}
    for key, value in numeric_values.items():
        result[key] = sum(value) / len(value)
    for key, value in non_numeric_values.items():
        result[key] = mode(value)

    return result

def convert_label_one_vs_the_rest(data, subtype):
    """
    Convert the original labels into binary format (one vs. the rest) based on the specified subtype.

    Parameters:
         data (pandas.DataFrame): The input data including features, patient numbers, and the labels.
        subtype (str or int): The subtype to convert into binary format. Can be a string ('Luminal A', 'Luminal B', 'HER2+', 'TN')
                              or the corresponding integers (0, 1, 2, 3).

    Returns:
        tuple: Tuple containing the original feature matrix (X) and target vector (Y)
              for the specified one vs. the rest classification.

    """
    f_num=data.shape[1]-2
    X_=data.iloc[:,0:f_num]
    Y_=data.iloc[:,-1]
    n_ = len(Y_)
    binary_label = np.zeros(n_, dtype=int)

    if isinstance(subtype, str):
        subtype = ['Luminal A', 'Luminal B', 'HER2+', 'TN'].index(subtype)

    for i in range(n_):
        if Y_[i] == subtype:
            binary_label[i] = 1

    return X_,binary_label

def convert_label_one_vs_one(data, subtype_1, subtype_2):
    """
    Process the data based on the specified subtypes.

    Parameters:
        The input data including features, patient numbers, and the labels.
        subtype_1 (int or str): First subtype, specified as either an integer or a string.
        subtype_2 (int or str): Second subtype, specified as either an integer or a string.
        ('Luminal A', 'Luminal B', 'HER2+', 'TN') or the corresponding integers (0, 1, 2, 3).

    Returns:
        tuple: Tuple containing processed feature matrix (X) and target vector (Y)
              for the specified pair of subtypes.

    """
    subtype_values = {
        'Luminal A': 0,
        'Luminal B': 1,
        'HER2+': 2,
        'TN': 3
    }

    f_num = data.shape[1] - 2

    if isinstance(subtype_1, str):
        subtype_1 = subtype_values[subtype_1]
    if isinstance(subtype_2, str):
        subtype_2 = subtype_values[subtype_2]

    data_i = data[data.iloc[:, f_num + 1].isin([subtype_1, subtype_2])]
    data_i = data_i.reset_index(drop=True)
    X_i = data_i.iloc[:, 0:f_num]
    Y_i = data_i.iloc[:, -1]

    return X_i, Y_i

def anova_feature_selection(X_train,y_train,X_test,n_features):
    """
    Perform ANOVA feature selection on the given datasets.

    Args:
        X_train (pd.DataFrame): Training dataset features.
        y_train (pd.Series): Training dataset labels.
        X_test (pd.DataFrame): Test dataset features.
        n_features (int): Number of top features to select.

    Returns:
        tuple: A tuple containing the transformed training and test datasets.
            - X_train_anov (pd.DataFrame): Transformed training dataset with the selected features.
            - X_test_anov (pd.DataFrame): Transformed test dataset with the selected features.
    """
    best = SelectKBest(k=n_features)
    fit_train = best.fit(X_train, y_train)
    X_train_anov = best.transform(X_train)
    X_test_anov = best.transform(X_test)
    return X_train_anov, X_test_anov

def evaluate_classifier(X, y, k_fold_cv=10, random_search_cv=5, n_iter=200,
                        max_features=150, classifier='None',n_neighbors_impute=10,n_neighbors_LOF=10,
                        hyperparameters=None,random_state=42):
    """
    Evaluate a classifier's performance on given data.

    Args:
        X (array-like): The feature matrix.
        y (array-like): The target vector.
        k_fold_cv (int, optional): The number of cross-validation folds. Defaults to 10.
        random_search_cv (int, optional): The number of iterations for randomized search. Defaults to 5.
        n_iter (int, optional): The number of iterations for randomized search. Defaults to 500.
        max_features (int, optional): The maximum number of features to consider for feature selection. Defaults to 150.
        classifier (str, optional): The classifier type. Must be 'svm' or 'rf'. Defaults to 'None'.
        n_neighbors_impute (int, optional): The number of neighbors for KNN imputation. Defaults to 5.
        n_neighbors_LOF (int, optional): The number of neighbors for Local Outlier Factor. Defaults to 5.
        hyperparameters (dict, optional): Hyperparameters for the classifier. Defaults to None.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing the maximum test score, optimal features, optimal number of features,
               and optimal parameters.

    """
    max_test_score = []
    optimal_features = []
    optimal_num_features = []
    optimal_param = []

    # Perform k-fold cross-validation
    kf = KFold(n_splits=k_fold_cv, shuffle=True, random_state=random_state)
    for train_index, test_index in tqdm(kf.split(X)):
        start = time.time()
        test_score = []

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        imputer = KNNImputer(n_neighbors=n_neighbors_impute)
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_imputed = imputer.fit_transform(X_train_scaled)
        X_test_scaled = scaler.transform(X_test)
        X_test_imputed = imputer.transform(X_test_scaled)

        # Loop for ANOVA feature selection
        for i in tqdm(range(1, max_features+1)):
            # Call the anova feature selection function
            X_train_anov,X_test_anov=anova_feature_selection(X_train_imputed,y_train,X_test_imputed,n_features=i)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors_LOF, contamination='auto')
            y_pred_train = lof.fit_predict(X_train_anov)
            X_train_inliers = X_train_anov[y_pred_train == 1]
            y_train_inliers = y_train[y_pred_train == 1]

            print("Training data shape:", X_train_anov.shape)
            print("Cleaned training data shape:", X_train_inliers.shape)

            if classifier == 'svm':
                from sklearn import svm
                classifier_obj = svm.SVC()
            elif classifier == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                classifier_obj = RandomForestClassifier()
            f1 = make_scorer(f1_score, average='macro')
            Randomsearch = RandomizedSearchCV(classifier_obj, hyperparameters, cv=random_search_cv, n_iter=n_iter,
                                              scoring=f1,verbose=1, n_jobs=-1, random_state=42)
            Randomsearch.fit(X_train_inliers, y_train_inliers)
            test_score.append(Randomsearch.score(X_test_anov, y_test))
            del Randomsearch

        end = time.time()
        print("The elapsed time is:", end - start)
        print("\nThe random state is:", random)
        print(np.sort(test_score))

        max_test_score.append(np.max(test_score))
        best = SelectKBest(k=np.argmax(test_score) + 1)
        fit_train = best.fit(X_train_imputed, y_train)
        optimal_features.append(fit_train.get_support(indices=True))
        X_train_anov = best.transform(X_train_imputed)
        X_test_anov = best.transform(X_test_imputed)

        lof = LocalOutlierFactor(n_neighbors=n_neighbors_LOF, contamination='auto')
        y_pred_train = lof.fit_predict(X_train_anov)
        X_train_inliers = X_train_anov[y_pred_train == 1]
        y_train_inliers = y_train[y_pred_train == 1]

        if classifier == 'svm':
            from sklearn import svm
            classifier_obj = svm.SVC()
        elif classifier == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            classifier_obj = RandomForestClassifier()

        Randomsearch = RandomizedSearchCV(classifier_obj,hyperparameters,cv=random_search_cv,n_iter=n_iter,
                                          scoring=f1,verbose=1,n_jobs=-1,random_state=42)
        Randomsearch.fit(X_train_inliers,y_train_inliers)
        print("\n The number of feature is:",np.argmax(test_score)+1)
        optimal_num_features.append(np.argmax(test_score)+1)
        optimal_estimator=Randomsearch.best_estimator_
        optimal_estimator.fit(X_train_inliers,y_train_inliers)
        test_prediction =optimal_estimator.predict(X_test_anov)
        print("The optimal parameters are:", Randomsearch.best_params_)
        optimal_param.append(Randomsearch.best_params_)
        print("test score is:\t", (Randomsearch.score(X_test_anov,y_test)))
        print("train score is:\t",(Randomsearch.score(X_train_inliers,y_train_inliers)))
        print(confusion_matrix(y_test, test_prediction))
        print("Report: \n",classification_report(y_test, test_prediction))
        train_prediction =optimal_estimator.predict(X_train_inliers)
        print(confusion_matrix(y_train_inliers, train_prediction))
        print("Report: \n",classification_report(y_train_inliers, train_prediction))
    return max_test_score, optimal_features, optimal_num_features, optimal_param


def one_vs_the_rest_classification(data, subtype, k_fold_cv=2, random_search_cv=2, n_iter=5,
                        max_features=5, classifier='None',n_neighbors_impute=1,n_neighbors_LOF=1,
                        hyperparameters=None,random_state=42):
    """
    Perform one-vs-the-rest classification using the provided functions.

    Args:
        data (pandas.DataFrame): The input data including features, patient numbers, and labels.
        subtype (str or int): The subtype to determine the case of one-vs-the-rest classification. Can be a string ('Luminal A', 'Luminal B', 'HER2+', 'TN') or the corresponding integers (0, 1, 2, 3).
        k_fold_cv (int, optional): The number of cross-validation folds. Defaults to 2.
        random_search_cv (int, optional): The number of iterations for randomized search. Defaults to 2.
        n_iter (int, optional): The number of iterations for randomized search. Defaults to 10.
        max_features (int, optional): The maximum number of features to consider for feature selection. Defaults to 5.
        classifier (str, optional): The classifier type. Must be 'svm' or 'rf'. Defaults to 'None'.
        n_neighbors_impute (int, optional): The number of neighbors for KNN imputation. Defaults to 1.
        n_neighbors_LOF (int, optional): The number of neighbors for Local Outlier Factor. Defaults to 1.
        hyperparameters (dict, optional): Hyperparameters for the classifier. Defaults to None.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        None. Prints information regarding the chosen criteria, optimal hyperparameters, and number of features.
    """
    X, Y = convert_label_one_vs_the_rest(data, subtype)
    max_test_score, optimal_features, optimal_num_features, optimal_param = evaluate_classifier(X, Y, k_fold_cv, random_search_cv, n_iter,
                        max_features, classifier,n_neighbors_impute,n_neighbors_LOF,
                        hyperparameters,random_state)

    print('Max test scores:\n', max_test_score)
    print('Optimal features:\n', optimal_features)
    print('Optimal number of features:\n', optimal_num_features)
    print('Optimal parameters:\n', optimal_param)

    avg_mode = calculate_average_or_mode(optimal_param)
    print("Average of numerical hyperparameters and mode of string hyperparameters across different runs:", avg_mode)

    print("Confidence Interval for the mean of max_test_scores:\n")
    confidence_interval(max_test_score)

    print("Confidence Interval for the mean of optimal_num_features:\n")
    confidence_interval(optimal_num_features)

def one_vs_one_classification(data, subtype_1, subtype_2, k_fold_cv=2, random_search_cv=2, n_iter=5,
                        max_features=5, classifier='None',n_neighbors_impute=1,n_neighbors_LOF=1,
                        hyperparameters=None,random_state=42):
    """
    Perform one-vs-one classification using the provided functions.

    Args:
        data (pandas.DataFrame): The input data including features, patient numbers, and labels.
        subtype_1 (int or str): First subtype, specified as either an integer or a string.
        subtype_2 (int or str): Second subtype, specified as either an integer or a string.
        ('Luminal A', 'Luminal B', 'HER2+', 'TN') or the corresponding integers (0, 1, 2, 3).
        k_fold_cv (int, optional): The number of cross-validation folds. Defaults to 2.
        random_search_cv (int, optional): The number of iterations for randomized search. Defaults to 2.
        n_iter (int, optional): The number of iterations for randomized search. Defaults to 10.
        max_features (int, optional): The maximum number of features to consider for feature selection. Defaults to 5.
        classifier (str, optional): The classifier type. Must be 'svm' or 'rf'. Defaults to 'None'.
        n_neighbors_impute (int, optional): The number of neighbors for KNN imputation. Defaults to 1.
        n_neighbors_LOF (int, optional): The number of neighbors for Local Outlier Factor. Defaults to 1.
        hyperparameters (dict, optional): Hyperparameters for the classifier. Defaults to None.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        None. Prints information regarding the chosen criteria, optimal hyperparameters, and number of features.
    """
    X, Y = convert_label_one_vs_one(data,subtype_1,subtype_2 )
    max_test_score, optimal_features, optimal_num_features, optimal_param = evaluate_classifier(X, Y, k_fold_cv, random_search_cv, n_iter,
                        max_features, classifier,n_neighbors_impute,n_neighbors_LOF,
                        hyperparameters,random_state)

    print('Max test scores:\n', max_test_score)
    print('Optimal features:\n', optimal_features)
    print('Optimal number of features:\n', optimal_num_features)
    print('Optimal parameters:\n', optimal_param)

    avg_mode = calculate_average_or_mode(optimal_param)
    print("Average of numerical hyperparameters and mode of string hyperparameters across different runs:", avg_mode)

    print("Confidence Interval for the mean of max_test_scores:\n")
    confidence_interval(max_test_score)

    print("Confidence Interval for the mean of optimal_num_features:\n")
    confidence_interval(optimal_num_features)