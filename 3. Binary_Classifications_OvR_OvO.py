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

# Read the CSV file 'radiomics_clinical_features_data.csv' and store the data in the 'data' variable
data = pd.read_csv('Extracted Features/radiomics_clinical_features_data.csv')  
# Select all rows and columns starting from index 1 (excluding the first column)
data = data.iloc[:, 1:]  
data

#missing values
print(data.isnull().sum().sum())
#inf values
print(np.isinf(data).values.sum())

#Number of features
f_num=data.shape[1]-2
print("The number of features is:",f_num)

X_=data.iloc[:,0:f_num]
X_

Y_=data.iloc[:,-1]
Y_

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

def evaluate_classifier(X, y, k_fold_cv=10, random_search_cv=5, n_iter=500,
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
            f1 = make_scorer(f1_score, average='macro')
            best = SelectKBest(k=i)
            fit_train = best.fit(X_train_imputed, y_train)
            X_train_anov = best.transform(X_train_imputed)
            X_test_anov = best.transform(X_test_imputed)
            
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