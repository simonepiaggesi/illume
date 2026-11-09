from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets, svm
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid, ParameterSampler, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, spearmanr
from numpy.random import default_rng
from collections import Counter
from itertools import groupby
from sklearn.cluster import KMeans
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

import torch
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline


def catb_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
                param_grid = {'n_estimators':[50, 100, 200, 300],
                              'max_depth': [2, 4, 8, 10, 12],
                              'learning_rate':[0.01, 0.05, 0.1, 0.3],
                              'l2_leaf_reg': [2, 3, 5, 10]},
                n_iter=100, n_jobs=-1, random_state=42):

    param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))

    acc = []
    for params in param_list:

        clf = CatBoostClassifier(random_state=random_state, **params)
        clf.fit(Z_train, Y_train)
        Y_pred = clf.predict(Z_test)
        acc.append(f1_score(Y_test, Y_pred, average=f1_average))

    best_params = param_list[np.argmax(acc)]
    best_clf = CatBoostClassifier(random_state=random_state, **best_params)
    best_clf.fit(Z_train, Y_train)
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)


def lgbm_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
            param_grid = {'n_estimators': [50, 100, 200, 300],
                          'max_depth': [2, 4, 8, 10, 12],
                          'learning_rate': [0.01, 0.05, 0.1, 0.3],
                          'num_leaves': [4, 8, 16, 32, 64, 128]}, 
            n_iter=100, n_jobs=-1, random_state=42):

    param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))

    acc = []
    for params in param_list:

        clf = LGBMClassifier(verbosity=-1, random_state=random_state, n_jobs=n_jobs, **params)
        clf.fit(Z_train, Y_train)
        Y_pred = clf.predict(Z_test)
        acc.append(f1_score(Y_test, Y_pred, average=f1_average))

    best_params = param_list[np.argmax(acc)]
    best_clf = LGBMClassifier(verbosity=-1, random_state=random_state, n_jobs=n_jobs, **best_params)
    best_clf.fit(Z_train, Y_train)
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)

def xgb_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
            param_grid = {'n_estimators': [50, 100, 200, 300],
                          'max_depth': [2, 4, 8, 10, 12],
                          'learning_rate': [0.01, 0.05, 0.1, 0.3],
                          'min_child_weight': [1, 3, 5, 10]}, 
            n_iter=100, n_jobs=-1, random_state=42):

    param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))

    acc = []
    for params in param_list:

        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_jobs=n_jobs, **params)
        clf.fit(Z_train, Y_train)
        Y_pred = clf.predict(Z_test)
        acc.append(f1_score(Y_test, Y_pred, average=f1_average))

    best_params = param_list[np.argmax(acc)]
    best_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_jobs=n_jobs, **best_params)
    best_clf.fit(Z_train, Y_train)
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)

def rf_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
            param_grid = {'n_estimators': [50, 100, 200, 300],
                          'max_depth': [None, 2, 4, 6, 8, 10, 12],
                          'min_samples_split': [2, 0.002, 0.01, 0.1, 0.2],
                          'min_samples_leaf': [1, 0.001, 0.01, 0.1, 0.2]},
            n_iter=100, n_jobs=-1, random_state=42):

    param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))

    acc = []
    for params in param_list:

        clf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, **params)
        clf.fit(Z_train, Y_train)
        Y_pred = clf.predict(Z_test)
        acc.append(f1_score(Y_test, Y_pred, average=f1_average))

    best_params = param_list[np.argmax(acc)]
    best_clf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, **best_params)
    best_clf.fit(Z_train, Y_train)
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)
    

def compute_class_metrics(Z_train, Y_train, Z_val, Y_val, Z_test, Y_test, n_jobs=-1):

    clf_knn, val_knn_score = knn_eval(Z_train, Y_train, Z_val, Y_val, f1_average='macro', n_jobs=n_jobs)
    clf_dtree, val_dtree_score = tree_eval(Z_train, Y_train, Z_val, Y_val, f1_average='macro')
    clf_logreg, val_logreg_score = logistic_eval(Z_train, Y_train, Z_val, Y_val, f1_average='macro', n_jobs=n_jobs)

    clf_rf, val_rf_score = rf_eval(Z_train, Y_train, Z_val, Y_val, f1_average='macro', n_jobs=n_jobs)
    clf_xgb, val_xgb_score = xgb_eval(Z_train, Y_train, Z_val, Y_val, f1_average='macro', n_jobs=n_jobs)
    clf_lgbm, val_lgbm_score = lgbm_eval(Z_train, Y_train, Z_val, Y_val, f1_average='macro', n_jobs=n_jobs)

    return {'KNN':(clf_knn.predict_proba(Z_train), clf_knn.predict_proba(Z_test)),
            'DecTree':(clf_dtree.predict_proba(Z_train), clf_dtree.predict_proba(Z_test)),
            'LogReg':(clf_logreg.predict_proba(Z_train), clf_logreg.predict_proba(Z_test)),
            'RF':(clf_rf.predict_proba(Z_train), clf_rf.predict_proba(Z_test)),
            'XGB':(clf_xgb.predict_proba(Z_train), clf_xgb.predict_proba(Z_test)),
            'LGBM':(clf_lgbm.predict_proba(Z_train), clf_lgbm.predict_proba(Z_test)),
            }
