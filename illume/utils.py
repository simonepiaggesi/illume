from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, spearmanr
from collections import Counter
from itertools import groupby
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def catb_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
                param_list = {'n_estimators':[50, 100, 200, 300],
                              'max_depth': [2, 4, 8, 10, 12],
                              'learning_rate':[0.01, 0.05, 0.1, 0.3],
                              'l2_leaf_reg': [2, 3, 5, 10]},
                n_iter=200, n_cv=4, n_jobs=-1, random_state=42):

    clf = CatBoostClassifier(random_state=random_state, verbose=False)

    rs = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=n_iter, cv=n_cv, scoring='f1_'+f1_average,
                            random_state=random_state, n_jobs=n_cv, verbose=0)
    rs.fit(Z_train, Y_train)

    best_clf = rs.best_estimator_
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)

def lgbm_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
            param_list = {'n_estimators': [50, 100, 200, 300],
                          'max_depth': [2, 4, 8, 10, 12],
                          'learning_rate': [0.01, 0.05, 0.1, 0.3],
                          'num_leaves': [4, 8, 16, 32, 64, 128]}, 
            n_iter=200, n_cv=4, n_jobs=-1, random_state=42):

    clf = LGBMClassifier(verbosity=-1, random_state=random_state, n_jobs=n_jobs)

    rs = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=n_iter, cv=n_cv, scoring='f1_'+f1_average,
                            random_state=random_state, n_jobs=n_cv, verbose=0)
    rs.fit(Z_train, Y_train)

    best_clf = rs.best_estimator_
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)

def xgb_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
            param_list = {'n_estimators': [50, 100, 200, 300],
                          'max_depth': [2, 4, 8, 10, 12],
                          'learning_rate': [0.01, 0.05, 0.1, 0.3],
                          'min_child_weight': [1, 3, 5, 10]}, 
            n_iter=200, n_cv=4, n_jobs=-1, random_state=42):

    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_jobs=n_jobs)

    rs = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=n_iter, cv=n_cv, scoring='f1_'+f1_average,
                            random_state=random_state, n_jobs=n_cv, verbose=0)
    rs.fit(Z_train, Y_train)

    best_clf = rs.best_estimator_
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)    

def knn_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
                    param_list = {'n_neighbors':[1, 2, 3, 4, 5, 10, 15, 20]}, 
                    n_cv=4, n_jobs=-1):


    clf = KNeighborsClassifier(n_jobs=n_jobs)

    rs = GridSearchCV(clf, param_grid=param_list, cv=n_cv, scoring='f1_'+f1_average, n_jobs=n_cv, verbose=0)
    rs.fit(Z_train, Y_train)

    best_clf = rs.best_estimator_
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)

def tree_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
                    param_list = {'min_samples_split': [2, 0.002, 0.01, 0.05, 0.1, 0.2],
                                  'min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
                                  'max_depth': [None, 2, 4, 6, 8, 10, 12, 16]}, 
                    n_cv=4, n_jobs=-1):

    clf = DecisionTreeClassifier(random_state=42)

    rs = GridSearchCV(clf, param_grid=param_list, cv=n_cv, scoring='f1_'+f1_average, n_jobs=n_cv, verbose=0)
    rs.fit(Z_train, Y_train)

    best_clf = rs.best_estimator_
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)

def logistic_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
                    param_list = {'penalty' : ['l1', 'l2'],
                                  'C': [0.001, 0.01, 0.05, 0.1, 1., 10.],
                                  'max_iter' : [100, 1000, 2000, 5000]}, 
                    n_cv=4, n_jobs=-1):


    clf = LogisticRegression(solver='liblinear', random_state=42)

    rs = GridSearchCV(clf, param_grid=param_list, cv=n_cv, scoring='f1_'+f1_average, n_jobs=n_cv, verbose=0)
    rs.fit(Z_train, Y_train)

    best_clf = rs.best_estimator_
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)


def mixed_distance(XA, XB, idx_cat, idx_num, metric=('ncosine', 'hamming')):
    metric_continuous = metric[0]
    metric_categorical = metric[1]

    def neuclidean(x, y):
        return 0.5 * np.var(x - y) / (np.var(x) + np.var(y))
    if metric_continuous=='neuclidean':
        metric_continuous = lambda u,v: neuclidean(u, v) 

    def ncosine(x, y):
        return 0.5 * (1.- np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y)))
    if metric_continuous=='ncosine':
        metric_continuous = lambda u,v: ncosine(u, v)        

    if len(idx_cat)>0:
        dist_categorical = cdist(XA[:, idx_cat], XB[:, idx_cat],
                             metric=metric_categorical)
        ratio_categorical = len(idx_cat) / (len(idx_cat)+len(idx_num))
        dist = ratio_categorical * dist_categorical

        if len(idx_num)>0:
            dist_continuous = cdist(XA[:, idx_num], XB[:, idx_num],
                                metric=metric_continuous)
            ratio_continuous = len(idx_num) / (len(idx_cat)+len(idx_num))
            dist += ratio_continuous * dist_continuous 
    else:
        dist = cdist(XA, XB, metric=metric_continuous)

    return dist
