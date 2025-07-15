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
    

def knn_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
                    param_grid = {'n_neighbors':[1, 2, 3, 4, 5, 10, 15, 20, 30]}, 
                    n_jobs=-1):

    param_list = list(ParameterGrid(param_grid))

    acc = []
    for params in param_list:

        clf = KNeighborsClassifier(n_jobs=n_jobs, **params)
        clf.fit(Z_train, Y_train)
        Y_pred = clf.predict(Z_test)
        acc.append(f1_score(Y_test, Y_pred, average=f1_average))

    best_params = param_list[np.argmax(acc)]
    best_clf = KNeighborsClassifier(n_jobs=n_jobs, **best_params)
    best_clf.fit(Z_train, Y_train)
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)

def tree_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
                    param_grid = {'min_samples_split': [2, 0.002, 0.01, 0.05, 0.1, 0.2],
                                  'min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
                                  'max_depth': [None, 2, 4, 6, 8, 10, 12, 16]}, 
                    n_jobs=-1):

    param_list = list(ParameterGrid(param_grid))

    acc = []
    for params in param_list:

        clf = DecisionTreeClassifier(random_state=42, **params)
        clf.fit(Z_train, Y_train)
        Y_pred = clf.predict(Z_test)
        acc.append(f1_score(Y_test, Y_pred, average=f1_average))

    best_params = param_list[np.argmax(acc)]
    best_clf = DecisionTreeClassifier(random_state=42, **best_params)
    best_clf.fit(Z_train, Y_train)
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)    


def logistic_eval(Z_train, Y_train, Z_test, Y_test, f1_average='macro',
                    param_grid = {'penalty' : ['l1', 'l2'],
                                  'C': [0.001, 0.01, 0.05, 0.1, 1., 10.],
                                  'max_iter' : [100, 1000, 2000, 5000]}, 
                    n_jobs=-1):

    param_list = list(ParameterGrid(param_grid))

    acc = []
    for params in param_list:

        clf = LogisticRegression(solver='liblinear',random_state=42, **params)
        clf.fit(Z_train, Y_train)
        Y_pred = clf.predict(Z_test)
        acc.append(f1_score(Y_test, Y_pred, average=f1_average))

    best_params = param_list[np.argmax(acc)]
    best_clf = LogisticRegression(solver='liblinear',random_state=42, **best_params)
    best_clf.fit(Z_train, Y_train)
    Y_pred = best_clf.predict(Z_test)

    return best_clf, f1_score(Y_test, Y_pred, average=f1_average)    


def random_triplet_eval(X, X_new):

    # Sampling Triplets
    # Five triplet per point
    anchors = np.arange(X.shape[0])
    rng = default_rng(42)
    triplets = rng.choice(anchors, (X.shape[0], 5, 2))
    triplet_labels = np.zeros((X.shape[0], 5))
    anchors = anchors.reshape((-1, 1, 1))

    # Calculate the distances and generate labels
    b = np.broadcast(anchors, triplets)
    distances = np.empty(b.shape)
    distances.flat = [np.linalg.norm(X[u] - X[v]) for (u,v) in b]
    labels = distances[:, :, 0] < distances[: , :, 1]

    # Calculate distances for LD
    b = np.broadcast(anchors, triplets)
    distances_l = np.empty(b.shape)
    distances_l.flat = [np.linalg.norm(X_new[u] - X_new[v]) for (u,v) in b]
    pred_vals = distances_l[:, :, 0] < distances_l[:, :, 1]
    correct = np.sum(pred_vals == labels)
    acc = correct/X.shape[0]/5
    return acc

def random_pairs_eval(X, X_new):

    anchors = np.arange(X.shape[0])
    rng = default_rng(42)
    pairs = rng.choice(anchors, (X.shape[0], 5, 1))
    anchors = anchors.reshape((-1, 1, 1))

    # Calculate the distances
    b = np.broadcast(anchors, pairs)
    distances = np.empty(b.shape)
    distances.flat = [np.linalg.norm(X[u] - X[v]) for (u,v) in b]

    # Calculate distances for LD
    b = np.broadcast(anchors, pairs)
    distances_l = np.empty(b.shape)
    distances_l.flat = [np.linalg.norm(X_new[u] - X_new[v]) for (u,v) in b]

    acc = spearmanr(distances.ravel(), distances_l.ravel())[0]
    return acc
 
def isf_eval(X, Z, n_jobs=-1):
    clf = IsolationForest(n_jobs=n_jobs)
    clf.fit(X)
    outlier_factor_input_space = clf.score_samples(X)
    clf = IsolationForest(n_jobs=n_jobs)
    clf.fit(Z)
    outlier_factor_latent_space = clf.score_samples(Z)
    #isf_score = np.mean((outlier_factor_input_space-outlier_factor_latent_space)**2)
    return spearmanr(outlier_factor_input_space, outlier_factor_latent_space)[0]

def kld_eval(X, Z):
    def compute_similarity(X, sigma=1):
        D = torch.cdist(X, X)
        M = torch.exp((-D**2)/(2*sigma**2))
        return M / (torch.ones([M.shape[0],M.shape[1]])*(torch.sum(M, axis = 0))).transpose(0,1)

    def kld_loss_function(X, Z, sigma=1):
        similarity_KLD = torch.nn.KLDivLoss(reduction='batchmean')
        Sx = compute_similarity(X, sigma)
        Sz = compute_similarity(Z, sigma)
        loss = similarity_KLD(torch.log(Sz), Sx)
        return loss

    n = np.arange(X.shape[0])

    if X.shape[0]>4096:
        rng = default_rng(42)
        n = rng.choice(n, size=4096, replace=False)

    return kld_loss_function(X[n], Z[n]).item()

def compute_latent_metrics(X_train, Z_train, X_test, Z_test, n_jobs=-1):

    train_triplet_score = random_triplet_eval(X_train, Z_train)
    test_triplet_score = random_triplet_eval(X_test, Z_test)

    train_spearman_score = np.maximum(random_pairs_eval(X_train, Z_train), 0.)
    test_spearman_score = np.maximum(random_pairs_eval(X_test, Z_test), 0.)
    
    train_isf_score = np.maximum(isf_eval(X_train, Z_train, n_jobs=n_jobs), 0.)
    test_isf_score = np.maximum(isf_eval(X_test, Z_test, n_jobs=n_jobs), 0.)

    train_kld_score = kld_eval(torch.tensor(X_train).float(), torch.tensor(Z_train).float())
    test_kld_score = kld_eval(torch.tensor(X_test).float(), torch.tensor(Z_test).float())

    train_dim_pearson = 1.-np.nan_to_num(np.abs(1.-pdist(Z_train.T, metric='correlation')))
    train_dim_pearson = 1.-np.nan_to_num(train_dim_pearson[train_dim_pearson>0].mean())
    test_dim_pearson = 1.-np.nan_to_num(np.abs(1.-pdist(Z_test.T, metric='correlation')))
    test_dim_pearson = 1.-np.nan_to_num(test_dim_pearson[test_dim_pearson>0].mean())

    return {'Triplet': (train_triplet_score, test_triplet_score),
            'Spearman': (train_spearman_score, test_spearman_score),
            'IsF': (train_isf_score, test_isf_score),
            'KLD': (train_kld_score, test_kld_score),
            'Pearson': (train_dim_pearson, test_dim_pearson),
            }

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
