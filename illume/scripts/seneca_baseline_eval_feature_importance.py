import pandas as pd
import pickle
import numpy as np
import os
import argparse
import sys
sys.path.append("../")

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch

import data_utils
import models

from scipy.stats import spearmanr, pearsonr
from utils import mixed_distance
from expl_utils import feature_importance_similarity

import shap
from lime.lime_tabular import LimeTabularExplainer
from utils import logistic_eval

from scipy.spatial.distance import cdist

from seneca.syege import generate_synthetic_linear_classifier, get_feature_importance_explanation 

from sympy import diff, re, simplify
from seneca.symexpr import generate_expression, gen_classification_symbolic, eval_multinomial



#########################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--input_dim', default=16, type=int,
                  help='# features')

parser.add_argument('--seed', default=1, type=int,
                  help='seed')

#########################################################


def main():

    params = vars(parser.parse_args())

    input_dim = params['input_dim']
    m = min(16, input_dim)
    u = input_dim - m
    n = 2048
    dataset_name = f'seneca_linear_{m}+{u}'
    
    black_box = 'bb'
    seed = params['seed']
    
    ddict = generate_synthetic_linear_classifier(n_features=m, n_all_features=m+u, n_samples=n, random_state=seed)

    X = ddict['X']
    Y = ddict['Y']

    X_u = np.concatenate((X, np.random.default_rng(seed).uniform(np.min(X), np.max(X), size=(n, u))), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_u, Y, test_size=0.5, stratify=Y, random_state=seed)  
    idx_num_cat = [[i] for i,f in enumerate(ddict['feature_names'])]

    Ex_GT_test = np.array([get_feature_importance_explanation(x, ddict, n_features=m, get_values=True) 
                                     for x in X_test])
    expr = ddict['expr']
    evals = ddict['Y1']
    evals_binary = ddict['Y']
    evals0 = evals[evals_binary == 0]
    evals1 = evals[evals_binary == 1]

    mm0 = MinMaxScaler(feature_range=(0, 0.5))
    mm0.fit(evals0.reshape(-1, 1))
    mm1 = MinMaxScaler(feature_range=(0.5, 1.0))
    mm1.fit(evals1.reshape(-1, 1))

    feature_names = ddict['feature_names']
    class_name = ddict['class_name']
    class_values = ddict['class_values']
    numeric_columns = ddict['feature_names']

    def predict_proba(X):
        X = X[:, :m]
        evals = list()
        for x in X:
            evals.append(re(eval_multinomial(expr, vals=list(x))))
        evals = np.array(evals)
        evals = np.array(evals, dtype=float)
        evals_binary = evals > 0
        evals_binary = evals_binary.flatten()
        evals_binary = np.array(evals_binary, dtype=int)

        evals_scaled = list()
        for x, y in zip(evals, evals_binary):
            if np.isinf(x):
                val = 1.0 if x == +np.inf else 0.0
            elif np.isnan(x):
                val = 0.0
            else:
                if y == 0:
                    val = mm0.transform(x.reshape(-1, 1))[0][0]
                else:
                    val = mm1.transform(x.reshape(-1, 1))[0][0]
                val = max(0.0, min(val, 1.0))
            evals_scaled.append([1.0 - val, val])

        evals_scaled = np.array(evals_scaled)
        return evals_scaled

    def predict(X):
        proba = predict_proba(X)
        return np.argmax(proba, axis=1)

    y_train_pred = predict_proba(X_train)
    y_test_pred = predict_proba(X_test)
    y_train_bb = predict(X_train)
    y_test_bb = predict(X_test)
    class_to_explain = 1

    feature_names = ['x'+str(j) for j in range(X_train.shape[1])]
          
    for bb_name in ['lime', 'shap', 'inp-lr']: 

        folder_path = f'../results/{bb_name}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        result_path = folder_path +f'{dataset_name}_{black_box}_{bb_name.upper()}_expl.{seed}.pkl'

        expl_dict ={}
        print(result_path)
        print()

        if bb_name=='lime':

            for nsamples in [100,300,1000,5000]:

                expl_dict[(nsamples,)] = {}
                
                lime_explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=[0,1],
                                          discretize_continuous=False, discretizer='entropy', random_state=42)

                Ex_test = []
                for idx, x in enumerate(X_test):
                    lime_exp = lime_explainer.explain_instance(x, predict_proba, num_features=X_train.shape[1], 
                                                                num_samples=nsamples)
                    lime_exp_as_dict = {e[0]: e[1] for e in lime_exp.as_list()}
                    lime_expl_val = np.array([lime_exp_as_dict.get(f, 0.0) for f in feature_names])
                    Ex_test.append(lime_expl_val)

                expl_dict[(nsamples,)]['expl'] = Ex_test

        if bb_name=='shap':

            expl_dict[()] = {}

            reference = X_train.mean(axis=0)
            shap_explainer = shap.KernelExplainer(predict_proba, np.reshape(reference, (1, len(reference))), 
                                    feature_names=feature_names, random_state=42)
            Ex_test = []
            for idx, x in enumerate(X_test):
                shap_expl_val = shap_explainer.shap_values(x, l1_reg='num_features(%s)'%(m+u))[class_to_explain]
                Ex_test.append(shap_expl_val)
        
            expl_dict[()]['expl'] = Ex_test

        if bb_name=='inp-lr':

            expl_dict[()] = {}
                
            lg, f1 = logistic_eval(X_train, y_train_bb, X_test, y_test_bb, f1_average='macro')

            idx_train = np.arange(X_train.shape[0])
            cond_train = lg.predict(X_train)==y_train_bb
            conds_train = np.array([np.logical_and(lg.predict(x.reshape(1,-1))==y_train_bb, cond_train) for x in X_test], dtype=bool)

            idx_from_train = [mixed_distance(x.reshape(1,-1), X_train, [], list(range(m+u)), metric=('neuclidean', 'hamming'))[0][conds_train[i]].argsort()[0] 
                                if np.any(conds_train[i]) else None for i,x in enumerate(X_test)]
            idx_from_train = [idx_train[conds_train[i]][t] if np.any(conds_train[i]) else None for i,t in enumerate(idx_from_train)]

            Ex_train = lg.coef_*X_train
            Ex_test = [ex if lg.predict(X_test[[i]])==y_test_bb[i] 
                                else (Ex_train[idx_from_train[i]] if np.any(conds_train[i]) else None) for i,ex in enumerate(lg.coef_*X_test)]

            expl_dict[()]['expl'] = Ex_test

        for key in expl_dict:

            # Explanations for test set
            Ex_test = expl_dict[key]['expl'] 

            Eidx_test = np.array([i for i,ex in enumerate(Ex_test) if not np.any(pd.isnull(ex))])

            # Correctness
            sim_Ex_test = np.array([feature_importance_similarity(Ex_test[i], Ex_GT_test[i]) for i in Eidx_test])

            expl_dict[key]['cosine_pairs'] = sim_Ex_test

            expl_dict[key]['correctness'] = np.nanmean(np.maximum(0., sim_Ex_test))

            print(f'{bb_name} - {key} - Correctness: ', expl_dict[key]['correctness'])
            print()
                
        pickle.dump(expl_dict, open(result_path, 'wb'))

       
if __name__ == '__main__':
    main()