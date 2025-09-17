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
from expl_utils import rule_based_similarity_complete

from scipy.spatial.distance import cdist

from seneca.syege import generate_synthetic_rule_based_classifier, get_rule_explanation_complete, get_rule_explanation


#########################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--input_dim', default=16, type=int,
                  help='# features')

parser.add_argument('--lambda_y', default=1.0, type=float,
                  help='label conditioning')

parser.add_argument('--lambda_st', default=1.0, type=float,
                  help='optimize local stability of matrices')

parser.add_argument('--lambda_co', default=0.0, type=float,
                  help='optimize collinearity of latent features')

parser.add_argument('--lambda_so', default=1.0, type=float,
                  help='optimize soft-orthogonality of matrices')

parser.add_argument('--num_k', default=2, type=int,
                  help='non-zero matrix coefficients')

parser.add_argument('--seed', default=1, type=int,
                  help='seed')

#########################################################


def main():

    params = vars(parser.parse_args())

    input_dim = params['input_dim']
    m = min(16, input_dim)
    u = input_dim - m
    n = 2048
    dataset_name = f'seneca_tree_{m}+{u}'
    
    black_box = 'bb'
    seed = params['seed']
    
    l_y = np.floor(params['lambda_y'])
    l_st = params['lambda_st']
    l_co = params['lambda_co']
    l_so = params['lambda_so']
    num_k = params['num_k']
    l_rec, l_kld = (0.0, 1.0)
    bb_name = 'noy' if l_y==0.0 else black_box

    ddict = generate_synthetic_rule_based_classifier(n_features=m, n_all_features=m+u, n_samples=n, 
                                                            factor=10, sampling=0.1, explore_domain=False, 
                                                            random_state=seed)

    X = ddict['X']
    Y = ddict['Y']

    X_u = np.concatenate((X, np.random.default_rng(seed).uniform(np.min(X), np.max(X), size=(n, u))), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_u, Y, test_size=0.5, stratify=Y, random_state=seed)  
    idx_num_cat = [[i] for i,f in enumerate(ddict['feature_names'])]
    ddict['numeric_columns'] = ddict['feature_names'] 

    Ex_GT_dict_test = [get_rule_explanation_complete(x, ddict, n_features=m) for x in X_test]

    gtdt = ddict['dt']
    def predict_proba(X):
        return gtdt.predict_proba(X[:, :m])

    def predict(X):
        return gtdt.predict(X[:, :m])

    y_train_pred = predict_proba(X_train)
    y_test_pred = predict_proba(X_test)
    y_train_bb = predict(X_train)
    y_test_bb = predict(X_test)
    class_to_explain = 1

    folder_path = f'../results/k{num_k}_rec{l_rec}_kld{l_kld}_so{l_so}_co{l_co}_st{l_st}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    result_path = folder_path +f'{dataset_name}_{black_box}_ILL-DT_{bb_name}_expl.{seed}.pkl'
    # if os.path.isfile(result_path):
    #     continue
    # else:
    expl_dict ={}
    
    print(result_path)
    print()

    latent_dims = [2,4,8,16,32]

    for latent_dim in latent_dims:

        expl_dict[(latent_dim,)] = {}
            
        folder_path = f'../models/k{num_k}_rec{l_rec}_kld{l_kld}_so{l_so}_co{l_co}_st{l_st}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        latent = models.ILLUME(latent_dim=latent_dim, 
                             max_epochs=1000, early_stopping=30, learning_rate=0.001, batch_size=1024)

        X = (X_train, X_test)
        y = None if l_y==0.0 else (y_train_pred, y_test_pred) 

        model_path = folder_path + f'{dataset_name}_{black_box}_ILL_{latent_dim}.{seed}.pt'
        print(model_path)

        if os.path.isfile(model_path):
            # Latent Space Loading
            latent.load(X, y, idx_num_cat,model_path)
        else:
            # Latent Space Training
            if num_k == None:
                params={'num_k':num_k, 'l_rec':l_rec, 'l_kld':l_kld,\
                       'l_so':l_so, 'l_co':l_co, 'l_st':l_st}
                train_losses, test_losses = latent.fit(X, y, idx_num_cat, params, seed=seed)
                
                torch.save(latent.model.state_dict(), model_path)  
            else:
                params={'num_k':None, 'l_rec':l_rec, 'l_kld':l_kld,\
                       'l_so':l_so, 'l_co':l_co, 'l_st':l_st}
                train_losses, test_losses = latent.fit(X, y, idx_num_cat, params, seed=seed)

                params={'num_k':num_k, 'l_rec':l_rec, 'l_kld':l_kld,\
                       'l_so':l_so, 'l_co':l_co, 'l_st':l_st}
                train_losses, test_losses = latent.fit(X, y, idx_num_cat, params, seed=seed)
                
                torch.save(latent.model.state_dict(), model_path)  

        # Surrogate Training
        latent.explain_dectree(class_to_explain, num_k=num_k)

        lr_path = folder_path + f'{dataset_name}_{black_box}_ILL-LR_{bb_name}_{latent_dim}_surr.{seed}.pkl'
        pickle.dump(latent.dtd, open(lr_path, 'wb'))

        # Explanations for test set
        Ex_dict_test = latent.get_decision_rules(X_test, y_test_bb, class_to_explain, num_k=num_k)
        expl_dict[(latent_dim,)]['expl'] = Ex_dict_test

        Eidx_test = np.array([i for i,ex in enumerate(Ex_dict_test) if not np.any(pd.isnull(ex))])

        # Correctness
        sim_Ex_test = np.array([rule_based_similarity_complete(Ex_dict_test[i], Ex_GT_dict_test[i]) for i in Eidx_test])

        expl_dict[(latent_dim,)]['cplt_pairs'] = sim_Ex_test

        expl_dict[(latent_dim,)]['correctness'] = np.nanmean(np.maximum(0., sim_Ex_test))

        print(f'{latent_dim} latent dims - Correctness: ', expl_dict[(latent_dim,)]['correctness'])
        print()

        
    pickle.dump(expl_dict, open(result_path, 'wb'))

       
if __name__ == '__main__':
    main()