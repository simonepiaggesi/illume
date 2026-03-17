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

import random
from itertools import combinations
from more_itertools import random_combination
from scipy.stats import spearmanr, pearsonr
from utils import mixed_distance
from expl_utils import rule_based_similarity_complete, inverse_transform_rule_complete

from scipy.spatial.distance import cdist
from utils import xgb_eval, lgbm_eval, catb_eval


#########################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', default='breast', type=str,
                  help='dataset')

parser.add_argument('--bb', default='xgb', type=str,
                  help='black-box (xgb, lgbm, catb)')

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

parser.add_argument('--seed', default=0, type=int,
                  help='seed')

#########################################################


def main():

    params = vars(parser.parse_args())
    
    dataset_name = params['dataset']
    black_box = params['bb']
    seed = params['seed']
    
    l_y = np.floor(params['lambda_y'])
    l_st = params['lambda_st']
    l_co = params['lambda_co']
    l_so = params['lambda_so']
    num_k = params['num_k']
    l_rec, l_kld = (0.0, 1.0)
    bb_name = 'noy' if l_y==0.0 else black_box

    ddict = data_utils.get_tabular_dataset(dataset_name, path='../../dataset/', random_state=seed)

    X_train, X_test, y_train, y_test = (ddict['X_train'].copy(), ddict['X_test'].copy(), ddict['y_train'].copy(), ddict['y_test'].copy())
    features_map = ddict['features_map']

    X_train0, X_test0 = (ddict['X_train_orig'].copy(), ddict['X_test_orig'].copy())

    idx_num_cat = [list(d.values()) for i,d in features_map.items()]
    numeric_idx = [idx for idx,f in enumerate(ddict['feature_names']) if f in ddict['numeric_columns']]
    categorical_idx = [idx for idx,f in enumerate(ddict['feature_names']) if f not in ddict['numeric_columns']]

    class_to_explain = 1
    if np.unique(ddict['class_values']).shape[0]>2:
        class_to_explain = ddict['df'][ddict['class_name']].value_counts().idxmax()
        
    if black_box=='xgb':
        if not os.path.isfile(f'../blackboxes/{dataset_name}_xgb.{seed}.p'):
            clf_xgb, acc_xgb = xgb_eval(X_train, y_train, X_test, y_test, f1_average='macro')
            pickle.dump(clf_xgb, open(f'../blackboxes/{dataset_name}_xgb.{seed}.p','wb'))
        clf_xgb = pickle.load(open(f'../blackboxes/{dataset_name}_xgb.{seed}.p','rb'))
        y_test_pred = clf_xgb.predict_proba(X_test)
        y_train_pred = clf_xgb.predict_proba(X_train)
        y_test_bb = np.argmax(y_test_pred, axis=1)
        y_train_bb = np.argmax(y_train_pred, axis=1)
        print('XGBOOST')
        print('test acc:', '%0.3f'%f1_score(y_test, y_test_bb, average='macro'))

    if black_box=='lgbm':
        if not os.path.isfile(f'../blackboxes/{dataset_name}_lgbm.{seed}.p'):
            clf_lgbm, acc_lgbm = lgbm_eval(X_train, y_train, X_test, y_test, f1_average='macro')
            pickle.dump(clf_lgbm, open(f'../blackboxes/{dataset_name}_lgbm.{seed}.p','wb'))
        clf_lgbm = pickle.load(open(f'../blackboxes/{dataset_name}_lgbm.{seed}.p','rb'))
        y_test_pred = clf_lgbm.predict_proba(X_test)
        y_train_pred = clf_lgbm.predict_proba(X_train)
        y_test_bb = np.argmax(y_test_pred, axis=1)
        y_train_bb = np.argmax(y_train_pred, axis=1)
        print('LightGBM')
        print('test acc:', '%0.3f'%f1_score(y_test, y_test_bb, average='macro'))

    if black_box=='catb':
        if not os.path.isfile(f'../blackboxes/{dataset_name}_catb.{seed}.p'):
            clf_catb, acc_catb = catb_eval(X_train, y_train, X_test, y_test, f1_average='macro')
            pickle.dump(clf_catb, open(f'../blackboxes/{dataset_name}_catb.{seed}.p','wb'))
        clf_catb = pickle.load(open(f'../blackboxes/{dataset_name}_catb.{seed}.p','rb'))
        y_test_pred = clf_catb.predict_proba(X_test)
        y_train_pred = clf_catb.predict_proba(X_train)
        y_test_bb = np.argmax(y_test_pred, axis=1)
        y_train_bb = np.argmax(y_train_pred, axis=1)
        print('CATBOOST')
        print('test acc:', '%0.3f'%f1_score(y_test, y_test_bb, average='macro'))

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

        dt_path = folder_path + f'{dataset_name}_{black_box}_ILL-DT_{bb_name}_{latent_dim}_surr.{seed}.pkl'
        pickle.dump(latent.dtd, open(dt_path, 'wb'))

        # Explanations for test set
        Ex_dict_test = latent.get_decision_rules(X_test, y_test_bb, class_to_explain, num_k=num_k)
        #Ex_dict_test = [inverse_transform_rule_complete(ex, ddict['scaler'], latent.idx_num) if not np.any(pd.isnull(ex)) else None for ex in Ex_dict_test]
        
        expl_dict[(latent_dim,)]['expl'] = Ex_dict_test

        Eidx_test = np.array([i for i,ex in enumerate(Ex_dict_test) if not np.any(pd.isnull(ex))])

        # Faithfulness
        random.seed(seed)
        if Eidx_test.shape[0]>300:
            pair_ijs = [random_combination(list(Eidx_test), 2) for _ in range(50000)]
        else:
            pair_ijs = list(combinations(list(Eidx_test), 2))
        expl_dict[(latent_dim,)]['pairs_idx'] = pair_ijs
 
        sim_Ex_test = np.array([rule_based_similarity_complete(Ex_dict_test[i], Ex_dict_test[j]) for i,j in pair_ijs])
        sim_Bb_test = np.array([cdist(y_test_pred[[i]], y_test_pred[[j]], metric='euclidean')[0][0] for i,j in pair_ijs])

        expl_dict[(latent_dim,)]['cplt_pairs'] = sim_Ex_test
        expl_dict[(latent_dim,)]['bb_pairs'] = sim_Bb_test

        expl_dict[(latent_dim,)]['faithfulness'] = np.maximum(0., -spearmanr(sim_Ex_test, sim_Bb_test)[0])

        print(f'{latent_dim} latent dims - Faithfulness: ', expl_dict[(latent_dim,)]['faithfulness'])

        # Robustness
        conds_test = np.array([[y_test_bb[i]==y_test_bb[j] for j in Eidx_test] for i in Eidx_test], dtype=bool)

        nn_idx_test = [mixed_distance(x.reshape(1,-1), X_test[Eidx_test], 
                       categorical_idx, numeric_idx, metric=('neuclidean', 'hamming'))[0][conds_test[i]].argsort()[:21]
                                if np.any(conds_test[i]) else None for i,x in enumerate(X_test[Eidx_test])]

        nn_idx_test = [Eidx_test[conds_test[i]][t] if np.any(conds_test[i]) else None for i,t in enumerate(nn_idx_test)]
        expl_dict[(latent_dim,)]['nns_idx'] = nn_idx_test

        nn_sim_Ex_test = [[rule_based_similarity_complete(Ex_dict_test[i], Ex_dict_test[j]) for j in nn_idx_test[i]]
                                                if not np.any(pd.isnull(nn_idx_test[i])) else None for i in Eidx_test]

        expl_dict[(latent_dim,)]['cplt_nns'] = nn_sim_Ex_test

        expl_dict[(latent_dim,)]['robustness'] = np.array([[np.maximum(0., sim)[:knn].min() for sim in nn_sim_Ex_test] for knn in range(1, 21)], dtype=np.float64).mean(axis=1).mean()

        print(f'{latent_dim} latent dims - Robustness: ', expl_dict[(latent_dim,)]['robustness'])
        print()
      
    pickle.dump(expl_dict, open(result_path, 'wb'))

       
if __name__ == '__main__':
    main()