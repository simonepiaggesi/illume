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

import data_utils

import random
from itertools import combinations
from more_itertools import random_combination
from scipy.stats import spearmanr, pearsonr
from utils import mixed_distance
from expl_utils import rule_based_similarity_complete, inverse_transform_rule_complete, get_rule_explanation_all, inverse_transform_rule_complete, fix_cat_rule_complete

from scipy.spatial.distance import cdist
from utils import xgb_eval, lgbm_eval, catb_eval, tree_eval

from anchor.anchor_tabular import AnchorTabularExplainer
from lore_sa.lorem import LOREM


#########################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', default='breast', type=str,
                  help='dataset')

parser.add_argument('--bb', default='xgb', type=str,
                  help='black-box (xgb, lgbm, catb)')

parser.add_argument('--seed', default=0, type=int,
                  help='seed')

#########################################################


def main():

    params = vars(parser.parse_args())
    
    dataset_name = params['dataset']
    black_box = params['bb']
    seed = params['seed']

    ddict = data_utils.get_tabular_dataset(dataset_name, path='../dataset/', random_state=seed)

    X_train, X_test, y_train, y_test = (ddict['X_train'].copy(), ddict['X_test'].copy(), ddict['y_train'].copy(), ddict['y_test'].copy())
    features_map = ddict['features_map']

    X_train0, X_test0 = (ddict['X_train_orig'].copy(), ddict['X_test_orig'].copy())

    idx_num_cat = [list(d.values()) for i,d in features_map.items()]
    numeric_idx = [idx for idx,f in enumerate(ddict['feature_names']) if f in ddict['numeric_columns']]
    categorical_idx = [idx for idx,f in enumerate(ddict['feature_names']) if f not in ddict['numeric_columns']]
        
    if black_box=='xgb':
        clf = pickle.load(open(f'../blackboxes/{dataset_name}_xgb.{seed}.p','rb'))
        y_test_pred = clf.predict_proba(X_test)
        y_train_pred = clf.predict_proba(X_train)
        y_test_bb = np.argmax(y_test_pred, axis=1)
        y_train_bb = np.argmax(y_train_pred, axis=1)
        print('XGBOOST')
        print('test acc:', '%0.3f'%f1_score(y_test, y_test_bb, average='macro'))

    if black_box=='lgbm':
        clf = pickle.load(open(f'../blackboxes/{dataset_name}_lgbm.{seed}.p','rb'))
        y_test_pred = clf.predict_proba(X_test)
        y_train_pred = clf.predict_proba(X_train)
        y_test_bb = np.argmax(y_test_pred, axis=1)
        y_train_bb = np.argmax(y_train_pred, axis=1)
        print('LightGBM')
        print('test acc:', '%0.3f'%f1_score(y_test, y_test_bb, average='macro'))

    if black_box=='catb':
        clf = pickle.load(open(f'../blackboxes/{dataset_name}_catb.{seed}.p','rb'))
        y_test_pred = clf.predict_proba(X_test)
        y_train_pred = clf.predict_proba(X_train)
        y_test_bb = np.argmax(y_test_pred, axis=1)
        y_train_bb = np.argmax(y_train_pred, axis=1)
        print('CATBOOST')
        print('test acc:', '%0.3f'%f1_score(y_test, y_test_bb, average='macro'))


    clf_predict_fn = lambda xx: clf.predict(xx).ravel()
    clf_predict_proba_fn = lambda xx: clf.predict_proba(xx)

    class_to_explain = 1
    if np.unique(ddict['class_values']).shape[0]>2:    
        class_to_explain = ddict['df'][ddict['class_name']].value_counts().idxmax()
        y_train_bb = (y_train_bb==class_to_explain).astype(int).ravel()
        y_test_bb = (y_test_bb==class_to_explain).astype(int).ravel()

        clf_predict_fn = lambda xx: (clf.predict(xx)==class_to_explain).astype(int).ravel()
        clf_predict_proba_fn = lambda xx: np.concatenate((1.- clf.predict_proba(xx)[:, [class_to_explain]], 
                                               clf.predict_proba(xx)[:, [class_to_explain]]), axis=1)

    feature_names = ['x'+str(j) for j in range(X_train.shape[1])]
          
    for bb_name in ['lore', 'anchor', 'inp-dt']: 

        folder_path = f'../results/{bb_name}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        result_path = folder_path +f'{dataset_name}_{black_box}_{bb_name.upper()}_expl.{seed}.pkl'

        expl_dict ={}
        print(result_path)
        print()

        if bb_name=='lore':

            for nsamples in [300,1000]: 
                for nruns in [1,5,10]: 

                    expl_dict[(nsamples, nruns,)] = {}
                    
                    lore_explainer = LOREM(X_train, clf_predict_fn, clf_predict_proba_fn,
                                       feature_names, ddict['class_name'], [0,1], feature_names, features_map,
                                       K_transformed=X_train, neigh_type='rndgen', binary='binary_from_dts', ngen=10,
                                       continuous_fun_estimation=True, discretize=False, filter_crules=False, random_state=42)

                    Ex_dict_test = []
                    for idx, x in enumerate(X_test):
                        
                        lore_expl_dict = dict()
                        for f in feature_names:
                            lore_expl_dict[(f, '<=')] = np.inf
                            lore_expl_dict[(f, '>')] = -np.inf

                        lore_exp = lore_explainer.explain_instance_stable(x, runs=nruns, samples=nsamples)
                        for c in lore_exp.rule.premises:
                            fid = feature_names.index(c.att)
                        for p in lore_exp.rule.premises:
                            lore_expl_dict[(p.att, p.op)] = p.thr
                        Ex_dict_test.append(lore_expl_dict)

                    expl_dict[(nsamples, nruns,)]['expl'] = Ex_dict_test

        if bb_name=='anchor':

            for nsamples in [100,300]: 
                for bsize in [4,10]: 
                    for delta in [0.1, 0.05]:
                        for tau in [0.15, 0.05]: 

                            expl_dict[(nsamples, bsize, delta, tau,)] = {}
                            anchor_explainer = AnchorTabularExplainer(feature_names=feature_names, class_names=[0,1],
                                                          categorical_names={})
                            anchor_explainer.fit(X_train, y_train_bb, X_test, y_test_bb)

                            Ex_dict_test = []
                            for idx, x in enumerate(X_test[:1]):
                                anchor_exp, anchor_exp_dict = anchor_explainer.explain_instance(x, clf_predict_fn, 
                                                            threshold=0.95, delta=delta, tau=tau, 
                                                            batch_size=nsamples, beam_size=bsize)
                                anchor_expl_dict = dict()
                                for f in feature_names:
                                    anchor_expl_dict[(f, '<=')] = np.inf
                                    anchor_expl_dict[(f, '>')] = -np.inf
                                for k,v in anchor_exp_dict.items():
                                    anchor_expl_dict[(feature_names[k[0]], k[1])] = v

                                Ex_dict_test.append(anchor_expl_dict)

                            expl_dict[(nsamples, bsize, delta, tau,)]['expl'] = Ex_dict_test

        if bb_name=='inp-dt':

            expl_dict[()] = {}
                
            dt, f1 = tree_eval(X_train, y_train_bb, X_test, y_test_bb, f1_average='macro')

            dtd = {'X':X_train,
                   'Y':y_train_bb,
                   'GT':y_train,
                   'dt': dt,
                   'feature_names': feature_names,
                   'class_name': ddict['class_name'],
                   'class_values': [0,1],
                   'numeric_columns': feature_names,
                   'X_test':X_test,
                   'Y_test':y_test_bb,
                   'GT_test':y_test
                }   

            idx_train = np.arange(X_train.shape[0])
            cond_train = dt.predict(X_train)==y_train_bb
            conds_train = np.array([np.logical_and(dt.predict(x.reshape(1,-1))==y_train_bb, cond_train) for x in X_test], dtype=bool)

            idx_from_train = [mixed_distance(x.reshape(1,-1), X_train, categorical_idx, numeric_idx, metric=('neuclidean', 'hamming'))[0][conds_train[i]].argsort()[0] 
                                if np.any(conds_train[i]) else None for i,x in enumerate(X_test)]
            idx_from_train = [idx_train[conds_train[i]][t] if np.any(conds_train[i]) else None for i,t in enumerate(idx_from_train)]

            _, Ex_dict_train = get_rule_explanation_all(X_train, dtd, n_features=X_train.shape[1], get_values=False)
            _, Ex_dict_test = get_rule_explanation_all(X_test, dtd, n_features=X_train.shape[1], get_values=False)   

            Ex_dict_test = [ex if dt.predict(X_test[[i]])==y_test_bb[i] 
                            else (Ex_dict_train[idx_from_train[i]] if np.any(conds_train[i]) else None) for i,ex in enumerate(Ex_dict_test)]
            
            expl_dict[()]['expl'] = Ex_dict_test

        for key in expl_dict:

            # Explanations for test set
            Ex_dict_test = expl_dict[key]['expl'] 

            Ex_dict_test = [fix_cat_rule_complete(ex, numeric_idx) if not np.any(pd.isnull(ex)) else None for ex in Ex_dict_test]
            #Ex_dict_test = [inverse_transform_rule_complete(ex, ddict['scaler'], numeric_idx) if not np.any(pd.isnull(ex)) else None for ex in Ex_dict_test]

            expl_dict[key]['expl'] = Ex_dict_test

            Eidx_test = np.array([i for i,ex in enumerate(Ex_dict_test) if not np.any(pd.isnull(ex))])

            # Faithfulness
            random.seed(seed)
            if Eidx_test.shape[0]>300:
                pair_ijs = [random_combination(list(Eidx_test), 2) for _ in range(50000)]
            else:
                pair_ijs = list(combinations(list(Eidx_test), 2))
            expl_dict[key]['pairs_idx'] = pair_ijs
     
            sim_Ex_test = np.array([rule_based_similarity_complete(Ex_dict_test[i], Ex_dict_test[j]) for i,j in pair_ijs])
            sim_Bb_test = np.array([cdist(y_test_pred[[i]], y_test_pred[[j]], metric='euclidean')[0][0] for i,j in pair_ijs])

            expl_dict[key]['cplt_pairs'] = sim_Ex_test
            expl_dict[key]['bb_pairs'] = sim_Bb_test

            expl_dict[key]['faithfulness'] = np.maximum(0., -spearmanr(sim_Ex_test, sim_Bb_test)[0])

            print(f'{bb_name} - {key} - Faithfulness: ', expl_dict[key]['faithfulness'])

            # Robustness
            conds_test = np.array([[y_test_bb[i]==y_test_bb[j] for j in Eidx_test] for i in Eidx_test], dtype=bool)

            nn_idx_test = [mixed_distance(x.reshape(1,-1), X_test[Eidx_test], 
                           categorical_idx, numeric_idx, metric=('neuclidean', 'hamming'))[0][conds_test[i]].argsort()[:21]
                                    if np.any(conds_test[i]) else None for i,x in enumerate(X_test[Eidx_test])]

            nn_idx_test = [Eidx_test[conds_test[i]][t] if np.any(conds_test[i]) else None for i,t in enumerate(nn_idx_test)]
            expl_dict[key]['nns_idx'] = nn_idx_test

            nn_sim_Ex_test = [[rule_based_similarity_complete(Ex_dict_test[i], Ex_dict_test[j]) for j in nn_idx_test[i]]
                                                    if not np.any(pd.isnull(nn_idx_test[i])) else None for i in Eidx_test]
            expl_dict[key]['cplt_nns'] = nn_sim_Ex_test

            expl_dict[key]['robustness'] = np.array([[np.maximum(0., sim)[:knn].min() for sim in nn_sim_Ex_test] for knn in range(2, 21)], dtype=np.float64).mean(axis=1).mean()

            print(f'{bb_name} - {key} - Robustness: ', expl_dict[key]['robustness'])
            print()
      
        pickle.dump(expl_dict, open(result_path, 'wb'))

       
if __name__ == '__main__':
    main()