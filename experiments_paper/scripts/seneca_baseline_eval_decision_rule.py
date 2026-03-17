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

from scipy.stats import spearmanr, pearsonr
from utils import mixed_distance
from expl_utils import rule_based_similarity_complete, get_rule_explanation_all

from anchor.anchor_tabular import AnchorTabularExplainer
from lore_sa.lorem import LOREM
from utils import mixed_distance, tree_eval

from scipy.spatial.distance import cdist

from seneca.syege import generate_synthetic_rule_based_classifier, get_rule_explanation_complete, get_rule_explanation




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
    dataset_name = f'seneca_tree_{m}+{u}'

    black_box = 'bb'
    seed = params['seed']

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

    feature_names = ['x'+str(j) for j in range(X_train.shape[1])]
    features_map = {i: {f:i} for i,f in enumerate(feature_names)}
          
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
                    
                    lore_explainer = LOREM(X_train, predict, predict_proba,
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
                                anchor_exp, anchor_exp_dict = anchor_explainer.explain_instance(x, predict, 
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

            idx_from_train = [mixed_distance(x.reshape(1,-1), X_train, [], list(range(m+u)), metric=('neuclidean', 'hamming'))[0][conds_train[i]].argsort()[0] 
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

            expl_dict[key]['expl'] = Ex_dict_test

            Eidx_test = np.array([i for i,ex in enumerate(Ex_dict_test) if not np.any(pd.isnull(ex))])

            # Correctness
            sim_Ex_test = np.array([rule_based_similarity_complete(Ex_dict_test[i], Ex_GT_dict_test[i]) for i in Eidx_test])

            expl_dict[key]['cplt_pairs'] = sim_Ex_test

            expl_dict[key]['correctness'] = np.nanmean(np.maximum(0., sim_Ex_test))

            print(f'{bb_name} - {key} - Correctness: ', expl_dict[key]['correctness'])
            print()
      
        pickle.dump(expl_dict, open(result_path, 'wb'))

       
if __name__ == '__main__':
    main()