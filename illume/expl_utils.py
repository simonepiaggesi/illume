import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict
from collections import Counter
from itertools import groupby

from scipy.spatial.distance import cdist
from sklearn.datasets import make_classification

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from rule import get_rule, get_counterfactual_rules_all


def decode_latent_rule(w_latent, expl_binary):

    return ((w_latent!=0.).astype(int)[[i for i,b in enumerate(expl_binary) if b]].sum(axis=0)>0).astype(int)


def decode_latent_rule_complete(x_input, w_latent, expl_dict, dt_dict):

    weight_names = np.array(dt_dict['weight_names'])
    original_feature_names = np.array(dt_dict['original_feature_names'])
    
    rules_xz_all = []
    for k,v in expl_dict.items():
        
        k_dict = {}
        row = int(k[0][1:])
        
        col = np.where(w_latent[row]!=0.)[0]
        expr = ''
        for w, x in zip(weight_names[row][col], original_feature_names[col]):
            expr += w + '*' + x + '+'
        
        k_dict['expr'] = expr[:-1]
        k_dict['idx'] = col
        k_dict['w'] = w_latent[row][col].astype(float)
        k_dict['x'] = x_input[col].astype(float)
    
        k_dict['op'] = k[1]
        k_dict['val'] = v
        rules_xz_all.append(k_dict)

    rules_x_all = []
    for k in rules_xz_all:

        w = k['w']
        x = k['x']
        c = k['val']
        
        if c in [np.inf, -np.inf]:
            for enum_idx, feat_idx in enumerate(k['idx']):
                if w[enum_idx]<0:
                    op_new = '>' if k['op']=='<=' else '<='
                    val_new = np.inf if c==-np.inf else -np.inf
                else:
                    op_new = k['op']
                    val_new = c
                rules_x_all.append((feat_idx, 'x'+str(feat_idx), op_new, val_new))            
        else:
            for enum_idx, feat_idx in enumerate(k['idx']):
                val_new = c - (np.sum(w*x) - w[enum_idx]*x[enum_idx])
                val_new /= w[enum_idx]
    
                if w[enum_idx]<0:
                    op_new = '>' if k['op']=='<=' else '<='
                else:
                    op_new = k['op']
    
                rules_x_all.append((feat_idx, 'x'+str(feat_idx), op_new, val_new))

    rules_x_all = list(set(rules_x_all))

    #check consistency
    #rules_x_df = pd.DataFrame(rules_x_all, columns=['i', 'x', 'op', 'val'])
    #assert(np.all(rules_x_df.groupby(['i', 'x']).apply(lambda r: r[r.op=='<='].val.min() >= r[r.op=='>'].val.max()).values))

    rules_x_all = sorted(rules_x_all, key=lambda row: (row[0], row[2]))
    rules_x_all = [(*g_key, min([v[3] for v in g])) if g_key[2]=='<=' else (*g_key, max([v[3] for v in g]))  
       for g_key, g in groupby(rules_x_all, key= lambda row: (row[0], row[1], row[2]))]

    rules_x_df_dict = { (row[1], row[2]): row[3] for row in rules_x_all}

    expl_dict_new = {}
    for x in original_feature_names:
        for op in ['<=', '>']:
            if (x, op) not in rules_x_df_dict:
                expl_dict_new[(x, op)] = np.inf if op=='<=' else -np.inf
            else:
                expl_dict_new[(x, op)] = rules_x_df_dict[(x, op)]
           
    return expl_dict_new


def inverse_transform_rule_complete(expl_dict, scaler, numeric_idx):
    expl_dict_new = {}
    for (x, op) in expl_dict:
        val = expl_dict[(x, op)]
        expl_dict_new[(x, op)] = val
        if val not in [np.inf, -np.inf]:
            idx = int(x[1:])
            if idx in numeric_idx:
                idx_scaler = StandardScaler()
                idx_scaler.mean_, idx_scaler.scale_ = scaler.mean_[numeric_idx.index(idx)], scaler.scale_[numeric_idx.index(idx)] 
                expl_dict_new[(x, op)] = idx_scaler.inverse_transform(np.array([[val]]))[0][0]

    return expl_dict_new

def fix_cat_rule_complete(expl_dict, numeric_idx):
    expl_dict_new = {}
    for (x, op) in expl_dict:
        val = expl_dict[(x, op)]
        expl_dict_new[(x, op)] = val
        if val not in [np.inf, -np.inf]:
            idx = int(x[1:])
            if idx not in numeric_idx:
                if op=='>':
                    assert(val<=(1.+1e-3))
                    if val<0.:
                        expl_dict_new[(x, op)] =-np.inf
                else:
                    assert(val>=(0.-1e-3))
                    if val>1.:
                        expl_dict_new[(x, op)] = np.inf

    return expl_dict_new
           

def feature_importance_similarity(a, b, metric='cosine'):
    val = 1.0 - cdist(a.reshape(1, -1), b.reshape(1, -1), metric=metric)[0][0]
    return val


def rule_based_similarity_complete(a, b, p=2):
    score = 0.0
    features = set(a.keys() | b.keys())
    den = 0
    for f in features:
        default = np.inf if f[1] == '<=' else -np.inf

        v_a = a[f] if f in a else default
        v_b = b[f] if f in b else default

        if (v_a == v_b and v_a == np.inf) or (v_a == v_b and v_a == -np.inf):
            continue
        den += 1
        if p==2:
            val = 1 / (1 + (v_a - v_b)**2)
        elif p==1:
            val = 1 / (1 + np.abs(v_a - v_b))

        score += val
    if den > 0:
        score = score / den
    return score

def get_rule_explanation_all(X_test, srbc, n_features, get_values=False):

    dt = srbc['dt']
    feature_names = srbc['feature_names']
    class_name = srbc['class_name']
    class_values = srbc['class_values']
    numeric_columns = srbc['numeric_columns']

    rules = [get_rule(x[:n_features], dt, feature_names, class_name, class_values, numeric_columns) for x in X_test]

    explanation_mask_list = [] 
    explanation_dict_list = []

    for rule in rules:

        explanation_mask = list()
        explanation_dict = dict()
        rule_premise = defaultdict(float)
        for p in rule.premises:
            sign = 1 if p.op == '>' else -1
            val = sign * p.thr
            rule_premise[p.att] += val
    
            explanation_dict[(p.att, p.op)] = p.thr
    
        for feature in sorted(feature_names):
            if not get_values:
                val = 1 if feature in rule_premise else 0
                explanation_mask.append(val)
            else:
                val = rule_premise[feature] if feature in rule_premise else 0.0
                explanation_mask.append(val)
    
            if (feature, '<=') not in explanation_dict:
                explanation_dict[(feature, '<=')] = np.inf
            if (feature, '>') not in explanation_dict:
                explanation_dict[(feature, '>')] = -np.inf
    
        explanation_mask = np.array(explanation_mask)
        explanation_mask_list.append(explanation_mask)
        explanation_dict_list.append(explanation_dict)

    return explanation_mask_list, explanation_dict_list


def get_crule_explanation_all(X_test, Y_test, srbc, n_features, bb_predict=None, get_values=False):

    dt = srbc['dt']
    feature_names = srbc['feature_names']
    class_name = srbc['class_name']
    class_values = srbc['class_values']
    numeric_columns = srbc['numeric_columns']

    features_map, features_map_inv = (None, None)
    if bb_predict is not None:
        features_map, features_map_inv = (srbc['features_map'], srbc['features_map_inv'])

    X = srbc['X']
    Y = srbc['Y']
    cfs_tuples = get_counterfactual_rules_all(X_test[:, :n_features], Y_test, dt, 
                                              X[dt.predict(X)==Y, :n_features], Y[dt.predict(X)==Y], 
                                            feature_names, class_name, class_values, numeric_columns, features_map, features_map_inv, bb_predict)

    icfs_list = [] 
    explanations_mask_list = [] 
    explanations_dict_list = []
    
    for icfs, deltas, crules in cfs_tuples:
    
        assert(len(crules) == len(icfs))
        explanations_mask = list()
        explanations_dict = list()
        
        # for rule in crules:
        #     explanation_mask = list()
        #     explanation_dict = dict()
        #     rule_premise = defaultdict(float)
        #     for p in rule.premises:
        #         sign = 1 if p.op == '>' else -1
        #         val = sign * p.thr
        #         rule_premise[p.att] += val
    
        #         explanation_dict[(p.att, p.op)] = p.thr

        for delta in deltas:
            explanation_mask = list()
            explanation_dict = dict()
            rule_premise = defaultdict(float)
            for p in delta:
                sign = 1 if p.op == '>' else -1
                val = sign * p.thr
                rule_premise[p.att] += val
    
                explanation_dict[(p.att, p.op)] = p.thr
        
            for feature in sorted(feature_names):
                if not get_values:
                    val = 1 if feature in rule_premise else 0
                    explanation_mask.append(val)
                else:
                    val = rule_premise[feature] if feature in rule_premise else 0.0
                    explanation_mask.append(val)
    
                if (feature, '<=') not in explanation_dict:
                    explanation_dict[(feature, '<=')] = np.inf
                if (feature, '>') not in explanation_dict:
                    explanation_dict[(feature, '>')] = -np.inf
            
            explanation_mask = np.array(explanation_mask)
            explanations_mask.append(explanation_mask)
            explanations_dict.append(explanation_dict)

        icfs_list.append(icfs)
        explanations_mask_list.append(explanations_mask)
        explanations_dict_list.append(explanations_dict)

    return icfs_list, explanations_mask_list, explanations_dict_list
