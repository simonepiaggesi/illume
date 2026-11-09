import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import os
import tempfile

import pickle
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from .utils import linear_eval, tree_eval
from .expl_utils import get_rule_explanation_all, get_crule_explanation_all, decode_latent_rule_complete, inverse_transform_rule_complete, fix_cat_rule_complete, simplify_rule_complete
from scipy.spatial.distance import cdist

from more_itertools import random_combination
from itertools import product
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Input: nested feature-index groups; output: one flat list of indices.
flatten = lambda m: [item for row in m for item in row]
        

class LocalLinearAutoEnc(nn.Module):
    def __init__(self, input_dim, latent_dim, n_layers=3, dropout=0.0):
        """Build an encoder for input_dim features and latent_dim projections.

        Args:
            input_dim: Number of input features.
            latent_dim: Number of latent features.
            n_layers: Encoder depth; dropout: dropout probability.

        Returns:
            None; initializes the encoder.
        """
        super(LocalLinearAutoEnc, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        enc_hidden_dims = np.logspace(np.log2(input_dim), np.log2(input_dim*latent_dim), num=n_layers+1, base=2).astype(int)[1:-1]
        self.encoder = MLP(input_dim, list(enc_hidden_dims)+[input_dim*latent_dim], bias=True, dropout=dropout)
        
    def encode(self, x, num_k=None):
        """Project input rows into a local latent space.

        Args:
            x: Tensor of shape (batch, input_dim).
            num_k: Optional number of coefficients retained per projection.

        Returns:
            Normalized weights (batch, latent_dim, input_dim) and codes (batch, latent_dim).
        """
        self.weight = torch.reshape(self.encoder(x), (-1, self.latent_dim, self.input_dim))
        w = self.get_weight(self.weight, num_k)
        w_norm = w/torch.norm(w, p=2, dim=-1)[:,:,None]
        z = torch.bmm(torch.unsqueeze(x, dim=1), torch.transpose(w_norm, 1,2))[:, 0, :]
        return w_norm, z

    def decode(self, z, decoder_weights=None):
        """Reconstruct input rows from latent codes.

        Args:
            z: Codes of shape (batch, latent_dim).
            decoder_weights: Optional weights of shape (batch, latent_dim, input_dim);
                defaults to the weights from the latest encode call.

        Returns:
            Reconstructed tensor of shape (batch, input_dim).
        """
        weights = self.weight if decoder_weights is None else decoder_weights
        x_hat = torch.bmm(torch.unsqueeze(z, dim=1), weights)[:, 0, :]
        return x_hat

    def forward(self, x, num_k_sparse=None):
        """Encode and reconstruct a batch.

        Args:
            x: Input tensor of shape (batch, input_dim).
            num_k_sparse: Optional retained coefficients per projection.

        Returns:
            Reconstructed tensor of shape (batch, input_dim).
        """
        w, z = self.encode(x, num_k_sparse)
        x_hat = self.decode(z)
        return x_hat    

    def get_weight(self, w_batch, num_k=None):
        """Apply a straight-through top-k mask to projection weights.

        Args:
            w_batch: Weights of shape (batch, latent_dim, input_dim).
            num_k: Coefficients retained per projection, or None for all.

        Returns:
            Weight tensor with the same shape.
        """
        if num_k == None or num_k >= self.input_dim:
            return w_batch
        else:
            indices = torch.argsort(torch.abs(w_batch))[:,:,-num_k:]
            mask_batch = torch.zeros_like(w_batch)
            mask_batch.scatter_(2, indices, 1)
            masked_weight = torch.mul(w_batch, mask_batch)
            #return masked_weight
            return w_batch + (masked_weight - w_batch).detach()
            
def compute_similarity_z(Z, sigma=1):
    """Compute row-normalized latent-code similarities.

    Args:
        Z: Latent codes of shape (batch, latent_dim).
        sigma: Gaussian kernel width.

    Returns:
        Similarity tensor of shape (batch, batch).
    """
    D = 1 - F.cosine_similarity(Z[:, None, :], Z[None, :, :], dim=-1)
    M = torch.exp((-D**2)/(2*sigma**2))
    return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0))).transpose(0,1)

def compute_similarity_w(W, sigma=1):
    """Compute row-normalized projection-weight similarities.

    Args:
        W: Weights of shape (batch, latent_dim, input_dim).
        sigma: Gaussian kernel width.

    Returns:
        Similarity tensor of shape (batch, batch).
    """
    D = 1 - torch.matmul(W[:,0,:], W[:,0,:].T)
    for l in range(1, W.shape[1]):
        D += 1 - torch.matmul(W[:,l,:], W[:,l,:].T)
    D /= W.shape[1]

    M = torch.exp((-D**2)/(2*sigma**2))
    return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0))).transpose(0,1)

def compute_similarity_y(y, sigma=1):
    """Compute row-normalized label similarities.

    Args:
        y: Label vectors of shape (batch, n_classes).
        sigma: Gaussian kernel width.

    Returns:
        Similarity tensor of shape (batch, batch).
    """
    D = torch.cdist(y, y)
    # D = 1 - F.cosine_similarity(y[:, None, :], y[None, :, :], dim=-1)
    M = torch.exp((-D**2)/(2*sigma**2))
    return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0))).transpose(0,1)

def compute_similarity_x(X, idx_cat=None, sigma=1):
    """Compute row-normalized input similarities.

    Args:
        X: Input tensor of shape (batch, input_dim).
        idx_cat: Optional categorical feature indices.
        sigma: Gaussian kernel width.

    Returns:
        Similarity tensor of shape (batch, batch).
    """
    if idx_cat:
        X_cat = X[:, idx_cat]
        X_cont = X[:, np.delete(range(X.shape[1]), idx_cat)]
        h = X_cat.shape[1]
        m = X.shape[1]
        D_cat = 2*torch.cdist(X_cat, X_cat, p=0)/h
        D = h/m * D_cat
        if h<m:
            D_cont = 1 - F.cosine_similarity(X_cont[:, None, :], X_cont[None, :, :], dim=-1)
            D += ((m-h)/m) * D_cont
    else:
        D = 1 - F.cosine_similarity(X[:, None, :], X[None, :, :], dim=-1) 
    M = torch.exp((-D**2)/(2*sigma**2))
    return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0))).transpose(0,1)


def kld_loss_function_zxy(X, Z, y, idx_cat=None, sigma=1):
    """Compare latent similarities with input and available label similarities.

    Args:
        X: Input batch; Z: latent-code batch; y: label-vector batch.
        idx_cat: Optional categorical indices; sigma: kernel width.

    Returns:
        Scalar KL-divergence loss tensor.
    """
    similarity_KLD = torch.nn.KLDivLoss(reduction='batchmean')
    Sx = compute_similarity_x(X, idx_cat, sigma)
    Sz = compute_similarity_z(Z, sigma)
    loss = similarity_KLD(torch.log(Sz), Sx)
    if not torch.any(torch.isnan(y)):
        Sy = compute_similarity_y(y, sigma)
        loss += similarity_KLD(torch.log(Sz), Sy)
    return loss


def kld_loss_function_wxyz(X, W, Z, y, idx_cat=None, sigma=1):
    """Align input, weight, latent, and available label similarities.

    Args:
        X: Input batch; W: projection weights; Z: latent codes; y: labels.
        idx_cat: Optional categorical indices; sigma: kernel width.

    Returns:
        Scalar sum of KL-divergence losses.
    """
    similarity_KLD = torch.nn.KLDivLoss(reduction='batchmean')
    Sx = compute_similarity_x(X, idx_cat, sigma)
    Sw = compute_similarity_w(W, sigma)
    Sz = compute_similarity_z(Z, sigma)

    loss = similarity_KLD(torch.log(Sz), Sx) + similarity_KLD(torch.log(Sw), Sz)
    if not torch.any(torch.isnan(y)):
        Sy = compute_similarity_y(y, sigma)
        loss += similarity_KLD(torch.log(Sz), Sy)
    return loss

def orth_loss_function(Q):
    """Penalize deviation of weight Gram matrices from identity.

    Args:
        Q: Batched square Gram matrices of latent projections.

    Returns:
        Scalar mean-squared penalty tensor.
    """
    loss = ((Q - torch.eye(Q.shape[1]).to(device))**2).mean(dim=[1,2]).mean()
    return loss

def corr_loss_function(Z):

    """Penalize correlation among latent dimensions.

    Args:
        Z: Latent codes of shape (batch, latent_dim).

    Returns:
        Scalar correlation penalty tensor.
    """
    def pearson_matrix(X):
        """Compute normalized cross-products for latent columns.

        Args:
            X: Tensor of shape (batch, latent_dim).

        Returns:
            Tensor of shape (1, latent_dim, latent_dim).
        """
        return torch.mm(batch_norm(X, 0).T, batch_norm(X, 0))[None,:,:]/X.shape[0]

    C = pearson_matrix(Z)
    
    loss = ((torch.abs(C) - torch.eye(C.shape[1]).to(device))**2).mean(dim=[1,2]).mean()
    return loss

def rec_loss_function(X, X_hat, idx_cat_lol):
    """Measure reconstruction error for numeric and categorical groups.

    Args:
        X: Input batch; X_hat: reconstructed batch.
        idx_cat_lol: Lists of one-hot column indices for categorical features.

    Returns:
        Scalar reconstruction loss tensor.
    """
    if len(idx_cat_lol)>0:
        loss = torch.mean(torch.cat([F.cross_entropy(X_hat[:,idx], X[:,idx], reduction='none')[:, None]
                            for idx in idx_cat_lol], axis=1), axis=1).mean()

        idx_cont = np.delete(range(X.shape[1]), flatten(idx_cat_lol))
        if len(idx_cont)>0:
            loss += torch.mean(torch.cat([(X_hat[:,[idx]]-X[:,[idx]])**2 
                                for idx in idx_cont], axis=1), axis=1).mean()
    else:
        loss = torch.mean((X_hat-X)**2, axis=1).mean()

    return loss


def compute_jacobian(x, fx, n=-1):
    """Estimate the squared Jacobian norm of fx with respect to x.

    Copyright (c) Facebook, Inc. and its affiliates.

    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.

    PyTorch implementation of Jacobian regularization described in [1].

    Source: https://github.com/facebookresearch/jacobian_regularizer

    [1] Judy Hoffman, Daniel A. Roberts, and Sho Yaida,
        "Robust Learning with Jacobian Regularization," 2019.
        [arxiv:1908.02729](https://arxiv.org/abs/1908.02729)

    Args:
        x: Differentiable input tensor of shape (batch, input_dim).
        fx: Output tensor of shape (batch, output_dim) computed from x.
        n: Number of random projections; -1 uses every output dimension.

    Returns:
        Scalar estimate of the squared Jacobian norm.
    """
    
    def _random_vector(C, B):
        """Sample a unit-length projection vector for each row.

        Args:
            C: Output dimension; B: batch size.

        Returns:
            Tensor of shape (B, C), or (B,) when C is one.
        """
        if C == 1: 
            return torch.ones(B)
        v=torch.randn(B,C)
        arxilirary_zero=torch.zeros(B,C)
        vnorm=torch.norm(v, 2, 1,True)
        v=torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
        return v

    def _jacobian_vector_product(y, x, v, create_graph=False): 
        """Differentiate y against x using projection v.

        Args:
            y: Output tensor; x: input tensor; v: output-shaped projection.
            create_graph: Whether to retain a differentiable gradient graph.

        Returns:
            Projected gradient tensor with the shape of x.
        """
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        grad_x, = torch.autograd.grad(outputs=flat_y, inputs=x, grad_outputs=flat_v, 
                                        retain_graph=True, 
                                        create_graph=create_graph)
        return grad_x

    assert n == -1 or n > 0
    
    b,c = fx.shape
    if n == -1 or n >= c:
        num_proj = c
    else:
        num_proj = n
    J2 = 0
    for ii in range(num_proj):
        if n == -1 or n >= c:
            v=torch.zeros(b,c)
            v[:,ii]=1
        else:
            v = _random_vector(C=c,B=b)
        if x.is_cuda:
            v = v.cuda()
        Jv = _jacobian_vector_product(fx, x, v, create_graph=True)
        J2 += c*torch.norm(Jv)**2 / (num_proj*b)
    return J2

def batch_norm(X, axis=0):
    """Standardize a tensor along one axis.

    Args:
        X: Input tensor; axis: dimension used for mean and variance.

    Returns:
        Tensor with the same shape and normalized values.
    """
    mean = X.mean(axis=axis, keepdim=True)
    var = X.var(axis=axis, unbiased=False, keepdim=True)
    X_norm = (X - mean) / (var + 1e-15).sqrt()
    return X_norm


class ILLUME(torch.nn.Module):
    def __init__(self, latent_dim=2, mlp_layers=3, dropout=0.0, nj_proj=1,
                 max_epochs=1000, early_stopping=30, 
                 learning_rate=0.001, batch_size=1024, sigma=1):
        """Configure an ILLUME model; data is supplied by fit or load.

        Args:
            latent_dim: Latent features (at least two); mlp_layers: encoder depth.
            dropout: Encoder dropout; nj_proj: Jacobian projection count.
            max_epochs: Training limit; early_stopping: patience in epochs.
            learning_rate: Adam step size; batch_size: samples per batch.
            sigma: Similarity kernel width.

        Returns:
            None; stores configuration.
        """
        super().__init__()

        self.latent_dim=latent_dim
        self.mlp_layers=mlp_layers
        self.dropout=dropout
        self.nj=nj_proj

        self.max_epochs=max_epochs
        self.early_stopping=early_stopping

        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.sigma=sigma

        if latent_dim < 2:
            raise ValueError("latent_dim must be greater than 1")
        if not 0 <= dropout <= 1:
            raise ValueError('dropout must be between 0 and 1')
        if sigma <= 0:
            raise ValueError("sigm must be greater than 0")

    def _set(self, X, y, idx_num_cat, init_path=None):

        """Initialize data, feature groups, model, and optimizer.

        Args:
            X: Pair of training and validation feature arrays.
            y: Optional pair of label arrays; None denotes unavailable labels.
            idx_num_cat: Feature-index groups, singletons for numeric features.
            init_path: Optional path to saved model weights.

        Returns:
            None; updates the instance state.
        """
        self.X_train, self.X_val = X

        if y is None:
            self.y_train = np.zeros_like(self.X_train[:,0], dtype=float)
            self.y_train[:] = float('nan')
            self.y_val = np.zeros_like(self.X_val[:,0], dtype=float)
            self.y_val[:] = float('nan')
        else:
            self.y_train, self.y_val = y

        self.input_dim = self.X_train.shape[1]
        
        self.idx_num_cat = idx_num_cat
        self.idx_num = flatten([l for l in self.idx_num_cat if len(l)==1])
        self.idx_cat = flatten([l for l in self.idx_num_cat if len(l)>1])
        if len(self.idx_cat)==0:
            self.idx_cat = None

        if 'model' not in self.__dict__['_modules']:
            self.model = LocalLinearAutoEnc(self.input_dim, self.latent_dim, self.mlp_layers, self.dropout).to(device)
            
        num_trainable_params = sum([p.numel() for p in self.model.parameters()])
        print('num. parameters = ' + str(num_trainable_params))

        if init_path!=None:
            assert(os.path.isfile(init_path))
            self.model.load_state_dict(torch.load(init_path, map_location=device))

        model_params = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(model_params, lr=self.learning_rate)

    def load(self, X, y, idx_num_cat, init_path):

        """Initialize the model from saved weights and supplied data.

        Args:
            X: Training and validation feature arrays; y: their label arrays.
            idx_num_cat: Feature-index groups; init_path: weights file path.

        Returns:
            None; populates the model state.
        """
        self._set(X, y, idx_num_cat, init_path)

    def fit(self, X, y=None, idx_num_cat=None,
            params_dict={'num_k':None, 'l_rec':1.0, 'l_kld':1.0, 'l_so':1.0, 'l_co':1.0, 'l_st':1.0}, 
            init_path=None, seed=None):

        """Fit the encoder and retain training projections.

        Args:
            X: Training and validation feature arrays; y: optional label arrays.
            idx_num_cat: Feature-index groups, singletons for numeric features.
            params_dict: Sparsity setting num_k and five loss weights.
            init_path: Optional initial weights; seed: optional random seed.

        Returns:
            Training and validation loss histories; each epoch records
            (reconstruction, KL, orthogonality, correlation, Jacobian, total).
        """
        self._set(X, y, idx_num_cat, init_path)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if (params_dict['num_k']!=None) and (params_dict['num_k']<self.input_dim):
            start_k = self.input_dim
            end_k = params_dict['num_k'] 
            epochs_per_step = 30
            n_epochs_decaying = (start_k - end_k)*epochs_per_step

            if n_epochs_decaying > self.max_epochs:
                n_epochs_decaying = self.max_epochs
                epochs_per_step = n_epochs_decaying//(start_k - end_k)

            ks = [(start_k-1) - int((start_k - end_k)/(n_epochs_decaying)*i) for i in range(n_epochs_decaying)]
            ks += [end_k,]*int(n_epochs_decaying*0.3)

            print(f'Training Meta-encoder with sparsity scheduling with max {len(ks)} iterations.')
            train_losses, test_losses = self._train(num_k_list=ks, 
                                                    lambdas=(params_dict['l_rec'], params_dict['l_kld'], 
                                                             params_dict['l_so'], params_dict['l_co'], params_dict['l_st']), 
                                                    early_stop_from_epoch=n_epochs_decaying)
            print()
        else:
            print(f'Training Meta-encoder with max {self.max_epochs} iterations.')
            train_losses, test_losses = self._train(num_k_list=[None,]*self.max_epochs, 
                                                    lambdas=(params_dict['l_rec'], params_dict['l_kld'], 
                                                             params_dict['l_so'], params_dict['l_co'], params_dict['l_st']), 
                                                    early_stop_from_epoch=1)
            print()

        self.params_dict = params_dict

        self.W_train, self.Z_train = self.transform(self.X_train, num_k_sparse=params_dict['num_k'])

        if (self.W_train.shape[0]==1) and (self.X_train.shape[0]>1):
            self.W_train = np.repeat(self.W_train, self.X_train.shape[0], axis=0)

        return train_losses, test_losses


    def explain_linear(self, class_label=1, num_k=None):

        """Fit a binary logistic surrogate in latent space.

        Args:
            class_label: Class index treated as positive.
            num_k: Optional retained coefficients per projection.

        Returns:
            None; stores the surrogate and its training data.
        """
        self.W_train, self.Z_train = self.transform(self.X_train, num_k_sparse=num_k)
        self.W_val, self.Z_val = self.transform(self.X_val, num_k_sparse=num_k)

        if (self.W_train.shape[0]==1) and (self.X_train.shape[0]>1):
            self.W_train = np.repeat(self.W_train, self.X_train.shape[0], axis=0)

        self.y_train_bb = (np.argmax(self.y_train, axis=1)==class_label).astype(int)
        self.y_val_bb = (np.argmax(self.y_val, axis=1)==class_label).astype(int)

        print(f'Training Logistic Regression surrogate.')
        self.logreg, f1 = linear_eval(self.Z_train, self.y_train_bb, self.Z_val, self.y_val_bb, f1_average='binary')
        print('LR surrogate score:', '%0.4f'%f1)

        self.lrd = {'X':self.Z_train,
                   'Y':self.y_train_bb,
                   'lr': self.logreg,
                   'feature_names': ['z'+str(i) for i in range(self.Z_train.shape[1])],
                   'original_feature_names': ['x'+str(j) for j in range(self.X_train.shape[1])],
                   'weight_names': [['w_'+str(i)+'_'+str(j) for j in range(self.X_train.shape[1])] for i in range(self.Z_train.shape[1])],
                   'class_name': 'class',
                   'class_values': [0,1],
                   'numeric_columns':['z'+str(i) for i in range(self.Z_train.shape[1])],
                   'X_val':self.Z_val,
                   'Y_val':self.y_val_bb,
                }

        return

    def explain_dectree(self, class_label=1, num_k=None):

        """Fit a binary decision-tree surrogate in latent space.

        Args:
            class_label: Class index treated as positive.
            num_k: Optional retained coefficients per projection.

        Returns:
            None; stores the surrogate and its training data.
        """
        self.W_train, self.Z_train = self.transform(self.X_train, num_k_sparse=num_k)
        self.W_val, self.Z_val = self.transform(self.X_val, num_k_sparse=num_k)

        if (self.W_train.shape[0]==1) and (self.X_train.shape[0]>1):
            self.W_train = np.repeat(self.W_train, self.X_train.shape[0], axis=0)

        self.y_train_bb = (np.argmax(self.y_train, axis=1)==class_label).astype(int)
        self.y_val_bb = (np.argmax(self.y_val, axis=1)==class_label).astype(int)

        print(f'Training Decision Tree surrogate.')
        self.dtree, f1 = tree_eval(self.Z_train, self.y_train_bb, self.Z_val, self.y_val_bb, f1_average='binary')
        print('DT surrogate score:', '%0.4f'%f1)

        self.dtd = {'X':self.Z_train,
               'Y':self.y_train_bb,
               'dt': self.dtree,
               'feature_names': ['z'+str(i) for i in range(self.Z_train.shape[1])],
               'original_feature_names': ['x'+str(j) for j in range(self.X_train.shape[1])],
               'weight_names': [['w_'+str(i)+'_'+str(j) for j in range(self.X_train.shape[1])] for i in range(self.Z_train.shape[1])],
               'class_name': 'class',
               'class_values': [0,1],
               'numeric_columns':['z'+str(i) for i in range(self.Z_train.shape[1])],
               'X_val':self.Z_val,
               'Y_val':self.y_val_bb,
                }

        return

    def _get_feature_importance_training(self):

        """Map surrogate coefficients to original training features.

        Args:
            None; uses the fitted logistic surrogate and training weights.

        Returns:
            Feature-importance array of shape (n_train, input_dim).
        """
        ex_train = (self.logreg.coef_[None,:,:].transpose(0,2,1)*self.W_train).sum(axis=1)

        return ex_train

    def _get_rules_training(self):

        """Decode training decision rules into input features.

        Args:
            None; uses the fitted decision-tree surrogate.

        Returns:
            List of decoded rule dictionaries, one per training row.
        """
        _, ez_dict_training = get_rule_explanation_all(self.Z_train, self.dtd, n_features=self.latent_dim)

        ex_dict_training = [decode_latent_rule_complete(x, w, e, self.dtd) for x,w,e in zip(self.X_train, self.W_train, ez_dict_training)]
        ex_dict_training = [fix_cat_rule_complete(e, self.idx_num) for e in ex_dict_training]

        return ex_dict_training

    def _get_crules_training(self):

        """Find training counterfactual rules and source rows.

        Args:
            None; uses the fitted decision-tree surrogate.

        Returns:
            Counterfactual rule dictionaries and matching training-row arrays.
        """
        cond_train = self.dtree.predict(self.Z_train)==self.y_train_bb

        cf_idx_training, _, cx_dict_training = get_crule_explanation_all(self.Z_train, self.X_train, self.y_train_bb, 
                                                                        self.X_train[cond_train], self.W_train[cond_train], 
                                                                        self.dtd, n_features=self.latent_dim)

        cf_list_training = [np.atleast_2d(self.X_train[cond_train][cdx]) for cdx in cf_idx_training]

        return cx_dict_training, cf_list_training

    
    def _step(self, X_batch, Y_batch, num_k, with_grads=True, lambda_tuple=None):

        """Compute the five component losses for one batch.

        Args:
            X_batch: Feature tensor; Y_batch: label tensor.
            num_k: Optional sparse coefficient count.
            with_grads: Gradient flag for zero-valued loss placeholders.
            lambda_tuple: Optional weights enabling each loss component.

        Returns:
            Scalar tensors for reconstruction, KL, orthogonality,
            correlation, and Jacobian losses, in that order.
        """
        if lambda_tuple is None:
            l_rec, l_kld, l_orth, l_corr, l_jac = (1.0, 1.0, 1.0, 1.0, 1.0)
        else:
            l_rec, l_kld, l_orth, l_corr, l_jac = lambda_tuple
    
        X_batch.requires_grad_(True)
        W_batch, Z_batch = self.model.encode(X_batch, num_k)

        if l_jac==0.0:
            jac_loss = torch.tensor(0., requires_grad=with_grads)
        else:
            jac_loss = compute_jacobian(X_batch, Z_batch - torch.bmm(torch.unsqueeze(X_batch, dim=1), 
                                            torch.transpose(W_batch.detach(), 1,2)).squeeze(), self.nj)/(self.latent_dim*self.input_dim)

        if l_kld==0.0:
            kld_loss = torch.tensor(0., requires_grad=with_grads)
        else:
            kld_loss = kld_loss_function_wxyz(X_batch, W_batch, Z_batch, Y_batch, self.idx_cat, self.sigma)

        if l_orth==0.0:
            orth_loss = torch.tensor(0., requires_grad=with_grads)
        else:
            orth_loss = orth_loss_function(torch.bmm(torch.abs(W_batch), 
                                  torch.transpose(torch.abs(W_batch), 1,2)))

        if l_corr==0.0:
            corr_loss = torch.tensor(0., requires_grad=with_grads)
        else:                       
            corr_loss = corr_loss_function(Z_batch)
    
        X_hat = self.model.decode(Z_batch)
        if l_rec==0.0:
            rec_loss = torch.tensor(0., requires_grad=with_grads)
        else: 
            rec_loss = rec_loss_function(X_batch, X_hat, [idx for idx in self.idx_num_cat if len(idx)>1])
    
        return rec_loss, kld_loss, orth_loss, corr_loss, jac_loss  
        

    def _train(self, num_k_list, lambdas, early_stop_from_epoch):

        """Run training and validation across a sparsity schedule.

        Args:
            num_k_list: Sparse coefficient setting for each epoch.
            lambdas: Five loss weights in _step order.
            early_stop_from_epoch: First epoch eligible for checkpointing.

        Returns:
            Training and validation histories of six-value loss tuples.
        """
        train_dataset = TensorDataset(torch.tensor(self.X_train).float().to(device), torch.tensor(self.y_train).float().to(device))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) 

        test_dataset = TensorDataset(torch.tensor(self.X_val).float().to(device), torch.tensor(self.y_val).float().to(device))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) 

        l_rec, l_kld, l_orth, l_corr, l_jac = lambdas

        epoch_train_losses = []
        epoch_val_losses = []
        epoch = 1
        best = np.inf
        # progress bar
        pbar = tqdm(bar_format="{postfix[0]} {postfix[1][value]:03d} {postfix[2]} {postfix[3][value]:.5f} {postfix[4]} {postfix[5][value]:.5f} {postfix[6]} {postfix[7][value]:d}",
            postfix=["Epoch:", {'value':0}, "Train Loss", {'value':0}, "Test Loss", {'value':0}, "Early Stopping", {"value":0}])

        with tempfile.TemporaryDirectory(dir = './') as dname:
            # start training
            for k in num_k_list:
                # ------- TRAIN ------- #
                # set model as training mode
                self.model.train()
                batch_rec, batch_kld, batch_orth, batch_corr, batch_jac, batch_loss = [], [], [], [], [], []
                for batch, (X_batch, Y_batch) in enumerate(train_loader):

                    if X_batch.shape[0]==1:
                        continue

                    self.optimizer.zero_grad()
                    rec_loss, kld_loss, orth_loss, corr_loss, jac_loss = self._step(X_batch, Y_batch, k, True, lambdas)
                    
                    total_loss = l_rec*rec_loss + l_kld*kld_loss + l_orth*orth_loss + l_corr*corr_loss + l_jac*jac_loss  
                    total_loss.backward()
                    self.optimizer.step()
                    
                    batch_loss.append(total_loss.item())
                    batch_rec.append(rec_loss.item())
                    batch_kld.append(kld_loss.item())
                    batch_orth.append(orth_loss.item())
                    batch_corr.append(corr_loss.item())
                    batch_jac.append(jac_loss.item())

                # save result
                epoch_train_losses.append((np.mean(batch_rec), np.mean(batch_kld), np.mean(batch_orth),
                                           np.mean(batch_corr), np.mean(batch_jac), np.mean(batch_loss)))
                pbar.postfix[3]["value"] = np.mean(batch_loss)

                # -------- VALIDATION --------
                # set model as testing mode
                self.model.eval()
                batch_rec, batch_kld, batch_orth, batch_corr, batch_jac, batch_loss = [], [], [], [], [], []
                for batch, (X_batch, Y_batch) in enumerate(test_loader):
                    #with torch.no_grad():

                    if X_batch.shape[0]==1:
                        continue

                    self.model.zero_grad()
                    rec_loss, kld_loss, orth_loss, corr_loss, jac_loss  = self._step(X_batch, Y_batch, k, False, lambdas)
                    
                    total_loss = l_rec*rec_loss + l_kld*kld_loss + l_orth*orth_loss + l_corr*corr_loss + l_jac*jac_loss  
                    
                    batch_loss.append(total_loss.item())
                    batch_rec.append(rec_loss.item())
                    batch_kld.append(kld_loss.item())
                    batch_corr.append(corr_loss.item())
                    batch_orth.append(orth_loss.item())
                    batch_jac.append(jac_loss.item())

                # save information
                epoch_val_losses.append((np.mean(batch_rec), np.mean(batch_kld), np.mean(batch_orth),
                                           np.mean(batch_corr), np.mean(batch_jac), np.mean(batch_loss)))
                pbar.postfix[5]["value"] = np.mean(batch_loss)
                pbar.postfix[1]["value"] = epoch 

                if epoch >= early_stop_from_epoch:
                    if epoch_val_losses[-1][-1] < best:
                        wait = 0
                        best = epoch_val_losses[-1][-1]
                        best_epoch = epoch
                        torch.save(self.model.state_dict(), dname+'/ModelTemp.pt')
                        torch.save(self.optimizer.state_dict(), dname+'/OptTemp.pt')
                        saved = True
                    else:
                        wait += 1
                    pbar.postfix[7]["value"] = wait
                    if wait == self.early_stopping:
                        break    
                epoch += 1
                pbar.update()
            if saved:
                self.model.load_state_dict(torch.load(dname+'/ModelTemp.pt'))
                self.optimizer.load_state_dict(torch.load(dname+'/OptTemp.pt'))
            else:
                print('Warning: no checkpoint improved on the initial loss; '
                      'keeping the final weights.')

        pbar.close()

        return epoch_train_losses, epoch_val_losses

    def transform(self, X, num_k_sparse=None, return_decoder_weights=False):
        """Encode feature rows with the fitted model.

        Args:
            X: Feature array of shape (n_samples, input_dim).
            num_k_sparse: Optional retained coefficients per projection.
            return_decoder_weights: Include unnormalized weights for decoding.

        Returns:
            Normalized weights (n, latent_dim, input_dim) and codes (n, latent_dim);
            optionally also decoder weights of shape (n, latent_dim, input_dim).
        """
        with torch.no_grad():
            self.model.eval()
            W, Z = self.model.encode(torch.tensor(X).float().to(device), num_k_sparse)
            if return_decoder_weights:
                decoder_weights = self.model.weight.cpu().detach().numpy()
                return W.cpu().detach().numpy(), Z.cpu().detach().numpy(), decoder_weights
        return W.cpu().detach().numpy(), Z.cpu().detach().numpy() 

    def inverse_transform(self, Z, decoder_weights):
        """Decode latent rows using weights paired with those rows.

        Args:
            Z: Latent codes of shape (n_samples, latent_dim).
            decoder_weights: Unnormalized weights returned by transform for Z.

        Returns:
            Reconstructed feature array of shape (n_samples, input_dim).
        """
        Z = torch.tensor(Z).float().to(device)
        decoder_weights = torch.tensor(decoder_weights).float().to(device)
        if Z.ndim != 2 or Z.shape[1] != self.latent_dim:
            raise ValueError('Z must have shape (n_samples, latent_dim)')
        expected_weights_shape = (Z.shape[0], self.latent_dim, self.input_dim)
        if decoder_weights.shape != expected_weights_shape:
            raise ValueError('decoder_weights must have shape (n_samples, latent_dim, input_dim)')
        with torch.no_grad():
            self.model.eval()
            X = self.model.decode(Z, decoder_weights)
            X = torch.cat([ (X[:,idx]==X[:,idx].max(dim=1).values[:,None]).float() if len(idx)>1 else X[:,idx] for idx in self.idx_num_cat], axis=1)
        return X.cpu().detach().numpy() 


    def _latent_update(self, xi, xnn, Wi, Wnn, clf):

        """Move a latent point toward an oppositely classified neighbor.

        Args:
            xi, xnn: Input feature rows for the point and neighbor.
            Wi, Wnn: Their projection-weight matrices.
            clf: Fitted classifier acting on latent codes.

        Returns:
            Updated latent code, projection weights, and input-induced latent shift.
        """
        zi = np.matmul(Wi, xi)
        znn = np.matmul(Wnn, xnn)

        yi = clf.predict([zi])
        ynn = clf.predict([znn])
        assert(np.all(yi != ynn))

        dzw = np.matmul(Wnn-Wi, xi)
        dzx = np.matmul(Wnn, xnn-xi)
        assert(clf.predict([zi+dzx+dzw])==ynn)

        gridw = [0.]+list(np.logspace(-3, 0, 100))
        gridx = [0.]+list(np.logspace(-3, 0, 100))
        gridwx = random_combination(product(gridw, gridx), 2999)

        for aw, ax in sorted(list(gridwx)+[(1.,1.)], key=lambda x: x[0]+x[1]):
            za = zi + aw*dzw + ax*dzx
            if clf.predict([za])!=yi:
                return za, Wi+aw*(Wnn-Wi), ax*dzx
        
        return za, Wnn, dzx

    
    def get_feature_importance(self, x_test, y_test_bb, class_label=1, num_k=None):

        """Explain test rows with original-feature importance vectors.

        Args:
            x_test: Test feature rows; y_test_bb: black-box class labels.
            class_label: Positive class; num_k: optional sparse coefficient count.

        Returns:
            List of importance vectors, or None where no explanation is found.
        """
        y_test_bb = (y_test_bb==class_label).astype(int)

        w_test, z_test = self.transform(x_test, num_k_sparse=num_k)
        if (w_test.shape[0]==1) and (x_test.shape[0]>1):
            w_test = np.repeat(w_test, x_test.shape[0], axis=0)

        # range indices of training set
        idx_train = np.arange(self.Z_train.shape[0])
        # boolean indices where surrogate==black-box in training set
        cond_train = self.logreg.predict(self.Z_train)==self.y_train_bb
        # boolean indices where (surrogate==black-box) but (y_train_surr!=y_test_surr) 
        mask_train = np.array([self.logreg.predict(z.reshape(1,-1))!=self.y_train_bb[cond_train] for z in z_test], dtype=bool)

        # indices of the closest training instance among the training instances with (surrogate==black-box) and (y_train_surr!=y_test_surr) for each test instance
        idx_from_train = [cdist(z.reshape(1,-1), self.Z_train[cond_train], metric='cosine')[0][mask_train[i]].argsort()[0] if np.any(mask_train[i]) else None for i,z in enumerate(z_test)]
        # indices of the closest training instance among all training instances for each test instance
        self.idx_from_train = [idx_train[cond_train][mask_train[i]][t] if np.any(mask_train[i]) else None for i,t in enumerate(idx_from_train)]

        # latent refinement
        _, w_test_new, _ = zip(*[(z_test[i], w_test[i], np.zeros_like(z_test[i])) if (self.logreg.predict(z_test[[i]])==y_test_bb[i] or ~np.any(mask_train[i])) else
                            self._latent_update(x_test[i], self.X_train[self.idx_from_train[i]], w_test[i], self.W_train[self.idx_from_train[i]], self.logreg) for i,_ in enumerate(z_test)])

        ex_test = (self.logreg.coef_[None,:,:].transpose(0,2,1)*np.array(w_test_new)).sum(axis=1)

        ex_test = [ex if (self.logreg.predict(z_test[[i]])==y_test_bb[i] or np.any(mask_train[i])) else
                         None for i,ex in enumerate(ex_test)]

        return ex_test


    def get_decision_rules(self, x_test, y_test_bb, class_label=1, num_k=None):

        """Explain test rows with decoded decision rules.

        Args:
            x_test: Test feature rows; y_test_bb: black-box class labels.
            class_label: Positive class; num_k: optional sparse coefficient count.

        Returns:
            List of rule dictionaries, or None where no rule is found.
        """
        y_test_bb = (y_test_bb==class_label).astype(int)

        w_test, z_test = self.transform(x_test, num_k_sparse=num_k)
        if (w_test.shape[0]==1) and (x_test.shape[0]>1):
            w_test = np.repeat(w_test, x_test.shape[0], axis=0)

        # range indices of training set
        idx_train = np.arange(self.Z_train.shape[0])
        # boolean indices where surrogate==black-box in training set
        cond_train = self.dtree.predict(self.Z_train)==self.y_train_bb
        # boolean indices where (surrogate==black-box) but (y_train_surr!=y_test_surr) 
        mask_train = np.array([self.dtree.predict(z.reshape(1,-1))!=self.y_train_bb[cond_train] for z in z_test], dtype=bool)

        # indices of the closest training instance among the training instances with (surrogate==black-box) and (y_train_surr!=y_test_surr) for each test instance
        idx_from_train = [cdist(z.reshape(1,-1), self.Z_train[cond_train], metric='cosine')[0][mask_train[i]].argsort()[0] if np.any(mask_train[i]) else None for i,z in enumerate(z_test)]
        # indices of the closest training instance among all training instances for each test instance
        self.idx_from_train = [idx_train[cond_train][mask_train[i]][t] if np.any(mask_train[i]) else None for i,t in enumerate(idx_from_train)]

        # latent refinement
        z_test_new, w_test_new, eps_new = zip(*[(z_test[i], w_test[i], np.zeros_like(z_test[i])) if (self.dtree.predict(z_test[[i]])==y_test_bb[i] or ~np.any(mask_train[i])) else
                            self._latent_update(x_test[i], self.X_train[self.idx_from_train[i]], w_test[i], self.W_train[self.idx_from_train[i]], self.dtree) for i,_ in enumerate(z_test)])

        _, ez_dict_test = get_rule_explanation_all(z_test_new, self.dtd, n_features=self.latent_dim, eps=eps_new)

        ex_dict_test = [decode_latent_rule_complete(x, w, e, self.dtd) for x,w,e in zip(x_test, w_test_new, ez_dict_test)]
        ex_dict_test = [simplify_rule_complete(fix_cat_rule_complete(e, self.idx_num)) for e in ex_dict_test]

        ex_dict_test = [ex if (self.dtree.predict(z_test[[i]])==y_test_bb[i] or np.any(mask_train[i])) else 
                        None for i,ex in enumerate(ex_dict_test)]

        return ex_dict_test

    def get_counterfactuals(self, x_test, y_test_bb, class_label=1, num_k=None):

        """Find counterfactual rules and training counter-examples for test rows.

        Args:
            x_test: Test feature rows; y_test_bb: black-box class labels.
            class_label: Positive class; num_k: optional sparse coefficient count.

        Returns:
            Counterfactual rule dictionaries and matching training-row arrays;
    individual list entries may be None when no counterfactual is found.
        """
        y_test_bb = (y_test_bb==class_label).astype(int)

        w_test, z_test = self.transform(x_test, num_k_sparse=num_k)
        if (w_test.shape[0]==1) and (x_test.shape[0]>1):
            w_test = np.repeat(w_test, x_test.shape[0], axis=0)

        # range indices of training set
        idx_train = np.arange(self.Z_train.shape[0])
        # boolean indices where surrogate==black-box in training set
        cond_train = self.dtree.predict(self.Z_train)==self.y_train_bb
        # boolean indices where (surrogate==black-box) but (y_train_surr!=y_test_surr) 
        mask_train = np.array([self.dtree.predict(z.reshape(1,-1))!=self.y_train_bb[cond_train] for z in z_test], dtype=bool)

        # indices of the closest training instance among the training instances with (surrogate==black-box) and (y_train_surr!=y_test_surr) for each test instance
        idx_from_train = [cdist(z.reshape(1,-1), self.Z_train[cond_train], metric='cosine')[0][mask_train[i]].argsort()[0] if np.any(mask_train[i]) else None for i,z in enumerate(z_test)]
        # indices of the closest training instance among all training instances for each test instance
        self.idx_from_train = [idx_train[cond_train][mask_train[i]][t] if np.any(mask_train[i]) else None for i,t in enumerate(idx_from_train)]

        # latent refinement
        z_test_new, _, _ = zip(*[(z_test[i], w_test[i], np.zeros_like(z_test[i])) if (self.dtree.predict(z_test[[i]])==y_test_bb[i] or ~np.any(mask_train[i])) else
                    self._latent_update(x_test[i], self.X_train[self.idx_from_train[i]], w_test[i], self.W_train[self.idx_from_train[i]], self.dtree) for i,_ in enumerate(z_test)])

        cf_idx_test, _, cx_dict_test = get_crule_explanation_all(np.array(z_test_new), x_test, y_test_bb, 
                                                                        self.X_train[cond_train], self.W_train[cond_train], 
                                                                        self.dtd, n_features=self.latent_dim)

        cx_dict_test = [cx if (self.dtree.predict(z_test[[i]])==y_test_bb[i] or np.any(mask_train[i])) else 
                        None for i,cx in enumerate(cx_dict_test)]

        cf_list_test = [np.atleast_2d(self.X_train[cond_train][cdx]) if (self.dtree.predict(z_test[[i]])==y_test_bb[i] or np.any(mask_train[i])) else 
                        None for i,cdx in enumerate(cf_idx_test)]

        return cx_dict_test, cf_list_test
