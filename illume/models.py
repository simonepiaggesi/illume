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

from utils import linear_eval, tree_eval
from expl_utils import get_rule_explanation_all, get_crule_explanation_all, decode_latent_rule_complete, inverse_transform_rule_complete, fix_cat_rule_complete
from scipy.spatial.distance import cdist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

flatten = lambda m: [item for row in m for item in row]
        
class LinearAutoEnc(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LinearAutoEnc, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.weight = nn.Parameter(nn.init.uniform_(torch.Tensor(latent_dim, input_dim), 
                                                    a=-1./np.sqrt(input_dim),b=1./np.sqrt(input_dim)))

    def encode(self, x, num_k=None):
        w = self.get_weight(num_k)
        w_norm = w/torch.norm(w, p=2, dim=-1)[:,None]
        z = torch.mm(x, torch.t(w_norm)) 
        return w_norm[None,:,:], z
    def decode(self, z):
        x_hat = torch.mm(z, torch.t(self.weight)) 
        return x_hat
    def forward(self, x, num_k_sparse=None):
        w, z = self.encode(x, num_k_sparse)
        x_hat = self.decode(z)
        return x_hat    
    def get_weight(self, num_k=None):
        if num_k == None or num_k >= self.input_dim:
            return self.weight
        else:
            indices = torch.argsort(torch.abs(self.weight))[:,-num_k:]
            mask = torch.zeros_like(self.weight)
            mask.scatter_(1, indices, 1)
            masked_weight = torch.mul(self.weight, mask)
            return masked_weight


class LocalLinearAutoEnc(nn.Module):
    def __init__(self, input_dim, latent_dim, mlp_layers):
        super(LocalLinearAutoEnc, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        enc_hidden_dims = np.logspace(np.log2(input_dim), np.log2(input_dim*latent_dim), num=mlp_layers+1, base=2).astype(int)[1:-1]
        self.encoder = MLP(input_dim, list(enc_hidden_dims)+[input_dim*latent_dim], bias=True)
        
    def encode(self, x, num_k=None):
        self.weight = torch.reshape(self.encoder(x), (-1, self.latent_dim, self.input_dim))
        w = self.get_weight(self.weight, num_k)
        w_norm = w/torch.norm(w, p=2, dim=-1)[:,:,None]
        z = torch.bmm(torch.unsqueeze(x, dim=1), torch.transpose(w_norm, 1,2)).squeeze()
        return w_norm, z
    def decode(self, z):
        x_hat = torch.bmm(torch.unsqueeze(z, dim=1), self.weight).squeeze()
        return x_hat
    def forward(self, x, num_k_sparse=None):
        w, z = self.encode(x, num_k_sparse)
        x_hat = self.decode(z)
        return x_hat    
    def get_weight(self, w_batch, num_k=None):
        if num_k == None or num_k >= self.input_dim:
            return w_batch
        else:
            indices = torch.argsort(torch.abs(w_batch))[:,:,-num_k:]
            mask_batch = torch.zeros_like(w_batch)
            mask_batch.scatter_(2, indices, 1)
            masked_weight = torch.mul(w_batch, mask_batch)
            return masked_weight
            
def compute_similarity_z(Z, sigma=1):
    D = 1 - F.cosine_similarity(Z[:, None, :], Z[None, :, :], dim=-1)
    M = torch.exp((-D**2)/(2*sigma**2))
    return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0))).transpose(0,1)

def compute_similarity_w(W, sigma=1):
    D = 1 - torch.matmul(W[:,0,:], W[:,0,:].T)
    for l in range(1, W.shape[1]):
        D += 1 - torch.matmul(W[:,l,:], W[:,l,:].T)
    D /= W.shape[1]

    M = torch.exp((-D**2)/(2*sigma**2))
    return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0))).transpose(0,1)

def compute_similarity_y(y, sigma=1):
    D = torch.cdist(y, y)
    M = torch.exp((-D**2)/(2*sigma**2))
    return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0))).transpose(0,1)

def compute_similarity_x(X, y=None, idx_cat=None, sigma=1):
    if y:
        D_class = torch.cdist(X[:,-y:], X[:,-y:])
        X = X[:, :-y]
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
    if y:
        D += D_class
    M = torch.exp((-D**2)/(2*sigma**2))
    return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0))).transpose(0,1)


def kld_loss_function_zxy(X, Z, y, idx_cat=None, sigma=1):
    similarity_KLD = torch.nn.KLDivLoss(reduction='batchmean')
    Sx = compute_similarity_x(X, None, idx_cat, sigma)
    Sz = compute_similarity_z(Z, sigma)
    loss = similarity_KLD(torch.log(Sz), Sx)
    if not torch.any(torch.isnan(y)):
        Sy = compute_similarity_y(y, sigma)
        loss += similarity_KLD(torch.log(Sz), Sy)
    return loss


def kld_loss_function_wxyz(X, W, Z, y, idx_cat=None, sigma=1):
    similarity_KLD = torch.nn.KLDivLoss(reduction='batchmean')
    Sx = compute_similarity_x(X, None, idx_cat, sigma)
    Sw = compute_similarity_w(W, sigma)
    Sz = compute_similarity_z(Z, sigma)

    loss = similarity_KLD(torch.log(Sz), Sx) + similarity_KLD(torch.log(Sw), Sz)
    if not torch.any(torch.isnan(y)):
        Sy = compute_similarity_y(y, sigma)
        loss += similarity_KLD(torch.log(Sz), Sy)
    return loss

def orth_loss_function(Q):
    loss = ((Q - torch.eye(Q.shape[1]).to(device))**2).mean(dim=[1,2]).mean()
    return loss

def corr_loss_function(Z):

    def pearson_matrix(X):
        return torch.mm(batch_norm(X, 0).T, batch_norm(X, 0))[None,:,:]/X.shape[0]

    C = pearson_matrix(Z)
    
    loss = ((torch.abs(C) - torch.eye(C.shape[1]).to(device))**2).mean(dim=[1,2]).mean()
    return loss

def rec_loss_function(X, X_hat, idx_cat_lol):
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


def compute_jacobian(x, fx):
    #from https://github.com/AmanDaVinci/SENN
    b = x.shape[0]
    n = x.shape[-1]
    m = fx.shape[-1]
    jacobians = []
    for i in range(m):
        grad_tensor = torch.zeros(b, m).to(device)
        grad_tensor[:,i] = 1
        j = torch.autograd.grad(outputs=fx, inputs=x, grad_outputs=grad_tensor,
                                create_graph=True, only_inputs=True)[0]
        jacobians.append(j.view(b,-1).unsqueeze(-1)) 
    J = torch.cat(jacobians,dim=2)
    return torch.transpose(J, 1, 2)

def batch_norm(X, axis=0):
    mean = X.mean(axis=axis, keepdim=True)
    var = X.var(axis=axis, unbiased=False, keepdim=True)
    X_norm = (X - mean) / (var + 1e-15).sqrt()
    return X_norm


class ILLUME(torch.nn.Module):
    def __init__(self, encdec='local_linear', latent_dim=2, mlp_layers=3, 
                 max_epochs=1000, early_stopping=30, learning_rate=0.001, batch_size=1024, sigma=1):
        super().__init__()

        self.encdec = encdec  #'linear' , 'local_linear'

        self.latent_dim=latent_dim
        self.mlp_layers=mlp_layers

        self.max_epochs=max_epochs
        self.early_stopping=early_stopping

        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.sigma=sigma

    def _set(self, X, y, idx_num_cat, init_path=None):

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
            if self.encdec == 'local_linear':
                self.model = LocalLinearAutoEnc(self.input_dim, self.latent_dim, self.mlp_layers).to(device)
            elif self.encdec == 'linear':
                self.model = LinearAutoEnc(self.input_dim, self.latent_dim).to(device)

        num_trainable_params = sum([p.numel() for p in self.model.parameters()])
        print('num. parameters = ' + str(num_trainable_params))

        if init_path!=None:
            assert(os.path.isfile(init_path))
            self.model.load_state_dict(torch.load(init_path, map_location=device))

        model_params = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(model_params, lr=self.learning_rate)

    def load(self, X, y, idx_num_cat, init_path):

        self._set(X, y, idx_num_cat, init_path)


    def fit(self, X, y=None, idx_num_cat=None,
            params_dict={'num_k':None, 'l_rec':1.0, 'l_kld':1.0, 'l_so':1.0, 'l_co':1.0, 'l_st':1.0}, 
            init_path=None, seed=None):

        self._set(X, y, idx_num_cat, init_path)

        if seed:
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

        self.W_train, self.Z_train = self.transform(self.X_train, num_k_sparse=num_k)
        _, self.Z_val = self.transform(self.X_val, num_k_sparse=num_k)

        if (self.W_train.shape[0]==1) and (self.X_train.shape[0]>1):
            self.W_train = np.repeat(self.W_train, self.X_train.shape[0], axis=0)

        self.y_train_bb = (np.argmax(self.y_train, axis=1)==class_label).astype(int)
        self.y_val_bb = (np.argmax(self.y_val, axis=1)==class_label).astype(int)

        print(f'Training Logistic Regression surrogate.')
        self.logreg, f1 = linear_eval(self.Z_train, self.y_train_bb, self.Z_val, self.y_val_bb, f1_average='macro')
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

        print(f'Generating feature importance for training set.')
        self.Ex_train = self._get_feature_importance_training()
        print()

        return

    def explain_dectree(self, class_label=1, num_k=None):

        self.W_train, self.Z_train = self.transform(self.X_train, num_k_sparse=num_k)
        _, self.Z_val = self.transform(self.X_val, num_k_sparse=num_k)

        if (self.W_train.shape[0]==1) and (self.X_train.shape[0]>1):
            self.W_train = np.repeat(self.W_train, self.X_train.shape[0], axis=0)

        self.y_train_bb = (np.argmax(self.y_train, axis=1)==class_label).astype(int)
        self.y_val_bb = (np.argmax(self.y_val, axis=1)==class_label).astype(int)

        print(f'Training Decision Tree surrogate.')
        self.dtree, f1 = tree_eval(self.Z_train, self.y_train_bb, self.Z_val, self.y_val_bb, f1_average='macro')
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

        print(f'Generating factual and counterfactual rules for training set.')
        self.Ex_dict_train = self._get_rules_training()
        self.Cx_dict_train, self.Cx_list_train = self._get_crules_training()
        print()

        return

    def _get_feature_importance_training(self):

        ex_train = (self.logreg.coef_[None,:,:].transpose(0,2,1)*self.W_train).sum(axis=1)

        return ex_train

    def _get_rules_training(self):

        _, ez_dict_training = get_rule_explanation_all(self.Z_train, self.dtd, n_features=self.latent_dim, get_values=False)

        ex_dict_training = [decode_latent_rule_complete(x, w, e, self.dtd) for x,w,e in zip(self.X_train, self.W_train, ez_dict_training)]
        ex_dict_training = [fix_cat_rule_complete(e, self.idx_num) for e in ex_dict_training]

        return ex_dict_training

    def _get_crules_training(self):

        cond_train = self.dtree.predict(self.Z_train)==self.y_train_bb

        cf_idx_training, _, cz_dict_training = get_crule_explanation_all(self.Z_train, self.y_train_bb, self.dtd, n_features=self.latent_dim, get_values=False)

        cx_dict_training = [[decode_latent_rule_complete(self.X_train[cond_train][cdx[i]], self.W_train[cond_train][cdx[i]], cz[i], self.dtd) 
                                 for i in range(len(cdx))] for cdx,cz in zip(cf_idx_training, cz_dict_training)]
        cx_dict_training = [[fix_cat_rule_complete(cx[i], self.idx_num) for i in range(len(cx))] for cx in cx_dict_training]

        cf_list_training = [np.atleast_2d(self.X_train[cond_train][cdx]) for cdx in cf_idx_training]

        return cx_dict_training, cf_list_training
    
    def _step(self, X_batch, Y_batch, num_k):
    
        X_batch.requires_grad_(True)
        W_batch, Z_batch = self.model.encode(X_batch, num_k)
    
        if self.encdec=='linear':
            jac_loss = torch.tensor(0., requires_grad=True) 
            kld_loss = kld_loss_function_zxy(X_batch, Z_batch, Y_batch, self.idx_cat, self.sigma)
        else:
            jac_loss = torch.mean((compute_jacobian(X_batch, Z_batch) - W_batch)**2, dim=[1,2]).mean() 
            kld_loss = kld_loss_function_wxyz(X_batch, W_batch, Z_batch, Y_batch, self.idx_cat, self.sigma)
    
        orth_loss = orth_loss_function(torch.bmm(torch.abs(W_batch), 
                                  torch.transpose(torch.abs(W_batch), 1,2)))
        corr_loss = corr_loss_function(Z_batch)
    
        X_hat = self.model.decode(Z_batch)
        rec_loss = rec_loss_function(X_batch, X_hat, [idx for idx in self.idx_num_cat if len(idx)>1])
    
        return rec_loss, kld_loss, orth_loss, corr_loss, jac_loss  
        

    def _train(self, num_k_list, lambdas, early_stop_from_epoch):

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

                    self.optimizer.zero_grad()
                    rec_loss, kld_loss, orth_loss, corr_loss, jac_loss = self._step(X_batch, Y_batch, k)
                    
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

                    self.model.zero_grad()
                    rec_loss, kld_loss, orth_loss, corr_loss, jac_loss  = self._step(X_batch, Y_batch, k)
                    
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
                    else:
                        wait += 1
                    pbar.postfix[7]["value"] = wait
                    if wait == self.early_stopping:
                        break    
                epoch += 1
                pbar.update()
            self.model.load_state_dict(torch.load(dname+'/ModelTemp.pt'))
            self.optimizer.load_state_dict(torch.load(dname+'/OptTemp.pt'))

        return epoch_train_losses, epoch_val_losses

    def transform(self, X, num_k_sparse=None):
        with torch.no_grad():
            self.model.eval()
            W, Z = self.model.encode(torch.tensor(X).float().to(device), num_k_sparse)
        return W.cpu().detach().numpy(), Z.cpu().detach().numpy() 

    def inverse_transform(self, Z):
        with torch.no_grad():
            self.model.eval()
            X = self.model.decode(torch.tensor(Z).float().to(device))
            X = torch.cat([ (X[:,idx]==X[:,idx].max(dim=1).values[:,None]).float() if len(idx)>1 else X[:,idx] for idx in self.idx_num_cat], axis=1)
        return X.cpu().detach().numpy() 

    def get_feature_importance(self, x_test, y_test_bb, class_label=1, num_k=None):

        y_test_bb = (y_test_bb==class_label).astype(int)

        w_test, z_test = self.transform(x_test, num_k_sparse=num_k)
        if (w_test.shape[0]==1) and (x_test.shape[0]>1):
            w_test = np.repeat(w_test, x_test.shape[0], axis=0)

        idx_train = np.arange(self.Z_train.shape[0])
        cond_train = self.logreg.predict(self.Z_train)==self.y_train_bb
        #mask_train = np.array([np.logical_and(self.logreg.predict(z.reshape(1,-1))==self.y_train_bb[cond_train], cond_train) for z in z_test], dtype=bool)
        mask_train = np.array([self.logreg.predict(z.reshape(1,-1))==self.y_train_bb[cond_train] for z in z_test], dtype=bool)

        idx_from_train = [cdist(z.reshape(1,-1), self.Z_train[cond_train], metric='cosine')[0][mask_train[i]].argsort()[0] if np.any(mask_train[i]) else None for i,z in enumerate(z_test)]
        idx_from_train = [idx_train[cond_train][mask_train[i]][t] if np.any(mask_train[i]) else None for i,t in enumerate(idx_from_train)]

        ex_test = (self.logreg.coef_[None,:,:].transpose(0,2,1)*w_test).sum(axis=1)

        ex_test = [ex if self.logreg.predict(z_test[[i]])==y_test_bb[i] else 
                         (self.Ex_train[idx_from_train[i]] if np.any(mask_train[i]) else None) for i,ex in enumerate(ex_test)]

        return ex_test

    def get_decision_rules(self, x_test, y_test_bb, class_label=1, num_k=None):

        y_test_bb = (y_test_bb==class_label).astype(int)

        w_test, z_test = self.transform(x_test, num_k_sparse=num_k)
        if (w_test.shape[0]==1) and (x_test.shape[0]>1):
            w_test = np.repeat(w_test, x_test.shape[0], axis=0)

        idx_train = np.arange(self.Z_train.shape[0])
        cond_train = self.dtree.predict(self.Z_train)==self.y_train_bb
        mask_train = np.array([self.dtree.predict(z.reshape(1,-1))==self.y_train_bb[cond_train] for z in z_test], dtype=bool)

        idx_from_train = [cdist(z.reshape(1,-1), self.Z_train[cond_train], metric='cosine')[0][mask_train[i]].argsort()[0] if np.any(mask_train[i]) else None for i,z in enumerate(z_test)]
        idx_from_train = [idx_train[cond_train][mask_train[i]][t] if np.any(mask_train[i]) else None for i,t in enumerate(idx_from_train)]

        _, ez_dict_test = get_rule_explanation_all(z_test, self.dtd, n_features=self.latent_dim, get_values=False)

        ex_dict_test = [decode_latent_rule_complete(x, w, e, self.dtd) for x,w,e in zip(x_test, w_test, ez_dict_test)]
        ex_dict_test = [fix_cat_rule_complete(e, self.idx_num) for e in ex_dict_test]

        ex_dict_test = [ex if self.dtree.predict(z_test[[i]])==y_test_bb[i] else 
                         (self.Ex_dict_train[idx_from_train[i]] if np.any(mask_train[i]) else None) for i,ex in enumerate(ex_dict_test)]

        return ex_dict_test


    def get_counterfactuals(self, x_test, y_test_bb, class_label=1, num_k=None):

        y_test_bb = (y_test_bb==class_label).astype(int)

        w_test, z_test = self.transform(x_test, num_k_sparse=num_k)
        if (w_test.shape[0]==1) and (x_test.shape[0]>1):
            w_test = np.repeat(w_test, x_test.shape[0], axis=0)

        idx_train = np.arange(self.Z_train.shape[0])
        cond_train = self.dtree.predict(self.Z_train)==self.y_train_bb
        mask_train = np.array([self.dtree.predict(z.reshape(1,-1))==self.y_train_bb[cond_train] for z in z_test], dtype=bool)

        idx_from_train = [cdist(z.reshape(1,-1), self.Z_train[cond_train], metric='cosine')[0][mask_train[i]].argsort()[0] if np.any(mask_train[i]) else None for i,z in enumerate(z_test)]
        idx_from_train = [idx_train[cond_train][mask_train[i]][t] if np.any(mask_train[i]) else None for i,t in enumerate(idx_from_train)]

        cf_idx_test, _, cz_dict_test = get_crule_explanation_all(z_test, y_test_bb, self.dtd, n_features=self.latent_dim, get_values=False)

        cx_dict_test = [[decode_latent_rule_complete(self.X_train[cond_train][cdx[i]], self.W_train[cond_train][cdx[i]], cz[i], self.dtd) 
                                 for i in range(len(cdx))] for cdx,cz in zip(cf_idx_test, cz_dict_test)]
        cx_dict_test = [[fix_cat_rule_complete(cx[i], self.idx_num) for i in range(len(cx))] for cx in cx_dict_test]

        cx_dict_test = [cx if self.dtree.predict(z_test[[i]])==y_test_bb[i] else 
                         (self.Cx_dict_train[idx_from_train[i]] if np.any(mask_train[i]) else None) for i,cx in enumerate(cx_dict_test)]

        cf_list_test = [np.atleast_2d(self.X_train[cond_train][cdx]) if self.dtree.predict(z_test[[i]])==y_test_bb[i] else 
                         (self.Cx_list_train[idx_from_train[i]] if np.any(mask_train[i]) else None) for i,cdx in enumerate(cf_idx_test)]

        return cx_dict_test, cf_list_test


