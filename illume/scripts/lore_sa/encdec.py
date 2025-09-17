from abc import abstractmethod
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.spatial.distance import cdist


#EncDec is an abstract class
#It is implemented by different classes, each of which must implements the functions: enc, dec, enc_fit_transform
#the idea is that the user sends the complete record and here only the categorical variables are handled
class EncDec():
    def __init__(self, dataset=None, class_name = None):
        self.dataset = dataset
        self.class_name = class_name
        self.encdec = None
        self.features = None
        self.cate_features_names = None
        self.cate_features_idx = None

    @abstractmethod
    def enc(self, x, kwargs=None):
        return

    @abstractmethod
    def dec(self, x, kwargs=None):
        return

extend: TargetEncoder
class MyTargetEnc(EncDec):
    def __init__(self, dataset, class_name):
        super(MyTargetEnc, self).__init__(dataset, class_name)
        self.dataset_enc = None
        self.cate_map = dict()
        self.inverse_cate_map = list()
        self.dataset_enc_complete = None

    def retrieve_values(self, index, interval, op):
        inverse_dataset = self.dec(self.dataset_enc_complete)
        feature_values = self.dataset_enc_complete[:, index]
        if len(interval) == 1:
            if op == '<':
                indexes = [feature_values.tolist().index(i) for i in feature_values if i <= interval[0]]
            else:
                indexes = [feature_values.tolist().index(i) for i in feature_values if i > interval[0]]
        else:
            index_values_min = [feature_values.tolist().index(i) for i in feature_values if i > interval[0]]
            index_values_max = [feature_values.tolist().index(i) for i in feature_values if i <= interval[1]]
            indexes = list(set(index_values_min) & set(index_values_max))
        res = set(inverse_dataset[indexes,index])
        return list(res)


    #given a dataset and the class name, this function applies target encoder on the categorical variables
    #self.encdec is the trained encoder
    #self.dataset_enc is the encoded dataset
    def enc_fit_transform(self, dataset=None, class_name=None, kwargs=None):

        if self.dataset is None:
            self.dataset = dataset
        if self.class_name is None:
            self.class_name = class_name

        self.features = [c for c in self.dataset.columns if c not in [self.class_name]]
        self.cont_features_names = list(self.dataset[self.features]._get_numeric_data().columns)
        self.cate_features_names = [c for c in self.dataset.columns if c not in self.cont_features_names and c != self.class_name]
        self.cate_features_idx = [self.features.index(f) for f in self.cate_features_names]
        self.cont_features_idx = [self.features.index(f) for f in self.cont_features_names]
        #print(self.cont_features_names, ' cont feature name')
        self.encdec = TargetEncoder(return_df=False)
        dataset_values = self.dataset[self.features].values
        y = self.dataset[self.class_name].values
        self.dataset_enc = self.encdec.fit_transform(dataset_values[:, self.cate_features_idx], y)
        self.dataset_enc_complete = np.zeros((self.dataset_enc.shape[0],len(self.features)))
        #print('index ', self.cate_features_idx, self.cont_features_idx)
        for p in range(self.dataset_enc.shape[0]):
            for i in range(0, len(self.cate_features_idx)):
                self.dataset_enc_complete[p][self.cate_features_idx[i]] = self.dataset_enc[p][i]
            for j in self.cont_features_idx:
                self.dataset_enc_complete[p][j] = dataset_values[p][j]
        for i, idx in enumerate(self.cate_features_idx):
            cate_map_i = dict()
            inverse_cate_map_i = dict()
            values = np.unique(dataset_values[:, idx])
            for v1, v2 in zip(dataset_values[:, idx], self.dataset_enc[:, i]):
                cate_map_i[v1] = v2
                inverse_cate_map_i[v2] = v1
                if len(cate_map_i) == len(values):
                    break
            self.cate_map[idx] = cate_map_i
            self.inverse_cate_map.append(inverse_cate_map_i)
        print('cate map ', self.cate_map)
        return self.dataset_enc_complete

    #todo fix get cate map
    def get_cate_map(self, i, value):
        found = False
        if i in self.cate_map.keys():
            print('sono in cate map')
            try:
                print(self.cate_map[i][value])
                return self.cate_map[i][value]
            except:
                return value
            '''for key, v in self.cate_map[i].items():
                if v == value:
                    found = True
                    break
            if found == True:
                return key
            else:
                key_list = list(self.cate_map[i].keys())
                val_list = list(self.cate_map[i].values())
                array = np.asarray(val_list)
                idx = (np.abs(array - value)).argmin()
                ind = val_list.index(array[idx])
                return key_list[ind]'''
        else:
            return value

    def enc(self, x, y, kwargs=None):
        if len(x.shape) == 1:
            x_cat = x[self.cate_features_idx]
            x_cat = x_cat.reshape(1, -1)
            x = x.reshape(1,-1)
        else:
            x_cat = x[:, self.cate_features_idx]
        if len(y.shape) == 1:
            y = y.reshape(1, -1)
        #print(x_cat, y, ' x cat y ')
        x_cat_enc = self.encdec.transform(x_cat, y)
        #print(x.shape[0], x.shape[1], 'shapes')
        x_res = np.zeros((x.shape[0], x.shape[1]))
        for p in range(x.shape[0]):
            for i in range(0, len(self.cate_features_idx)):
                x_res[p][self.cate_features_idx[i]] = x_cat_enc[p][i]
            for j in self.cont_features_idx:
                x_res[p][j] = x[p][j]
        return x_res


    def dec(self, x, kwargs=None):
        return self.inverse_transform(x)

    def inverse_transform(self, X):
        if len(X.shape) == 1:
            X_cat = X[self.cate_features_idx]
            X = X.reshape(1, -1)
            X_cat = X_cat.reshape(1, -1)
        else:
            X_cat = X[:, self.cate_features_idx]
        X_cat = np.array(X_cat, dtype=float)
        X_new = list()
        for i in range(X_cat.shape[1]):
            values = np.array(list(self.inverse_cate_map[i].keys()))
            keys = np.array(list(self.inverse_cate_map[i].values()))
            closest_val = np.argmin(cdist(values.reshape(-1, 1), X_cat[:, i].reshape(-1, 1)), axis=0)
            X_new.append(np.array([keys[j] for j in closest_val]))
        X_new = np.array(X_new)
        #print(X_new)
        X_new = X_new.T
        x_res = np.empty((X.shape[0], X.shape[1]), dtype=object)
        for p in range(X.shape[0]):
            for i in range(0, len(self.cate_features_idx)):
                x_res[p][self.cate_features_idx[i]] = X_new[p][i]
            for j in self.cont_features_idx:
                x_res[p][j] = X[p][j]
        #print(x_res.shape)
        return x_res

extend: OneHotEncoder
class OneHotEnc(EncDec):
    def __init__(self, dataset, class_name):
        super(OneHotEnc, self).__init__(dataset, class_name)
        self.dataset_enc = None


    #select the categorical variable
    #apply onehot encoding on them
    #self.encdec is the encoder already fitted
    #self.dataset_enc is the dataset encoded
    def enc_fit_transform(self, kwargs=None):
        self.features = [c for c in self.dataset.columns if c not in [self.class_name]]
        self.cont_features_names = list(self.dataset[self.features]._get_numeric_data().columns)
        self.cate_features_names = [c for c in self.dataset.columns if
                                    c not in self.cont_features_names and c != self.class_name]
        self.cate_features_idx = [self.features.index(f) for f in self.cate_features_names]
        self.cont_features_idx = [self.features.index(f) for f in self.cont_features_names]
        print('cate features idx ', self.cate_features_idx)
        dataset_values = self.dataset[self.features].values
        self.encdec = OneHotEncoder(handle_unknown='ignore')
        self.dataset_enc = self.encdec.fit_transform(dataset_values[:, self.cate_features_idx]).toarray()

        self.onehot_feature_idx = list()
        self.new_cont_idx = list()
        for f in self.cate_features_idx:
            uniques = len(np.unique(dataset_values[:, f]))
            for u in range(0,uniques):
                self.onehot_feature_idx.append(f+u)
        npiu = i = j = 0
        while j < len(self.cont_features_idx):
            if self.cont_features_idx[j] < self.cate_features_idx[i]:
                self.new_cont_idx.append(self.cont_features_idx[j] + npiu - 1)
            elif self.cont_features_idx[j] > self.cate_features_idx[i]:
                npiu += len(np.unique(dataset_values[:, self.cate_features_idx[i]]))
                self.new_cont_idx.append(self.cont_features_idx[j] + npiu - 1)
                i += 1
            j += 1
        n_feat_tot = self.dataset_enc.shape[1] + len(self.cont_features_idx)
        self.dataset_enc_complete = np.zeros((self.dataset_enc.shape[0], n_feat_tot))
        for p in range(self.dataset_enc.shape[0]):
            for i in range(0, len(self.onehot_feature_idx)):
                self.dataset_enc_complete[p][self.onehot_feature_idx[i]] = self.dataset_enc[p][i]
            for j in range(0, len(self.new_cont_idx)):
                self.dataset_enc_complete[p][self.new_cont_idx[j]] = dataset_values[p][self.cont_features_idx[j]]
        #print(len(self.onehot_feature_idx))
        #print(len(self.cate_features_idx))
        return self.dataset_enc_complete

    def enc(self, x, y, kwargs=None):
        if len(x.shape) == 1:
            x_cat = x[self.cate_features_idx]
            x_cat = x_cat.reshape(1, -1)
            x = x.reshape(1,-1)
        else:
            x_cat = x[:, self.cate_features_idx]
        x_cat_enc = self.encdec.transform(x_cat).toarray()
        n_feat_tot = self.dataset_enc.shape[1] + len(self.cont_features_idx)
        x_res = np.zeros((x.shape[0], n_feat_tot))
        for p in range(x_res.shape[0]):
            for i in range(0, len(self.onehot_feature_idx)):
                x_res[p][self.onehot_feature_idx[i]] = x_cat_enc[p][i]
            for j in range(0, len(self.new_cont_idx)):
                x_res[p][self.new_cont_idx[j]] = x[p][self.cont_features_idx[j]]

        return x_res


    def dec(self, x, kwargs=None):
        if len(x.shape) == 1:
            x_cat = x[self.onehot_feature_idx]
            x = x.reshape(1, -1)
            x_cat = x_cat.reshape(1, -1)
        else:
            x_cat = x[:, self.onehot_feature_idx]
            #print(x_cat)
        X_new =  self.encdec.inverse_transform(x_cat)
        x_res = np.empty((x.shape[0], len(self.features)), dtype=object)
        for p in range(x.shape[0]):
            for i in range(0, len(self.cate_features_idx)):
                x_res[p][self.cate_features_idx[i]] = X_new[p][i]
            for j in self.cont_features_idx:
                x_res[p][j] = x[p][j]
        #print(x_res.shape)
        return x_res


