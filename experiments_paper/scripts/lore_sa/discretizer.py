from abc import abstractmethod
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score


class Discretizer():
    def __init__(self):

        self.discretizer = None

    @abstractmethod
    def fit(self, X, y, kwargs=None):
        return

    @abstractmethod
    def transform(self, X, kwargs=None):
        return




class RMEPDiscretizer():
    def __init__(self, to_discretize=None, proto_fn=None):
        super(RMEPDiscretizer, self).__init__()
        self.to_discretize = to_discretize
        self.nbr_features = None
        self.vals = None

        if proto_fn is None:
            self.proto_fn = np.mean

    def array_entropy(self, y):
        unique, unique_counts = np.unique(y, return_counts=True)
        probabilities = unique_counts / len(y)
        return -sum(probabilities * np.log2(probabilities))

    def recursive_partition(self, values):
        part_idx_list = []
        values = np.asarray(values)

        # entropy along_subsection of values:
        def ent(start_idx, end_idx):
            return self.array_entropy(values[start_idx:end_idx])

        # k values along subsection of values:
        def k_fn(start_idx, end_idx):
            return len(np.unique(values[start_idx:end_idx]))

        def partition_subset(start_idx, end_idx):
            # n: |values| being split
            # start_idx beginning of values
            # split_idx place to try a split
            # end_idx end of values

            n = end_idx - start_idx
            partial_entropy = [(ent(start_idx, candidate_split_idx) * (candidate_split_idx - start_idx) +
                                ent(candidate_split_idx, end_idx) * (end_idx - candidate_split_idx)) / n
                               for candidate_split_idx in range(start_idx + 1, end_idx)]
            smallest_entropy_idx = np.argmin(partial_entropy)
            split_ent = partial_entropy[smallest_entropy_idx]
            split_idx = start_idx + smallest_entropy_idx + 1
            gain = ent(start_idx, end_idx) - split_ent
            delta = (np.log2(3. ** k_fn(start_idx, end_idx) - 2) -
                     (k_fn(start_idx, end_idx) * ent(start_idx, end_idx) -
                      k_fn(start_idx, split_idx) * ent(start_idx, split_idx) -
                      k_fn(split_idx, end_idx) * ent(split_idx, end_idx)))
            thresh = (np.log2(n - 1) + delta) / n

            if gain > thresh:
                part_idx_list.append(split_idx)
                if split_idx - start_idx > 1:
                    partition_subset(start_idx, split_idx)

                if end_idx - split_idx > 1:
                    partition_subset(split_idx, end_idx)

            return

        partition_subset(0, len(values))
        return part_idx_list

    def fit(self, X, y):

        if self.to_discretize is None:
            self.nbr_features = X.shape[1]
            self.to_discretize = np.arange(self.nbr_features)
        else:
            self.nbr_features = len(self.to_discretize)

        self.vals = dict()
        for i in self.to_discretize:
            sorted_values = np.array(sorted(zip(X[:, i], y), key=lambda x: x[0]))
            part_idx_list = self.recursive_partition(sorted_values[:, 1])
            thresholds = np.unique(np.array(sorted_values[part_idx_list, 0]))
            dt = DecisionTreeClassifier(criterion='entropy',
                                        max_leaf_nodes=max(2, len(thresholds) + 1))
            X_i = np.reshape(X[:, i], (-1, 1))
            dt.fit(X_i, y)
            ffeat = dt.tree_.feature
            vals_i = np.sort(dt.tree_.threshold[np.where(ffeat != -2)])
            self.vals[i] = vals_i

    def transform(self, X):
        X_ = np.copy(X)
        for i in self.to_discretize:
            cuts = list(self.vals[i])
            for l_b, u_b in zip([-np.inf] + cuts, cuts + [np.inf]):
                idx_respecting_cond = np.where((X_[:, i] > l_b) & (X_[:, i] <= u_b))
                #if X_[idx_respecting_cond, i] == 0.0:
                    #print('Error division by zero')
                new_val = np.mean(X_[idx_respecting_cond, i])
                X_[idx_respecting_cond, i] = new_val
        return X_


