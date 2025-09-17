import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

class Surrogate():
    def __init__(self, kind = None, preprocessing=None):
        #decision tree, supertree
        self.kind = kind
        #kind of preprocessing to apply
        self.preprocessing = preprocessing

class DecTree(Surrogate):

    def __init__(self, kind = None, preprocessing=None):
        super(DecTree, self).__init__(kind, preprocessing)


    def learn_local_decision_tree(self, Z, Yb, weights, class_values, multi_label=False, one_vs_rest=False, cv=3,
                                  prune_tree=False):
        dt = DecisionTreeClassifier()
        if prune_tree:
            param_list = {'min_samples_split': [ 0.01, 0.05, 0.1, 0.2, 3, 2],
                          'min_samples_leaf': [0.001, 0.01, 0.05, 0.1,  2, 4],
                          'splitter' : ['best', 'random'],
                          'max_depth': [None, 2, 10, 12, 16, 20, 30],
                          'criterion': ['entropy', 'gini'],
                          'max_features': [0.2, 1, 5, 'auto', 'sqrt', 'log2']
                          }

            if not multi_label or (multi_label and one_vs_rest):
                if len(class_values) == 2 or (multi_label and one_vs_rest):
                    scoring = 'precision'
                else:
                    scoring = 'precision_macro'
            else:
                scoring = 'precision_samples'

            dt_search = HalvingGridSearchCV(dt, param_grid=param_list, scoring=scoring, cv=cv, n_jobs=-1)
            # print(datetime.datetime.now())
            dt_search.fit(Z, Yb, sample_weight=weights)
            # print(datetime.datetime.now())
            dt = dt_search.best_estimator_
            self.prune_duplicate_leaves(dt)
        else:
            dt.fit(Z, Yb, sample_weight=weights)

        return dt


    def is_leaf(self, inner_tree, index):
        # Check whether node is leaf node
        return (inner_tree.children_left[index] == TREE_LEAF and
                inner_tree.children_right[index] == TREE_LEAF)


    def prune_index(self, inner_tree, decisions, index=0):
        # Start pruning from the bottom - if we start from the top, we might miss
        # nodes that become leaves during pruning.
        # Do not use this directly - use prune_duplicate_leaves instead.
        if not self.is_leaf(inner_tree, inner_tree.children_left[index]):
            self.prune_index(inner_tree, decisions, inner_tree.children_left[index])
        if not self.is_leaf(inner_tree, inner_tree.children_right[index]):
            self.prune_index(inner_tree, decisions, inner_tree.children_right[index])

        # Prune children if both children are leaves now and make the same decision:
        if (self.is_leaf(inner_tree, inner_tree.children_left[index]) and
            self.is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and
            (decisions[index] == decisions[inner_tree.children_right[index]])):
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
            # print("Pruned {}".format(index))


    def prune_duplicate_leaves(self, dt):
        # Remove leaves if both
        decisions = dt.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
        self.prune_index(dt.tree_, decisions)




class SuperTree(Surrogate):


    def __init__(self, kind = None, preprocessing=None):
        super(SuperTree, self).__init__(kind, preprocessing)

    class Node():
        def __init__(self, feat_num=None, weights=None, thresh=None, labels=None, is_leaf=False, impurity=1, **kwargs):
            self.feat = feat_num
            self.thresh = thresh
            self.is_leaf = is_leaf
            self._weights = weights
            self._left_child = kwargs.get('left_child', None)
            self._right_child = kwargs.get('right_child', None)
            self.children = kwargs.get('children', None)
            self.impurity = impurity
            self.labels = labels
            if not (weights is None):
                self._features_involved = np.arange(weights.shape[0] - 1)

            else:
                if not self.children:
                    self.children = []
                    if self._left_child:
                        self.children.append(self._left_child)
                    if self._right_child:
                        self.children.append(self._right_child)

        def predict(self, X):
            def predict_datum(node, x):
                if node.is_leaf:
                    return np.argmax(node.labels)
                else:
                    if not node.feat is None:
                        Xf = node.feat
                        if node.thresh <= x[Xf] and node._left_child:
                            next_node = node._left_child
                        elif node._right_child:
                            next_node = node._right_child
                        else:
                            return np.argmax(node.labels)
                        #next_node = np.argmin(node.thresh <= x[Xf])
                    else:
                        print('ERRORE')
                        bias = node._weights[-1]
                        next_node = node._left_child
                    return predict_datum(next_node, x)

            return np.array([predict_datum(self, el) for el in X])

        def print_tree(self, Tree, level=0, feature_names=None):
            if Tree.is_leaf:
                print("|\t" * level + "|---", np.argmax(Tree.labels), Tree.labels)
            else:
                if Tree._weights is None:
                    if not Tree._left_child is None:
                        print("|\t" * level + "|---", "X", Tree.feat, "<=", Tree.thresh)
                        self.print_tree(Tree._left_child, level + 1)
                    if not Tree._right_child is None:
                        print("|\t" * level + "|---", "X", Tree.feat, ">", Tree.thresh)
                        self.print_tree(Tree._right_child, level + 1)
                else:

                    if feature_names is None:
                        feature_names = ["X_" + str(i) for i in Tree._features_involved]
                    rule = ''
                    for i, f in enumerate(feature_names):
                        rule += (("+" if i != 0 and np.sign(Tree._weights[i]) > 0 else "") + \
                                 str(np.round(Tree._weights[i], 4)) + " " + f + " ") \
                            if np.abs(Tree._weights[i]) > 1e-2 else ""
                    left_rule = rule + "<=" + str(np.round(Tree._weights[-1], 4))
                    right_rule = rule + "> " + str(np.round(Tree._weights[-1], 4))
                    if not Tree._left_child is None:
                        print("|\t" * level + "/--", left_rule)
                        self.print_tree(Tree._left_child, level + 1)
                    if not Tree._right_child is None:
                        print("|\t" * level + "/--", right_rule)
                        self.print_tree(Tree._right_child, level + 1)


    def rec_buildTree(self, DT: DecisionTreeClassifier, feature_used):
        nodes = DT.tree_.__getstate__()['nodes']
        values = DT.tree_.__getstate__()['values']
        root = nodes[0]

        def createNode(node_idx):
            # a leaf node
            line = nodes[node_idx]
            if line[0] == -1:
                return self.Node(feat_num=None, thresh=None, labels=values[node_idx][0], is_leaf=True)
            else:
                LC = createNode(line[0])
                RC = createNode(line[1])

                node = self.Node(feat_num=feature_used[line[2]], thresh=line[3], labels=values[node_idx], is_leaf=False,
                            left_child=LC, right_child=RC)
                return node

        return createNode(0)

    def pruneRedundant(self, BigTree):
        if BigTree.is_leaf:
            return BigTree
        l_c = self.pruneRedundant(BigTree._left_child)
        r_c = self.pruneRedundant(BigTree._right_child)

        if l_c.is_leaf and r_c.is_leaf:
            if np.argmax(l_c.labels) == np.argmax(r_c.labels):
                l_c.labels += r_c.labels
                return l_c
        BigTree._left_child = l_c
        BigTree._right_child = r_c
        return BigTree

    def prune_tree(self, BigTree: Node, value, Xf):
        if BigTree is None:
            return None
        if BigTree.is_leaf:
            return BigTree

        l_c = self.prune_tree(BigTree._left_child, value, Xf)
        r_c = self.prune_tree(BigTree._right_child, value, Xf)
        if l_c.is_leaf and r_c.is_leaf:
            if np.argmax(l_c.labels) == np.argmax(r_c.labels):
                l_c.labels += r_c.labels
                return l_c
        if l_c is None:
            return r_c
        if r_c is None:
            return l_c

        if BigTree.feat != Xf:  
            BigTree._left_child = l_c
            BigTree._right_child = r_c
            return BigTree
        

        if BigTree.thresh < value:
            BigTree._left_child = l_c
            BigTree._right_child = r_c
            return BigTree
        if BigTree.thresh >= value:
            return l_c

    class SuperNode():
        def __init__(self, feat_num=None, intervals=None, weights=None, labels=None, children=None, is_leaf=False, level = 0):
            self.feat = feat_num
            self.is_leaf = is_leaf
            self._weights = weights
            self.children = children
            self.labels = labels
            self.intervals = intervals
            self.level = level
            self._features_involved = None  
            self.impurity = None
            self.importance = None

        def predict(self, X):
            def predict_datum(node, x):
                if node.is_leaf:
                    return np.argmax(node.labels)
                else:
                    if not node.feat is None:
                        Xf = node.feat
                        next_node = np.argmin(node.intervals <= x[Xf])
                    else:
                        bias = node._weights[-1]
                        next_node = 0 if (
                                    x[node._features_involved].dot(np.array(node._weights[:-1]).T) - node._weights[
                                -1] <= 0) else 1
                    return predict_datum(node.children[next_node], x)

            return np.array([predict_datum(self, el) for el in X])

        def decision_path_indices(self, X):
            def decision_datum(node, x, indices):
                indices.append(id(node))
                if node.is_leaf:
                    return indices
                else:
                    if not node.feat is None:
                        Xf = node.feat
                        next_node = np.argmin(node.intervals <= x[Xf])
                    else:
                        bias = node._weights[-1]
                        next_node = 0 if (
                                    x[node._features_involved].dot(np.array(node._weights[:-1]).T) - node._weights[
                                -1] <= 0) else 1
                    return decision_datum(node.children[next_node], x, indices)

            return np.array([decision_datum(self, el, list()) for el in X])


        def set_impurity(self, dt):

            def set_impurity_node(node):
                if node.is_leaf:
                    p = 0
                    tot = sum(node.labels)
                    temp = list()
                    for u in node.labels:
                        temp.append(pow((u /float(tot)), 2))
                    for t in temp:
                        if p == 0:
                            p = t
                        else:
                            p = p - t
                    node.impurity = 1 - p
                    return (1 - p, len(node.labels))
                else:
                    if not node.feat is None:
                        fi_list = list()
                        for i in range(0, len(node.intervals)):
                            fi_list.append(set_impurity_node(node.children[i]))
                        tot_node = sum([pair[1] for pair in fi_list])
                        tot = 0
                        for f in fi_list:
                            tot = tot + (f[1]/float(tot_node) * f[0])
                        node.impurity = tot
                        return (tot, tot_node)

            return set_impurity_node(dt)

        def calculate_features_importance(self, tot):

            def calculate_feature_importance_node(node):
                if node.is_leaf:
                    node.importance = sum(node.labels) / float(tot) * node.impurity
                    return sum(node.labels)
                else:
                    labels = list()
                    for c in node.children:
                        r = calculate_feature_importance_node(c)
                        if r is not None:
                            labels.append(r)
                        else:
                            print('problema ', r, c.impurity)
                    part = 0
                    for c in node.children:
                        if part == 0:
                            part = c.importance
                        else:
                            part = part - c.importance
                    node.importance = sum(labels) / float(tot) * node.impurity - part
                    return sum(labels)
            calculate_feature_importance_node(self)

        def calculate_all_importances(self):
            features = dict()
            def traverse_tree(node, features):
                if not node.is_leaf:
                    for c in node.children:
                        res = traverse_tree(c, features)
                        features = {**features, **res}

                if node.feat not in features.keys() and not (node.feat is None):
                    features[node.feat] = node.importance
                elif node.feat in features.keys() and not (node.feat is None):
                    features[node.feat] = features[node.feat] + node.importance
                return features

            features = traverse_tree(self, features)
            tot = sum(features.values())
            for f in features.keys():
                features[f] = features[f]/float(tot)
            tot = sum(features.values())
            #per normalizzare
            for f in features.keys():
                features[f] = features[f]/float(tot)
            return features

        def calculate_fi_path(self, x):
            features = dict()
            def calculate_fi_node(node, x, features):
                if not node.feat is None:
                    Xf = node.feat
                    if node.feat not in features.keys():
                        features[Xf] = node.importance
                    else:
                        features[Xf] = features[Xf] + node.importance
                    next_node = np.argmin(node.intervals <= x[Xf])
                else:
                    if node.is_leaf:
                        return features
                    else:
                        next_node = 0 if (x[node._features_involved].dot(np.array(node._weights[:-1]).T) - node._weights[-1] <= 0) else 1
                res = calculate_fi_node(node.children[next_node], x, features)
                features = {**features, **res}
                return features

            features = calculate_fi_node(self, x, features)
            tot = sum(features.values())
            for f in features.keys():
                features[f] = features[f] / float(tot)
            tot = sum(features.values())
            # per normalizzare
            for f in features.keys():
                features[f] = features[f] / float(tot)
            return features

        def apply(self, X):
            def apply_datum(node, x):
                if node.is_leaf:
                    return id(node)
                else:
                    if not node.feat is None:
                        Xf = node.feat
                        next_node = np.argmin(node.intervals <= x[Xf])
                    else:
                        next_node = 0 if (
                                    x[node._features_involved].dot(np.array(node._weights[:-1]).T) - node._weights[
                                -1] <= 0) else 1
                    return apply_datum(node.children[next_node], x)
            idx = list()
            for el in X:
                idx.append(apply_datum(self, el))
            return idx

        def print_superTree(self, SuperTree, level=0, feature_names=None):

            if SuperTree.is_leaf:
                print("|\t" * SuperTree.level + "|---", "class:", np.argmax(SuperTree.labels))
            else:
                if SuperTree._weights is None:
                    print("\t" * SuperTree.level, "level", SuperTree.level, ",", len(SuperTree.children), "childs", SuperTree.feat)
                    feature_names = ["X_" + str(k) for k in range(SuperTree.feat + 1)]
                    for c_nr, i in enumerate(SuperTree.intervals):
                        print("|\t" * level + "|---", feature_names[SuperTree.feat], "<=", i)
                        self.print_superTree(SuperTree.children[c_nr], SuperTree.level + 1)
                else:
                    if feature_names is None:
                        feature_names = ["X_" + str(i) for i in SuperTree._features_involved]
                    rule = ''
                    for i, f in enumerate(feature_names):
                        rule += (("+" if i != 0 and np.sign(SuperTree._weights[i]) > 0 else "") + \
                                 str(np.round(SuperTree._weights[i], 4)) + " " + f + " ") \
                            if np.abs(SuperTree._weights[i]) > 1e-2 else ""
                    left_rule = rule + "<=" + str(np.round(SuperTree._weights[-1], 4))
                    right_rule = rule + "> " + str(np.round(SuperTree._weights[-1], 4))

                    # print left child first
                    print("|\t" * SuperTree.level + "/---", left_rule)
                    self.print_superTree(SuperTree.children[0], SuperTree.level + 1)
                    # than right
                    print("|\t" * SuperTree.level + "/---", right_rule)
                    self.print_superTree(SuperTree.children[1], SuperTree.level + 1)



    def check_size(self, node: SuperNode):
        if node.is_leaf:
            return np.array([0, 1, 1])  
        else:
            res = np.array([self.check_size(c) for c in node.children])
            return np.array([1 + np.sum(res[:, 0]), 1 + np.max(res[:, 1]), np.sum(res[:, 2])])

    def complexityDecisionTree(self, DT):
        nodes = DT.tree_.__getstate__()['nodes']
        values = DT.tree_.__getstate__()['values']
        root = nodes[0]
        path_lenghts = []

        def complexDT(node_idx, level):
            # a leaf node
            line = nodes[node_idx]
            if line[0] == -1:
                path_lenghts.append(level + 1)
                return 0
            else:
                LC = complexDT(line[0], level + 1)
                RC = complexDT(line[1], level + 1)
                return 1 + LC + RC  # leaves

        res = complexDT(0, 0)
        return np.max(path_lenghts), res, len(path_lenghts), np.mean(path_lenghts), 1

    def complexitiSuperTree(self, node: Node):
        path_lenghts = []
        averages = []

        def complexSuper(radix, level=0, n_feat_used=0):
            if radix.is_leaf:
                path_lenghts.append(level)
                return 0
            else:

                if radix._weights is None:
                    my_feats = 1
                else:
                    my_feats = len(set(radix._features_involved))
                res = np.array([complexSuper(c, level + 1, n_feat_used + my_feats) for c in radix.children])
                return 1 + np.sum(res)

        res = complexSuper(node)
        return np.max(path_lenghts), res, len(path_lenghts), np.mean(path_lenghts), np.mean(averages)

    def is_leaf(self, inner_tree, index):
        # Check whether node is leaf node
        return (inner_tree.children_left[index] == TREE_LEAF and
                inner_tree.children_right[index] == TREE_LEAF)

    def prune_index(self, inner_tree, decisions, index=0):
        # Start pruning from the bottom - if we start from the top, we might miss
        # nodes that become leaves during pruning.
        # Do not use this directly - use prune_duplicate_leaves instead.
        if not self.is_leaf(inner_tree, inner_tree.children_left[index]):
            
            self.prune_index(inner_tree, decisions, inner_tree.children_left[index])
        if not self.is_leaf(inner_tree, inner_tree.children_right[index]):
             self.prune_index(inner_tree, decisions, inner_tree.children_right[index])
        # Prune children if both children are leaves now and make the same decision:
        if (self.is_leaf(inner_tree, inner_tree.children_left[index]) and
                self.is_leaf(inner_tree, inner_tree.children_right[index]) and
                (decisions[index] == decisions[inner_tree.children_left[index]]) and
                (decisions[index] == decisions[inner_tree.children_right[index]])):
            
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF

    def prune_duplicate_leaves(self, dt):
        # Remove leaves if both
        
        decisions = dt.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
        self.prune_index(dt.tree_, decisions)
        

    def supert2b(self, superT, X, class_values=[0,1], cv=3, multi_label=False, one_vs_rest=False):
        dt = DecisionTreeClassifier()
        param_list = {'min_samples_split': [0.01, 0.05, 0.2, 3, 2],
                          'min_samples_leaf': [0.01, 0.05, 1, 2, 4],
                          'splitter': ['best', 'random'],
                          'max_depth': [None, 2, 4, 10, 20, 30],
                          'criterion': ['entropy', 'gini'],
                          'max_features': [0.2, 1, 'auto', 'sqrt', 'log2']
                          }

        if not multi_label or (multi_label and one_vs_rest):
            if len(class_values) == 2 or (multi_label and one_vs_rest):
                scoring = 'f1'
            else:
                scoring = 'f1_macro'
        else:
            scoring = 'f1_samples'

        dt_search = HalvingGridSearchCV(dt, param_grid=param_list, scoring=scoring, cv=cv, n_jobs=-1)
        # print(datetime.datetime.now())
        labels = superT.predict(X)
        dt_search.fit(X, labels)
        # print(datetime.datetime.now())
        dt = dt_search.best_estimator_
        self.prune_duplicate_leaves(dt)
        return dt

    def supert2binary(self, superT, newn):
        if superT.is_leaf:
            if newn is None:
                print('problem')
                raise Exception
            else:
                newn.is_leaf = True
                return
        
        if newn is None:
            newn = self.Node(feat_num=superT.feat, labels=superT.labels)
        children = list()
        if superT.children[0]:
            if superT.children[0].is_leaf:
                newl = self.Node(feat_num=superT.feat, labels=superT.children[0].labels)
            else:
                newl = self.Node(feat_num=superT.children[0].feat, labels=superT.children[0].labels)
            
            newn._left_child = newl
            newn.thresh = superT.intervals[0]
            children.append(newl)
            for i in range(1, len(superT.children)):
                if superT.children[i]:
                    if superT.children[i].is_leaf:
                        newr = self.Node(feat_num=superT.feat, labels=superT.children[i].labels)
                    else:
                        newr = self.Node(feat_num=superT.children[i].feat, labels=superT.children[i].labels)
                    
                    children[-1]._right_child = newr
                    children.append(newr)
        for c in range(len(children)):
            
            self.supert2binary(superT.children[c],children[c])
        return newn


    def computeBranch(self, nodeNr, IntervalIf, attributeXf, verbose=True):
        if verbose:
            nodeNr.print_tree()
        if len(IntervalIf) == 0:
            return [nodeNr]
        if nodeNr is None:
            return [None for i in range(len(IntervalIf))]
        if nodeNr.is_leaf:
            pbranch = []
            for i, v in enumerate(IntervalIf):
                pbranch.append(self.Node(labels=nodeNr.labels, is_leaf=True))
            if verbose:
                print('----in Leaf: class_', np.argmax(nodeNr.labels), "Returning", pbranch)
            return pbranch

        else:
            msg = '(not involved)'
            if verbose:
                print('in node with Test', nodeNr.feat, nodeNr.thresh,
                      (msg if not (nodeNr.feat == attributeXf) else 'involved'))
            pbranch = []
            if not (nodeNr.feat == attributeXf):
                pbranch = []
                

                left_children = self.computeBranch(nodeNr._left_child, IntervalIf, attributeXf, verbose)
                right_children = self.computeBranch(nodeNr._right_child, IntervalIf, attributeXf, verbose)
                if verbose:
                    print('I was not involbed got:', len(left_children), "left childrens and ", len(right_children),
                          'rigth')
                for left_c, right_c in zip(left_children, right_children):
                    pbranch.append(self.Node(feat_num=nodeNr.feat,
                                        thresh=nodeNr.thresh,
                                        left_child=left_c,
                                        right_child=right_c))

                if verbose:
                    print("----attribute was not involved, returning", len(pbranch), "roots, just as I am")
                    for p in pbranch:
                        p.print_tree()
                return pbranch


            else:
                new_childs = []
                If1 = IntervalIf[IntervalIf[:, 0] < nodeNr.thresh]
                If2 = IntervalIf[IntervalIf[:, 0] >= nodeNr.thresh]
                if verbose:
                    print("dividing interval in", If1, If2)
                pbranch = []

                left_children = []
                right_children = []
                if If1.shape[0] == 1:
                    if verbose:
                        print("*" * 5, "rec call on left part")
                        print("just added the left child")
                    left_children = [nodeNr._left_child]  # computeBranch(nodeNr._left_child,If1,attributeXf,verbose)
                if If1.shape[0] > 1:
                    left_children = self.computeBranch(nodeNr._left_child, If1, attributeXf, verbose)

                if If2.shape[0] == 1:
                    if verbose:
                        print("*" * 5, "rec call on right part")
                        print("just added the right child")
                    right_children = [nodeNr._right_child]
                if If2.shape[0] > 1:
                    right_children = self.computeBranch(nodeNr._right_child, If2, attributeXf, verbose)
                if verbose:
                    print("PRE UNIFICATION")
                    print(len(left_children), "left childs")
                    print(len(right_children), "Right childs")
                    for left_c in left_children + right_children:
                        left_c.print_tree()
                if verbose:
                    print("----attribute was involved results:")
                    print(len(left_children), "left childs")
                    print(len(right_children), "Right childs")

                for left_c in left_children:
                    pbranch.append(left_c)

                for right_c in right_children:
                    pbranch.append(right_c)
                assert len(pbranch) == len(IntervalIf), "not the same thing"

                if nodeNr.thresh in np.unique(IntervalIf):

                    if verbose:
                        print("I'm inside the interval")
                        print(IntervalIf, "pruning on", IntervalIf[len(left_children)][0])
                    if len(left_children) == 0:
                        adding_node = right_children[0]
                        left_children.append(adding_node)
                    else:
                        adding_node = self.Node(feat_num=nodeNr.feat,
                                           thresh=nodeNr.thresh,
                                           left_child=left_children[-1],
                                           right_child=right_children[-1])

                    left_children[-1] = self.prune_tree(adding_node, IntervalIf[len(left_children)][0], attributeXf)

                    if verbose:
                        print("last decision at index", len(left_children))
                else:
                    if verbose:
                        print("i'm outside the interval" * 6)
                        print("intervals", len(If1), len(If2))
                        print(IntervalIf)
                    major_all_minors = (IntervalIf[:, 0] < nodeNr.thresh)
                    minor_all_majors = (IntervalIf[:, 1] > nodeNr.thresh)
                    if verbose:
                        print(major_all_minors, minor_all_majors)
                        if (major_all_minors.all()):
                            print('im the greather')
                        if (minor_all_majors.all()):
                            print('im the smaller')
                    my_newly_child_r = (right_children[-1] if len(If2) > 0 else nodeNr._right_child)
                    my_newly_child_l = (left_children[-1] if len(left_children) > 0 else nodeNr._left_child)

                    adding_node = self.Node(feat_num=nodeNr.feat,
                                       thresh=nodeNr.thresh,
                                       left_child=my_newly_child_l,
                                       right_child=my_newly_child_r)
                    
                    if minor_all_majors.all():
                        if verbose:
                            print("added at extreme left")
                        adding_node = self.prune_tree(adding_node, IntervalIf[0][1], attributeXf)
                        if len(left_children) == 0:
                            pbranch[0] = adding_node
                        else:
                            pbranch[0] = adding_node
                    
                    if major_all_minors.all():
                        if verbose:
                            print("added at extreme right")
                        adding_node = self.prune_tree(adding_node, IntervalIf[-1][1], attributeXf)
                        if len(right_children) == 0:
                            pbranch[len(left_children) - 1] = adding_node
                        else:
                            pbranch[-1] = adding_node
                    if (not minor_all_majors.all()) and (not major_all_minors.all()):
                        my_pos = np.argmax(major_all_minors & minor_all_majors)
                        if verbose:
                            print("added at my specific position", my_pos)
                        adding_node = self.prune_tree(adding_node, IntervalIf[my_pos][1], attributeXf)
                        pbranch[my_pos] = adding_node

                if verbose:
                    print("result")
                    for p in pbranch:
                        p.print_tree()
            return pbranch

    def mergeDecisionTrees(self, roots, num_classes, level=0, verbose=True, r=''):
        k = len(roots)
        if verbose > 0:
            print("MERGE " + str(k) + " trees, level", level, "previous splits")
        
        if np.array([r.is_leaf for r in roots]).all():
            out_propose = [np.argmax(r.labels) for r in roots]
            val, cou = np.unique(out_propose, return_counts=True)
            labels = np.zeros(num_classes)

            for j, vv in enumerate(val):
                labels[vv] = cou[j]
            if verbose > 0:
                print('all_leafs: votes', val,
                      "voters", cou, "(return superLeaf with labels ", labels, ")")

            return self.SuperNode(is_leaf=True, labels=labels, level = level+1)
        
        else:
            val, cou = np.unique([r.feat for r in roots if r.feat != None], return_counts=True)
            if np.sum(cou) < (k / 2):
                majority = [np.argmax(r.labels) for r in roots if r.is_leaf]
                val_out, cou_out = 0, 0
                if majority:

                    val_out, cou_out = np.unique(majority, return_counts=True)
                    if verbose == 1:
                        print("Can i terminate already?", val_out, cou_out, "majority at", (k / 2) + 1)
                    if np.max(cou_out) >= (k / 2) + 1:
                        labels = np.zeros(num_classes)

                        for j, vv in enumerate(val_out):
                            labels[vv] = cou_out[j]
                        if verbose == 1:
                            print('there are almost all_leafs,values are:',
                                  "votes", val_out, "voters", cou_out, "(return superLeaf with labels", majority, ")")
                            print("ENDING EARLIER!" * 10)
                        return self.SuperNode(is_leaf=True, labels=labels, level = level+1)

            Xf = val[np.argmax(cou)]

            If = list(np.unique([r.thresh for r in roots if r.feat == Xf]))

            if verbose:
                print("most used features", val, cou, 'superNode is X_', Xf, If)

            If = np.array(list(zip([-np.inf] + If, If + [np.inf])))
            branch = []  # sub_trees under conditions
            for i in range(k):
                # sould be an array of roots of the condition sub trees (in which each subtree respect the rule recursivey)
                if (roots[i].is_leaf):
                    branch.append(self.computeBranch(roots[i], If, Xf, verbose=False))
                else:
                    branch.append(self.computeBranch(roots[i], If, Xf, verbose=False))
                assert len(branch[-1]) == len(If), str(self.computeBranch(roots[i], If, Xf, verbose=True))
            if verbose == 2:
                print(len(branch), 'condition trees')
                for K in range(k):
                    for j, v in enumerate(If):
                        print("under interval X", Xf, "<=", v)
                        print("subTRee", K, "has", len(branch[K]), "cdTree printing subtree", j)
                        branch[K][j].print_tree()
            assert len(branch[0]) == (len(If)), "Not equal sub trees intervals tree0 has " + str(
                len(branch[0])) + " condition trees but threre are conditions" + str(len(If))

            children = []
            r += ', X_' + str(Xf)
            for j, v in enumerate(If):
                children.append(self.mergeDecisionTrees([branch[K][j] for K in range(k)], num_classes, level + 1, verbose,
                                                   r + '(' + str(j + 1) + '\\' + str(len(If)) + ')'))
            # print("ended computation of sub tree level",level)

            # mergeDecisionTrees()
            If = If[:, 1]

            new_root = self.SuperNode(feat_num=Xf, intervals=np.array(If), children=children, level = level)
            return new_root


