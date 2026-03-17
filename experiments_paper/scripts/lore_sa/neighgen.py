import copy
import pickle
import numbers
from scipy.stats import binned_statistic
import pandas
import multiprocessing
import pandas as pd
import sys
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from abc import abstractmethod
from scipy.spatial.distance import cdist, hamming, cosine
import math
from .encdec import *
import random
from deap import base, creator, tools, algorithms
from .util import sigmoid, calculate_feature_values, neuclidean

import warnings

warnings.filterwarnings("ignore")


class NeighborhoodGenerator(object):

    def __init__(self, bb_predict=None,  bb_predict_proba=None, feature_values=None, features_map=None, nbr_features=None, nbr_real_features=None,
                 numeric_columns_index=None, ocr=0.1, encdec=None, original_data=None):
        self.bb_predict = bb_predict

        self.bb_predict_proba = bb_predict_proba
        self.feature_values = feature_values
        self.features_map = features_map
        self.nbr_features = nbr_features
        self.nbr_real_features = nbr_real_features
        self.numeric_columns_index = numeric_columns_index
        self.ocr = ocr  # other class ratio
        self.encdec = encdec
        self.original_data = original_data

    @abstractmethod
    def generate(self, x, num_samples=1000):
        return

    def apply_bb_predict(self, X, encoded = None):
        if self.encdec is not None and encoded == None:
            X = self.encdec.dec(X)
        return self.bb_predict(X)

    def apply_bb_predict_proba(self, X, encoded = None):
        if self.encdec is not None and encoded == None:
            X = self.encdec.dec(X)
        return self.bb_predict_proba(X)

    def generate_synthetic_instance(self, from_z=None, mutpb=1.0):
        z = np.zeros(self.nbr_features) if from_z is None else from_z
        for i in range(self.nbr_real_features):
            if np.random.random() <= mutpb:
                real_feature_value = np.random.choice(self.feature_values[i], size=1, replace=True)
                if i in self.numeric_columns_index:
                    z[i] = real_feature_value
                elif type(self.encdec) is OneHotEnc:
                    idx = self.features_map[i][real_feature_value[0]]
                    z[idx] = 1.0
                elif type(self.encdec) is MyTargetEnc:
                    encs = self.encdec.get_cate_map(i, real_feature_value[0])
                    z[i] = encs
        return z

    def balance_neigh(self, x, Z, num_samples):
        Yb = self.apply_bb_predict(Z)
        #Yb = self.bb_predict(Z)
        class_counts = np.unique(Yb, return_counts=True)

        if len(class_counts[0]) <= 2:
            ocs = int(np.round(num_samples * self.ocr))
            #Z1 = self.__rndgen_not_class(ocs, self.bb_predict(x)[0])
            Z1 = self.__rndgen_not_class(ocs, self.apply_bb_predict(x.reshape(1, -1))[0])
            if len(Z1) > 0:
                Z = np.concatenate((Z, Z1), axis=0)
        else:
            max_cc = np.max(class_counts[1])
            max_cc2 = np.max([cc for cc in class_counts[1] if cc != max_cc])
            if max_cc2 / len(Yb) < self.ocr:
                ocs = int(np.round(num_samples * self.ocr)) - max_cc2
                Z1 = self.__rndgen_not_class(ocs, self.apply_bb_predict(x.reshape(1, -1))[0])
                #Z1 = self.__rndgen_not_class(ocs, self.bb_predict(x)[0])
                if len(Z1) > 0:
                    Z = np.concatenate((Z, Z1), axis=0)
        return Z

    def __rndgen_not_class(self, num_samples, class_value, max_iter=1000):
        Z = list()
        iter_count = 0
        multi_label = isinstance(class_value, np.ndarray)
        while len(Z) < num_samples:
            z = self.generate_synthetic_instance()
            y = self.apply_bb_predict(z.reshape(1, -1))[0]
            #y = self.bb_predict(z)[0]
            flag = y != class_value if not multi_label else np.all(y != class_value)
            if flag:
                Z.append(z)
            iter_count += 1
            if iter_count >= max_iter:
                break

        Z = np.array(Z)
        Z = np.nan_to_num(Z)
        return Z

#new class for the generation of the neighborhood based on the extraction of counterfactuals
#here x is already coded, hence the generation of the neighbourhood is in the latent space
class CounterGenerator(NeighborhoodGenerator):
    def __init__(self, bb_predict, bb_predict_proba, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index, ocr=0.1, original_data = None, encdec=None, alpha1=0.5, alpha2=0.5,
                 metric=neuclidean, ngen=100, mutpb=0.2, random_seed = None,
                 cxpb=0.5, tournsize=3, halloffame_ratio=0.1, closest=True, max_counter = 1, verbose=False):
        super(CounterGenerator, self).__init__(bb_predict=bb_predict, bb_predict_proba=bb_predict_proba, feature_values=feature_values, features_map=features_map,
                                              nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                              numeric_columns_index= numeric_columns_index, ocr=ocr,
                                              original_data = original_data, encdec=encdec )
        self.closest = closest
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.metric = metric
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.max_counter = max_counter
        self.tournsize = tournsize
        self.halloffame_ratio = halloffame_ratio
        self.random_seed = random_seed
        if original_data is None:
            raise ValueError('This method can not be applied without a sample of the original data.')

    #for every feature we define the bins and take the mean value
    def create_bins(self):
        feature_bins = dict()
        if type(self.encdec) is OneHotEnc:
            for i in range(self.nbr_real_features):
                if i in self.numeric_columns_index:
                    bins = (max(self.original_data[i]) - min(self.original_data[i])) / float(
                        math.ceil(math.log2(len(self.original_data[i]))) + 1)

                    binned = pd.cut(self.original_data[i], bins)
                    mean_bins = self.original_data[binned].mean()
                    feature_bins[i] = dict()
                    feature_bins[i]['bins'] = mean_bins.keys()
                    feature_bins[i]['avg'] = mean_bins.values()
                else:
                    print('in synth neigh gen ', i)
                    feature_bins[i] = dict()
                    feature_bins[i]['bins'] = self.features_map[i].keys()
                    feature_bins[i]['avg'] = self.features_map[i].values()
        else:
            for f in range(self.original_data.shape[1]):
                #define the ideal number of bins
                bins = math.ceil((max(self.original_data[:, f]) - min(self.original_data[:, f]))/float(math.ceil(math.log2(len(self.original_data[: , f])))+1))
                #create bins
                binned =  binned_statistic(self.original_data[:,f], values = self.original_data[:,f], bins=bins, statistic='mean')
                stat = binned.statistic
                correct_values = list()
                for i in binned.binnumber:
                    correct_values.append(stat[i-1])
                feature_bins[f] = dict()
                feature_bins[f]['avg'] = stat
                feature_bins[f]['bins'] = correct_values
                feature_bins[f]['edges'] = binned.bin_edges
        return feature_bins

    def find_closest_counter(self, counter_list, x):
        closests = list()
        for feat in counter_list.keys():
            vals = list()
            element = None
            closest = sys.maxsize
            for new_record in counter_list[feat]:
                print('new record ', new_record, new_record.shape, x.shape)
                distance = 0
                for f in feat:
                    distance += abs(new_record[0][f] - x[0][f])
                if distance < closest:
                    closest = distance
                    element = (feat, new_record)
            closests.append(element)

        new_c = list()
        for c in closests:
            print('generate closest ', c[0])
            
            trial = np.copy(x)
            bb_x = self.apply_bb_predict(x)
            cicla = True
            while cicla:
                for el in c[0]:
                    if c[1][0][el] < x[0][el]:
                        trial[0][el] = np.random.uniform(low=c[1][0][el], high=x[0][el], size=1)
                    else:
                        trial[0][el] = np.random.uniform(low=x[0][el], high=c[1][0][el], size=1)
                pred = self.bb_predict(trial)
                if pred != bb_x:
                    cicla = False
                
                print('prima dell errore ', self.apply_bb_predict_proba(trial))
                print('prima prima ', self.apply_bb_predict_proba(c[1]))
                if self.apply_bb_predict_proba(trial)[:, bb_x] > self.apply_bb_predict_proba(c[1])[:,bb_x]:
                    new_c.append(trial)
        return new_c

    def for_loop_counter(self, n_feat, n_iter, possibilities, x):
        counters = dict()
        for i in range(n_iter):
            feats = random.sample(list(possibilities.keys()), n_feat)
            inds = list()
            for f in range(n_feat):
                inds.append(random.choice(range(0,len(possibilities[feats[f]]))))
            to_analize = list()
            for f in range(0, len(inds)):
                to_analize.append(possibilities[feats[f]][inds[f]])
            prova = to_analize[0]
            for t in range(1, len(to_analize)):
                prova[0][inds[t]] = to_analize[t][0][inds[t]]
            if self.apply_bb_predict(prova) != self.apply_bb_predict(x):
                inds_string = tuple(inds)
                if inds_string in counters.keys():
                    counters[inds_string].append(prova)
                else:
                    counters[inds_string] = list()
                    counters[inds_string].append(prova)
        print('counters in possibilities ', len(counters), counters.keys())
        return counters

    def find_in_possibilities(self, possibilities, x, n_iter):
        print('la len di possibilities ', len(possibilities.keys()))
        for i in range(2, len(possibilities.keys())):
            counters = self.for_loop_counter(i, 1000, possibilities, x)
            if counters:
                print('sono in break')
                break
        print('ecco il counter che ho trovato con ', len(counters), i, len(possibilities.keys()))
        if not counters:
            raise Exception(' Impossible to find a counterfactual')
        else:
            generate_on = self.find_closest_counter(counters, x)
        return generate_on

    def generate(self, x, num_samples=1000):
        counter_list = dict()
        possibilities = dict()
        counter_found = False
        try:
            pred_proba_x = self.apply_bb_predict_proba(x)
            pred_x = self.apply_bb_predict(x)
            to_explain = copy.deepcopy(x)
        except:
            raise ValueError('Predict proba has to be defined for the Counter Generation')
        feature_bins = self.create_bins()
    
        for f in feature_bins.keys():
            
            actual = feature_bins[f]
            for val in actual['avg']:
                temp = copy.deepcopy(to_explain)
                temp[:,f] = val
                pred_actual = self.apply_bb_predict(temp)
                if pred_actual != pred_x:
                    if f in counter_list.keys():
                        counter_list[f].append(temp)
                    else:
                        counter_list[f] = list()
                        counter_list[f].append(temp)
                        counter_found = True
                else:
                    pred_proba_actual = self.apply_bb_predict_proba(temp)
                    if pred_proba_actual[0][pred_x] < pred_proba_x[0][pred_x]:
                        if f in possibilities.keys():
                            possibilities[f].append(temp)
                        else:
                            possibilities[f] = list()
                            possibilities[f].append(temp)
        if counter_found:
            generate_on = self.find_closest_counter(counter_list, x)
        
        elif possibilities.keys():
            generate_on = self.find_in_possibilities(possibilities, x, 20)
        else:
            raise Exception('no counterfactual found')
        
        randgen = RandomGeneticGenerator(self.bb_predict, self.feature_values, self.features_map,
                                        self.nbr_features, self.nbr_real_features, self.numeric_columns_index,
                                        ocr=self.ocr, alpha1=self.alpha1, alpha2=self.alpha2,
                                        metric=self.metric, ngen=self.ngen, mutpb=self.mutpb, cxpb=self.cxpb,
                                        tournsize=self.tournsize, halloffame_ratio=self.halloffame_ratio,
                                        random_seed=self.random_seed, encdec=self.encdec, verbose=False)
        
        cont = 0
        Z = randgen.generate(x, num_samples=1000)
        for ind in range(0, self.max_counter):
            rec_index = random.choice([0,len(generate_on)-1])
            rec = generate_on[rec_index]
            Z_temp = (randgen.generate(rec, num_samples=500))
            Z = np.concatenate((Z, Z_temp), axis=0)
            print('finito con cont ', cont, Z.shape)
        return Z



class RandomGenerator(NeighborhoodGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index, ocr=0.1, encdec=None):
        super(RandomGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                              nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                              numeric_columns_index= numeric_columns_index, ocr=ocr, encdec=encdec)

    def generate(self, x, num_samples=1000):
        Z = np.zeros((num_samples, self.nbr_features))
        for j in range(num_samples):
            Z[j] = self.generate_synthetic_instance()

        Z = super(RandomGenerator, self).balance_neigh(x, Z, num_samples)
        Z = np.nan_to_num(Z)
        Z[0] = x.copy()
        return Z


class GeneticGenerator(NeighborhoodGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features, numeric_columns_index,
                 ocr=0.1, alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5,
                 tournsize=3, halloffame_ratio=0.1, random_seed=None, encdec = None, verbose=False):
        super(GeneticGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                               nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                               numeric_columns_index=numeric_columns_index, ocr=ocr, encdec=encdec)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.metric = metric
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.tournsize = tournsize
        self.halloffame_ratio = halloffame_ratio
        self.verbose = verbose
        random.seed(random_seed)

    def generate(self, x, num_samples=1000):
        if self.encdec is not None:
            x = x.flatten()
        num_samples_eq = int(np.round(num_samples * 0.5))
        num_samples_noteq = int(np.round(num_samples * 0.5))
        toolbox_eq = self.setup_toolbox(x, self.fitness_equal, num_samples_eq)
        population_eq, halloffame_eq, logbook_eq = self.fit(toolbox_eq, num_samples_eq)
        Z_eq = self.add_halloffame(population_eq, halloffame_eq)
        toolbox_noteq = self.setup_toolbox(x, self.fitness_notequal, num_samples_noteq)
        population_noteq, halloffame_noteq, logbook_noteq = self.fit(toolbox_noteq, num_samples_noteq)
        Z_noteq = self.add_halloffame(population_noteq, halloffame_noteq)
        Z = np.concatenate((Z_eq, Z_noteq), axis=0)

        Z = super(GeneticGenerator, self).balance_neigh(x, Z, num_samples)
        Z = np.nan_to_num(Z)
        Z[0] = x.copy()
        return Z

    def add_halloffame(self, population, halloffame):
        fitness_values = [p.fitness.wvalues[0] for p in population]
        fitness_values = sorted(fitness_values)
        fitness_diff = [fitness_values[i + 1] - fitness_values[i] for i in range(0, len(fitness_values) - 1)]

        sorted_array = np.argwhere(fitness_diff == np.amax(fitness_diff)).flatten().tolist()
        if len(sorted_array) == 0:
            fitness_value_thr = -np.inf
        else:
            index = np.max(sorted_array)
            fitness_value_thr = fitness_values[index]

        Z = list()
        for p in population:
            Z.append(p)

        for h in halloffame:
            if h.fitness.wvalues[0] > fitness_value_thr:
                Z.append(h)

        return np.array(Z)

    def setup_toolbox(self, x, evaluate, population_size):

        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", np.ndarray, fitness=creator.fitness)

        toolbox = base.Toolbox()
        toolbox.register("feature_values", self.record_init, x)
        toolbox.register("individual", tools.initIterate, creator.individual, toolbox.feature_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)
        toolbox.register("clone", self.clone)
        toolbox.register("evaluate", evaluate, x)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate, toolbox)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        return toolbox

    def setup_toolbox_noteq(self, x, x1, evaluate, population_size):

        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", np.ndarray, fitness=creator.fitness)

        toolbox = base.Toolbox()
        toolbox.register("feature_values", self.record_init, x1)
        toolbox.register("individual", tools.initIterate, creator.individual, toolbox.feature_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)

        toolbox.register("clone", self.clone)
        toolbox.register("evaluate", evaluate, x)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate, toolbox)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

        return toolbox

    def fit(self, toolbox, population_size):
        halloffame_size = int(np.round(population_size * self.halloffame_ratio))
        population = toolbox.population(n=population_size)
        halloffame = tools.HallOfFame(halloffame_size, similar=np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
                                                  ngen=self.ngen, stats=stats, halloffame=halloffame,
                                                  verbose=self.verbose)

        return population, halloffame, logbook

    def record_init(self, x):
        return x
    def random_init(self):
        z = self.generate_synthetic_instance()
        return z

    def clone(self, x):
        return pickle.loads(pickle.dumps(x))

    def mutate(self, toolbox, x):
        z = toolbox.clone(x)
        z = self.generate_synthetic_instance(from_z=z, mutpb=self.mutpb)
        return z,

    def fitness_equal(self, x, x1):
        if isinstance(self.metric, numbers.Number):
            self.metric = neuclidean
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        feature_similarity = sigmoid(feature_similarity_score) if feature_similarity_score < 1.0 else 0.0
        
        y = self.apply_bb_predict(x.reshape(1, -1))
        y1 = self.apply_bb_predict(x1.reshape(1, -1))

        target_similarity_score = 1.0 - hamming(y, y1)
        # target_similarity = target_similarity_score if target_similarity_score >= self.eta2 else 0.0
        target_similarity = sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,

    def fitness_notequal(self, x, x1):
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        # feature_similarity = feature_similarity_score if feature_similarity_score >= self.eta1 else 0.0
        feature_similarity = sigmoid(feature_similarity_score)

        #y = self.bb_predict(x.reshape(1, -1))[0]
        #y1 = self.bb_predict(x1.reshape(1, -1))[0]

        y = self.apply_bb_predict(x.reshape(1, -1))#[0]
        y1 = self.apply_bb_predict(x1.reshape(1, -1))#[0]

        target_similarity_score = 1.0 - hamming(y, y1)
        # target_similarity = target_similarity_score if target_similarity_score < self.eta2 else 0.0
        target_similarity = 1.0 - sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,


class GeneticProbaGenerator(GeneticGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index,
                 ocr=0.1, alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2,
                 cxpb=0.5, tournsize=3, halloffame_ratio=0.1, bb_predict_proba=None, random_seed=None, encdec = None, verbose=False):
        super(GeneticProbaGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                                    nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                                    numeric_columns_index=numeric_columns_index,
                                                    ocr=ocr, alpha1=alpha1, alpha2=alpha2, metric=metric, ngen=ngen,
                                                    mutpb=mutpb,cxpb=cxpb, tournsize=tournsize,
                                                    halloffame_ratio=halloffame_ratio, random_seed=random_seed, encdec=encdec)
        self.bb_predict_proba = bb_predict_proba

    def fitness_equal(self, x, x1):
        return self.fitness_equal_proba(x, x1)

    def fitness_notequal(self, x, x1):
        return self.fitness_notequal_proba(x, x1)

    def fitness_equal_proba(self, x, x1):
        x = np.array(x)
        x1 = np.array(x1)
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        feature_similarity = sigmoid(feature_similarity_score) if feature_similarity_score < 1.0 else 0.0
        # feature_similarity = sigmoid(feature_similarity_score)

        #y = self.bb_predict_proba(x.reshape(1, -1))[0]
        #y1 = self.bb_predict_proba(x1.reshape(1, -1))[0]
        y = self.apply_bb_predict_proba(x.reshape(1, -1))[0]
        y1 = self.apply_bb_predict_proba(x1.reshape(1, -1))[0]

        # target_similarity_score = np.sum(np.abs(y - y1))
        target_similarity_score = 1.0 - cosine(y, y1)
        target_similarity = sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,

    def fitness_notequal_proba(self, x, x1):
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        feature_similarity = sigmoid(feature_similarity_score)

        #y = self.bb_predict_proba(x.reshape(1, -1))[0]
        #y1 = self.bb_predict_proba(x1.reshape(1, -1))[0]

        y = self.apply_bb_predict_proba(x.reshape(1, -1))[0]
        y1 = self.apply_bb_predict_proba(x1.reshape(1, -1))[0]

        # target_similarity_score = np.sum(np.abs(y - y1))
        target_similarity_score = 1.0 - cosine(y, y1)
        target_similarity = 1.0 - sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,


class RandomGeneticGenerator(GeneticGenerator, RandomGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features, numeric_columns_index,
                 ocr=0.1, alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5,
                 tournsize=3, halloffame_ratio=0.1, random_seed=None, encdec=None, verbose=False):
        super(RandomGeneticGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                                     nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                                     numeric_columns_index=numeric_columns_index, ocr=ocr, encdec=encdec)
        super(RandomGeneticGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                                     nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                                     numeric_columns_index=numeric_columns_index, ocr=ocr, encdec=encdec,
                                                     alpha1=alpha1, alpha2=alpha2, metric=metric, ngen=ngen, mutpb=mutpb,
                                                     cxpb=cxpb, tournsize=tournsize,
                                                     halloffame_ratio=halloffame_ratio, random_seed=random_seed, verbose=verbose)

    def generate(self, x, num_samples=1000):
        Zg = GeneticGenerator.generate(self, x, num_samples // 2)
        Zr = RandomGenerator.generate(self, x, num_samples // 2)
        Z = np.concatenate((Zg, Zr[1:]), axis=0)
        Z = np.nan_to_num(Z)
        return Z


class RandomGeneticProbaGenerator(GeneticProbaGenerator, RandomGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index,
                 ocr=0.1, alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2,
                 cxpb=0.5, tournsize=3, halloffame_ratio=0.1, bb_predict_proba=None, random_seed=None, encdec=None, verbose=False):
        super(RandomGeneticProbaGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values,
                                                          features_map=features_map,
                                                          nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                                          numeric_columns_index=numeric_columns_index, ocr =ocr, encdec=encdec)
        super(RandomGeneticProbaGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                                          nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                                          numeric_columns_index=numeric_columns_index,
                                                          ocr=ocr, alpha1=alpha1, alpha2=alpha2, metric=metric, ngen=ngen, mutpb=mutpb,
                                                          cxpb=cxpb, tournsize=tournsize, halloffame_ratio=halloffame_ratio, bb_predict_proba=bb_predict_proba,
                                                          random_seed=random_seed, encdec=encdec, verbose=verbose)

    def generate(self, x, num_samples=1000):
        Zg = GeneticProbaGenerator.generate(self, x, num_samples // 2)
        Zr = RandomGenerator.generate(self, x, num_samples // 2)
        Z = np.concatenate((Zg, Zr[1:]), axis=0)
        Z = np.nan_to_num(Z)
        return Z


class ClosestInstancesGenerator(NeighborhoodGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index, ocr=0.1, K=None, rK=None, k=None, core_neigh_type='unified', alphaf=0.5,
                 alphal=0.5, metric_features=neuclidean, metric_labels='hamming', categorical_use_prob=True,
                 continuous_fun_estimation=False, size=1000, encdec=None, verbose=False):
        super(ClosestInstancesGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                                        nbr_features=nbr_features, nbr_real_features=nbr_real_features,
                                                        numeric_columns_index=numeric_columns_index, ocr=ocr, encdec=encdec)
        self.K = K
        self.rK = rK
        self.k = k if k is not None else int(0.5 * np.sqrt(len(self.rK))) + 1
        # self.k = np.min([self.k, len(self.rK)])
        self.core_neigh_type = core_neigh_type
        self.alphaf = alphaf
        self.alphal = alphal
        self.metric_features = metric_features
        self.metric_labels = metric_labels
        self.categorical_use_prob = categorical_use_prob
        self.continuous_fun_estimation = continuous_fun_estimation
        self.size = size
        self.verbose = verbose

    def generate(self, x, num_samples=1000):

        K = np.concatenate((x.reshape(1, -1), self.K), axis=0)
        #Yb = self.bb_predict(K)
        Yb = self.apply_bb_predict(K)
        if self.core_neigh_type == 'mixed':
            Kn = (K - np.min(K)) / (np.max(K) - np.min(K))
            fdist = cdist(Kn, Kn[0].reshape(1, -1), metric=self.metric_features).ravel()
            rk_idxs = np.where(np.argsort(fdist)[:max(int(self.k * self.alphaf), 2)] < len(self.rK))[0]
            Zf = self.rK[rk_idxs]

            ldist = cdist(Yb, Yb[0].reshape(1, -1), metric=self.metric_labels).ravel()
            rk_idxs = np.where(np.argsort(ldist)[:max(int(self.k * self.alphal), 2)] < len(self.rK))[0]
            Zl = self.rK[rk_idxs]
            rZ = np.concatenate((Zf, Zl), axis=0)
        elif self.core_neigh_type == 'unified':
            def metric_unified(x, y):
                n = K.shape[1]
                m = Yb.shape[1]
                distf = cdist(x[:n].reshape(1, -1), y[:n].reshape(1, -1), metric=self.metric_features).ravel()
                distl = cdist(x[n:].reshape(1, -1), y[n:].reshape(1, -1), metric=self.metric_labels).ravel()
                return n / (n + m) * distf + m / (n + m) * distl
            U = np.concatenate((K, Yb), axis=1)
            Un = (U - np.min(U)) / (np.max(U) - np.min(U))
            udist = cdist(Un, Un[0].reshape(1, -1), metric=metric_unified).ravel()
            rk_idxs = np.where(np.argsort(udist)[:self.k] < len(self.rK))[0]
            rZ = self.rK[rk_idxs]
        else:  # self.core_neigh_type == 'simple':
            Kn = (K - np.min(K)) / (np.max(K) - np.min(K))
            fdist = cdist(Kn, Kn[0].reshape(1, -1), metric=self.metric_features).ravel()
            rk_idxs = np.where(np.argsort(fdist)[:self.k] < len(self.rK))[0]
            Zf = self.rK[rk_idxs]
            rZ = Zf

        if self.verbose:
            print('calculating feature values')

        feature_values = calculate_feature_values(rZ, self.numeric_columns_index,
                                                  categorical_use_prob=self.categorical_use_prob,
                                                  continuous_fun_estimation=self.continuous_fun_estimation,
                                                  size=self.size)
        rndgen = RandomGenerator(self.bb_predict, feature_values, self.features_map, self.nbr_features,
                                 self.nbr_real_features, self.numeric_columns_index, self.ocr)
        Z = rndgen.generate(x, num_samples)
        Z = np.nan_to_num(Z)
        return Z

class CFSGenerator(NeighborhoodGenerator):
    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index, ocr=0.1, n_search=10000,n_batch=1000,lower_threshold=0,upper_threshold=4,
            kind="gaussian_matched",sampling_kind=None, stopping_ratio=0.01,
            check_upper_threshold=True, final_counterfactual_search=True,
            custom_sampling_threshold=None, custom_closest_counterfactual=None,
            n=500, balance=False, verbose=False, cut_radius=None, forced_balance_ratio=0.5, downward_only=True, encdec=None):
        super(CFSGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                           nbr_features=nbr_features,nbr_real_features=nbr_real_features,
                                           numeric_columns_index=numeric_columns_index, ocr=ocr, encdec=encdec)
        self.n_search = n_search
        self.n_batch = n_batch
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.kind = kind
        self.sampling_kind = sampling_kind
        self.verbose = verbose
        self.check_upper_threshold = check_upper_threshold
        self.final_counterfactual_search = final_counterfactual_search
        self.stopping_ratio = stopping_ratio
        self.n = n
        self.forced_balance_ratio = forced_balance_ratio
        self.custom_closest_counterfactual = custom_closest_counterfactual
        self.balance = balance
        self.custom_sampling_threshold = custom_sampling_threshold
        self.cut_radius = cut_radius
        self.closest_counterfactual = None
        self.downward_only = downward_only


    def generate(self, x, num_samples=1000, **kwargs):
        x_label = self.apply_bb_predict(x.reshape(1, -1))
        if (self.closest_counterfactual is None) and (self.custom_closest_counterfactual is None):
            self.counterfactual_search(x,x_label=x_label, **kwargs)

        self.kind = self.sampling_kind if self.sampling_kind is not None else self.kind

        Z = self.neighborhood_sampling(x,x_label=x_label,**kwargs)
        Z = np.nan_to_num(Z)
        return Z

    def counterfactual_search(self,x,x_label, **kwargs):
        x = x.reshape(1,-1)
        self.closest_counterfactual, self.best_threshold = self.binary_sampling_search(x,x_label, downward_only=self.downward_only, **kwargs)
        return self.closest_counterfactual, self.best_threshold

    def binary_sampling_search(self, x, x_label, downward_only=True, **kwargs):
        if self.verbose:
            print("Binary sampling search:", self.kind)


        if self.check_upper_threshold:
            if self.verbose:
                print('---     ', self.n, self.n_batch, int(self.n / self.n_batch))
            print('binary sampling search ', self.n, self.n_batch)
            if int(self.n / float(self.n_batch)) < 1:
                raise Exception('Change the value of n or n_batch')
            for i in range(int(self.n / float(self.n_batch))):
                Z = self.vicinity_sampling(
                    x,
                    n=self.n_batch,
                    threshold=self.upper_threshold,
                    **kwargs
                )

                y = self.apply_bb_predict(Z)
                if not np.all(y == x_label):
                    break
            if i == list(range(int(self.n / self.n_batch)))[-1]:
                raise Exception("No counterfactual found, increase upper threshold or n_search.")

        change_lower = False
        latest_working_threshold = self.upper_threshold
        Z_counterfactuals = list()
        if self.verbose:
            print('lower threshold, upper threshold ', self.lower_threshold, self.upper_threshold)
        while self.lower_threshold / self.upper_threshold < self.stopping_ratio:
            if change_lower:
                if downward_only:
                    break
                self.lower_threshold = threshold
            threshold = (self.lower_threshold + self.upper_threshold) / 2
            change_lower = True
            if self.verbose:
                print("   Testing threshold value:", threshold)
            for i in range(int(self.n / self.n_batch)):
                Z = self.vicinity_sampling(
                    x,
                    n=self.n_batch,
                    threshold=threshold,
                    **kwargs
                )

                y = self.apply_bb_predict(Z)
                if not np.all(y == x_label):  
                    counterfactuals_idxs = np.argwhere(y != x_label).ravel()
                    Z_counterfactuals.append(Z[counterfactuals_idxs])
                    latest_working_threshold = threshold
                    self.upper_threshold = threshold
                    change_lower = False
                    break
        if self.verbose:
            print("   Best threshold found:", latest_working_threshold)
        if self.final_counterfactual_search:
            if self.verbose:
                print("   Final counterfactual search... (this could take a while)", end=" ")
            Z = self.vicinity_sampling(
                x,
                n = self.n,
                threshold=latest_working_threshold,
                **kwargs
            )
            y = self.apply_bb_predict(Z)
            counterfactuals_idxs = np.argwhere(y != x_label).ravel()
            Z_counterfactuals.append(Z[counterfactuals_idxs])
            if self.verbose:
                print("Done!")
        Z_counterfactuals = np.concatenate(Z_counterfactuals)
        closest_counterfactual = min(Z_counterfactuals, key=lambda p: sum((p - x.ravel()) ** 2))
        return closest_counterfactual, latest_working_threshold

    def neighborhood_sampling(self, x, x_label, custom_closest_counterfactual=None, custom_sampling_threshold=None, **kwargs):
        if custom_closest_counterfactual is not None:
            self.closest_counterfactual = custom_closest_counterfactual
        if self.cut_radius:
            self.best_threshold = np.linalg.norm(x - self.closest_counterfactual)
            if self.verbose:
                print("Setting new threshold at radius:", self.best_threshold)
            if self.kind not in ["uniform_sphere"]:
                warnings.warn("cut_radius=True, but for the method " + self.kind + " the threshold is not a radius.")
        if custom_sampling_threshold is not None:
            self.best_threshold = custom_sampling_threshold
            if self.verbose:
                print("Setting custom threshold:", self.best_threshold)
        Z = self.vicinity_sampling(
            self.closest_counterfactual.reshape(1,-1),
            n=self.n,
            threshold=self.best_threshold,
            **kwargs
        )

        if self.forced_balance_ratio is not None:
            y = self.apply_bb_predict(Z)
            y = 1 * (y == x_label)
            n_minority_instances = np.unique(y, return_counts=True)[1].min()
            if (n_minority_instances / self.n) < self.forced_balance_ratio:
                if self.verbose:
                    print("Forced balancing neighborhood...", end=" ")
                n_desired_minority_instances = int(self.forced_balance_ratio * self.n)
                n_desired_majority_instances = self.n - n_desired_minority_instances
                minority_class = np.argmin(np.unique(y, return_counts=True)[1])
                sampling_strategy = n_desired_minority_instances / n_desired_majority_instances
                while n_minority_instances < n_desired_minority_instances:
                    Z_ = self.vicinity_sampling(
                        self.closest_counterfactual.reshape(1,-1),
                        n=self.n_batch,
                        threshold=self.best_threshold if custom_sampling_threshold is None else custom_sampling_threshold,
                        **kwargs
                    )

                    y_ = self.apply_bb_predict(Z_)
                    y_ = 1 * (y_ == x_label)
                    n_minority_instances += np.unique(y_, return_counts=True)[1][minority_class]
                    Z = np.concatenate([Z, Z_])
                    y = np.concatenate([y, y_])
                rus = RandomUnderSampler(random_state=0, sampling_strategy=sampling_strategy)
                Z, y = rus.fit_resample(Z, y)
                if len(Z) > self.n:
                    Z, _ = train_test_split(Z, train_size=self.n, stratify=y)
                if self.verbose:
                    print("Done!")

        if self.balance:
            if self.verbose:
                print("Balancing neighborhood...", end=" ")
            rus = RandomUnderSampler(random_state=0)
            y = self.apply_bb_predict(Z)
            y = 1 * (y == x_label)
            Z, _ = rus.fit_resample(Z, y)
            if self.verbose:
                print("Done!")
        return Z

    def vicinity_sampling(self, x, n, threshold=None,**kwargs):
        if self.verbose:
            print("\nSampling -->", self.kind)
        if self.kind == "gaussian":
            Z = self.gaussian_vicinity_sampling(x, threshold, n)
        elif self.kind == "gaussian_matched":
            Z = self.gaussian_matched_vicinity_sampling(x, threshold, n)
        elif self.kind == "gaussian_global":
            Z = self.gaussian_global_sampling(x, n)
        elif self.kind == "uniform_sphere":
            Z = self.uniform_sphere_vicinity_sampling(x, n, threshold)
        elif self.kind == "uniform_sphere_scaled":
            Z = self.uniform_sphere_scaled_vicinity_sampling(x, n, threshold)
        else:
            raise Exception("Vicinity sampling kind not valid", self.kind)
        return Z

    def gaussian_vicinity_sampling(self, z, epsilon, n=1):
        return z + (np.random.normal(size=(n, z.shape[1])) * epsilon)

    def gaussian_vicinity_sampling(self, z, epsilon, n=1):
        return z + (np.random.normal(size=(n, z.shape[1])) * epsilon)

    def gaussian_global_sampling(self, z, n=1):
        return np.random.normal(size=(n, z.shape[1]))

    def uniform_sphere_origin(self, n, d, r=1):
        """Generate "num_points" random points in "dimension" that have uniform probability over the unit ball scaled
        by "radius" (length of points are in range [0, "radius"]).

        Parameters
        ----------
        n : int
            number of points to generate
        d : int
            dimensionality of each point
        r : float
            radius of the sphere

        Returns
        -------
        array of shape (n, d)
            sampled points
        """
        # First generate random directions by normalizing the length of a
        # vector of random-normal values (these distribute evenly on ball).
        random_directions = np.random.normal(size=(d, n))
        random_directions /= np.linalg.norm(random_directions, axis=0)
        # Second generate a random radius with probability proportional to
        # the surface area of a ball with a given radius.
        random_radii = np.random.random(n) ** (1 / d)
        # Return the list of random (direction & length) points.
        return r * (random_directions * random_radii).T

    def uniform_sphere_vicinity_sampling(self, z, n=1, r=1):
        Z = self.uniform_sphere_origin(n, z.shape[1], r)
        self.translate(Z, z)
        return Z

    def uniform_sphere_scaled_vicinity_sampling(self, z, n=1, threshold=1):
        Z = self.uniform_sphere_origin(n, z.shape[1], r=1)
        Z *= threshold
        self.translate(Z, z)
        return Z

    def translate(self, X, center):
        """Translates a origin centered array to a new center

        Parameters
        ----------
        X : array
            data to translate centered in the axis origin
        center : array
            new center point

        Returns
        -------
        None
        """
        for axis in range(center.shape[-1]):
            X[..., axis] += center[..., axis]











