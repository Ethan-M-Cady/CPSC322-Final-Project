from mysklearn import myutils

from math import ceil
from mysklearn import myutils
import numpy as np


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    if type(test_size) == float:
        test_len = ceil(test_size*len(X))
    else:
        test_len = test_size
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    if shuffle == True:
        if random_state == None:
            myutils.randomize_in_place(X, y)
        else:
            myutils.randomize_in_place(X, y, random_state)

    for i in range(len(X)):
        if i < len(X)-test_len:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])
    return X_train, X_test, y_train, y_test  # TODO: fix this


def kfold_split(X, n_splits=5, random_state=None, shuffle=False):

    X_indexes = [i for i in range(len(X))]
    if shuffle == True:
        if random_state == None:
            myutils.randomize_in_place(X_indexes)
        else:
            myutils.randomize_in_place(X_indexes, ran_seed=random_state)
    X_test_folds = [[] for i in range(n_splits)]
    for i in range(len(X)):
        X_test_folds[i % n_splits].append(X_indexes[i])
    X_train_folds = []
    for j in range(len(X_test_folds)):
        indexes = []
        for h in range(len(X_test_folds)):
            if h != j:
                indexes = indexes + X_test_folds[h]
        X_train_folds.append(indexes)
    answer = list(zip(X_train_folds, X_test_folds))
    return answer


def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    X_indexes = [i for i in range(len(X))]
    if shuffle == True:
        if random_state == None:
            myutils.randomize_in_place(X_indexes, y)
        else:
            myutils.randomize_in_place(X_indexes, y, random_state)
    grouped_X = []
    grouped_y = []
    for i in range(len(X)):
        if grouped_y.count(y[i]) > 0:
            grouped_X[grouped_y.index(y[i])].append(X_indexes[i])
        else:
            grouped_y.append(y[i])
            grouped_X.append([X_indexes[i]])

    X_test_folds = [[] for i in range(n_splits)]
    current_index = 0
    for i in range(len(grouped_X)):
        for j in range(len(grouped_X[i])):
            current_index += 1
            X_test_folds[current_index % n_splits].append(grouped_X[i][j])

    X_train_folds = []
    for j in range(len(X_test_folds)):
        indexes = []
        for h in range(len(X_test_folds)):
            if h != j:
                indexes = indexes + X_test_folds[h]
        X_train_folds.append(indexes)
    answer = list(zip(X_train_folds, X_test_folds))
    return answer


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    if random_state != None:
        np.random.seed(random_state)
    if n_samples == None:
        n_samples = len(X)
    X_sample = []
    X_out_of_bag = []
    rand_index_list = []
    for i in range(n_samples):
        rand_index = np.random.randint(0, len(X))
        rand_index_list.append(rand_index)
        X_sample.append(X[rand_index])
    for h in range(len(X)):
        if rand_index_list.count(h) == 0:
            X_out_of_bag.append(X[h])
    if y is None:
        return X_sample, X_out_of_bag, None, None
    else:
        y_sample = []
        y_out_of_bag = []
        for j in range(len(y)):
            if rand_index_list.count(j) == 0:
                y_out_of_bag.append(y[j])
        for value in rand_index_list:
            y_sample.append(y[value])
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag


def confusion_matrix(y_true, y_pred, labels):
    matrix = [[0 for i in range(len(labels))] for i in range(len(labels))]
    for i in range(len(y_pred)):
        matrix[labels.index(y_true[i])][labels.index(y_pred[i])] += 1

    return matrix


def accuracy_score(y_true, y_pred, normalize=True):
    score = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            score += 1
    if normalize == False:
        return score
    else:
        return score / len(y_pred)


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]
    true_pos = 0.0
    fp_and_tp = 0.0
    false_pos = 0.0
    # traverse through true and pre
    for true_y, predicted_y in zip(y_true, y_pred):
        if true_y in labels and predicted_y in labels:
            if true_y == predicted_y and predicted_y == pos_label:
                true_pos += 1  # add up accuracy
            if true_y != predicted_y and predicted_y == pos_label:
                false_pos += 1  # add up accuracy
    fp_and_tp = true_pos + false_pos
    if true_pos == 0 and fp_and_tp == 0:
        precision = 0.0
    else:
        precision = true_pos/fp_and_tp
    return precision


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]
    true_pos = 0.0
    fn_and_tp = 0.0
    false_neg = 0.0
    # traverse through true and pre
    for true_y, predicted_y in zip(y_true, y_pred):
        if true_y in labels and predicted_y in labels:
            if true_y == predicted_y and predicted_y == pos_label:
                true_pos += 1  # add up accuracy
            if predicted_y not in (true_y, pos_label):
                false_neg += 1  # add up accuracy
    fn_and_tp = true_pos + false_neg
    if true_pos == 0 and fn_and_tp == 0:
        recall = 0.0
    else:
        recall = true_pos/fn_and_tp
    return recall


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    if labels is None:
        labels = list(set(y_true))  # get labels
    if pos_label is None:
        pos_label = labels[0]  # set to first
    # get precision
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    # get recall
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    if precision + recall == 0:  # if zero then f_one is zero
        f_one = 0.0
    else:  # else use above formula
        f_one = 2 * (precision * recall) / (precision + recall)
    return f_one
