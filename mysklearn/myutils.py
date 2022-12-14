import numpy as np


def randomize_in_place(alist, parallel_list=None, ran_seed=None):
    for i in range(len(alist)):
        if ran_seed != None:
            np.random.seed(ran_seed)
        rand_index = np.random.randint(0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] =\
                parallel_list[rand_index], parallel_list[i]


def normalize(X_train, X_test):
    for i in range(len(X_train[0])):
        maximum = max([l[i] for l in X_train])
        minimum = min([l[i] for l in X_train])
        for j in range(len(X_train)):
            X_train[j][i] = (X_train[j][i] - minimum)/maximum
        for h in range(len(X_test)):
            X_test[h][i] = (X_test[h][i] - minimum)/maximum
    return X_train, X_test


def x_train_helper(X_train, data_array):
    for row in X_train:
        for i, j in enumerate(row):
            if i >= len(data_array):
                data_array.append({})
            if j not in data_array[i]:
                data_array[i][j] = 0
