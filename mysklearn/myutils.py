import numpy as np
import csv
import math

####################################################################
# NAIVE BAYES CLASSIFIER FUNCTIONS #
####################################################################
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
####################################################################
                
              
####################################################################
# DECISION TREE CLASSIFIER FUNCTIONS #
####################################################################
def select_attribute(instances, attributes):
    select_min_entropy = []
    for i in attributes:
        attribute_types = []
        # find all attribute instance types
        for row in instances:
            if row[i] not in attribute_types:
                attribute_types.append(row[i])
        attribute_instances = [[] for _ in attribute_types]
        # find amount for each attribute
        for row in instances:
            index_att = attribute_types.index(row[i])
            attribute_instances[index_att].append(1)

        class_types = []
        for values in instances:
            if values[-1] not in class_types:
                class_types.append(values[-1])
        class_type_check = [[[] for _ in class_types] for _ in attribute_types]

        for j, _ in enumerate(instances):
            class_type_check[attribute_types.index(
                instances[j][i])][class_types.index(instances[j][-1])].append(1)

        enew = 0
        for entropy_att, _ in enumerate(class_type_check):
            entropy = 0
            for class_entropy in range(len(class_type_check[entropy_att])):
                val_instance = sum(
                    class_type_check[entropy_att][class_entropy])
                einstance = val_instance / \
                    sum(attribute_instances[entropy_att])
                if einstance != 0:
                    entropy += -1 * einstance * math.log(einstance, 2)
            enew += entropy * \
                sum(attribute_instances[entropy_att]) / len(instances)
        select_min_entropy.append(enew)

    min_index = select_min_entropy.index(min(select_min_entropy))
    return attributes[min_index]


def partition_instances(instances, split_attribute, X_train):
    attribute_domains = {}
    for l, _ in enumerate(X_train[0]):
        no_repeats = []
        for row in X_train:
            if str(row[l]) not in no_repeats:
                no_repeats.append(str(row[l]))
        attribute_domains[l] = no_repeats
    partitions = {}
    att_index = split_attribute
    att_domain = attribute_domains[att_index]
    for att_value in att_domain:
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)
    return partitions


def all_same_class(instances):
    check_same = instances[0][-1]
    for attribute_vals in instances:
        if attribute_vals[-1] != check_same:
            return False
    return True


def majority_vote(att_partition, current_instances, value_subtree, tree):
    classifiers = []
    for value_class in att_partition:
        if value_class[-1] not in classifiers:
            classifiers.append(value_class[-1])

    find_majority = [[] for _ in classifiers]
    for value_class in att_partition:
        find_majority[classifiers.index(value_class[-1])].append(1)

    max = 0
    for count in find_majority:
        total_sum = sum(count)
        if total_sum > max:
            majority_rule = classifiers[find_majority.index(count)]

    leaf_node = ["Leaf", majority_rule, len(
        att_partition), len(current_instances)]
    value_subtree.append(leaf_node)
    tree.append(value_subtree)


def tdidt(current_instances, available_attributes, X_train):
    attribute = select_attribute(current_instances, available_attributes)
    available_attributes.remove(attribute)
    tree = ["Attribute", "att" + str(attribute)]
    partitions = partition_instances(current_instances, attribute, X_train)
    for att_value, att_partition in partitions.items():
        value_subtree = ["Value", att_value]

        if len(att_partition) > 0 and all_same_class(att_partition):
            leaf_node = ["Leaf", att_partition[0][-1],
                         len(att_partition), len(current_instances)]
            value_subtree.append(leaf_node)
            tree.append(value_subtree)

        elif len(att_partition) > 0 and len(available_attributes) == 0:
            majority_vote(att_partition, current_instances,
                          value_subtree, tree)

        elif len(att_partition) == 0:
            return None

        else:
            subtree = tdidt(
                att_partition, available_attributes.copy(), X_train)
            if subtree is None:
                majority_vote(att_partition, current_instances,
                              value_subtree, tree)
            else:
                value_subtree.append(subtree)
                tree.append(value_subtree)
    return tree


def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Leaf":
        return tree[1]
    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        values = tree[i]
        if values[1] == instance[att_index]:
            return tdidt_predict(header, values[2], instance)
####################################################################


####################################################################
# K NEAREST NEIGHBOR CLASSIFIER FUNCTIONS #
####################################################################
def normalize_data(col):
    '''normalizes the given column of date with numerical attributes to values [0,1]
    '''
    col = [(value-min(col)) / (max(col)-min(col))for value in col]
    return col

def compute_euclidean_distance(v1, v2):
    '''compute the euclidean distance of the two values

    Args:
        v1: a value of one or more dimension.
        v2: a value of one or more dimension.
    Returns:
        the euclidean distance calculated between v1 and v2.
    '''
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def get_frequencies(column):
    '''count the number of time a values occurs in the column
        Args:
            column (list): data source to used to count unique value occurences
        Returns:
            values (list): parallel list for the unique values
            counts (list): parallel list for the occurences
    '''
    col = sorted(column)

    values = []
    counts = []
    for value in col:
        if value not in values:
            values.append(value)
            counts.append(1)
        else:
            counts[-1] += 1
    return values, counts
####################################################################



