import numpy as np
import csv
import math
import itertools
from tabulate import tabulate


####################################################################
# SEAN NAVE UTILITY FUNCTIONS #
####################################################################


def group_by(table: list, group_by_col_index):
    '''return a list of unique names and subtables from the original
        table that contains unique name.

        Args:
            table (list): the data source for grouping.
            group_by_col_name: the name of the columns for the table to be grouped by.
        Returns:
            group_names (list): A list of each unique names
            group_subtables (list): a list of tables that contains the unique names.
    '''
    labels = [row[group_by_col_index] for row in table]
    group_names = sorted(list(set(labels)))
    group_subtables = [[] for _ in group_names]

    for row in table:
        group_by_val = row[group_by_col_index]
        subtable_index = group_names.index(group_by_val)
        group_subtables[subtable_index].append(row)
    return group_names, group_subtables


def get_column(table: list, index):
    col = [row[index] for row in table]
    return col


def compute_slope_intercept(x_list, y_list):
    '''compute the slope and y-intercept for the best fit line
        of the given data

        Args:
            x_list (list): data for the horizontal values.
            y_list (list): data for the vertical values.
        Returns:
            slope (int): the slope of the best fit line
            intercept (int): the y intercept of the best fit line
    '''
    meanx = np.mean(x_list)
    meany = np.mean(y_list)

    slope = sum([(x_list[i] - meanx) * (y_list[i] - meany) for i in range(len(x_list))]) / \
        sum([(x_list[i] - meanx) ** 2 for i in range(len(x_list))])

    # y = mx + b => b = y - mx
    intercept = meany - slope * meanx
    return slope, intercept


def calculate_covariance_and_coeff(x_list, y_list):
    '''compute the covariance and correlation coefficient for the best fit line
        of the given data

        Args:
            x_list (list): data for the horizontal values.
            y_list (list): data for the vertical values.
        Returns:
            cov (int): the covariance of the best fit line
            coeff (int): the correlation coefficient of the best fit line
    '''
    meanx = np.mean(x_list)
    meany = np.mean(y_list)

    cov = sum([(x_list[i] - meanx) * (y_list[i] - meany)
              for i in range(len(x_list))]) / len(x_list)
    coeff = cov / (np.std(x_list) * np.std(y_list))
    return cov, coeff


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
def select_attribute(table, attributes):  # check
    min_entropy = []
    for i in attributes:
        attribute_labels = []
        for row in table:
            if row[i] not in attribute_labels:
                attribute_labels.append(row[i])
        attribute_rows = []
        for item in attribute_labels:
            attribute_rows.append([])
        for row in table:
            index_att = attribute_labels.index(row[i])
            attribute_rows[index_att].append(1)
        class_types = []
        for values in table:
            if values[-1] not in class_types:
                class_types.append(values[-1])
        class_type_search = [[[] for _ in class_types]
                             for _ in attribute_labels]
        for j, _ in enumerate(table):
            class_type_search[attribute_labels.index(
                table[j][i])][class_types.index(table[j][-1])].append(1)
        E_new = 0
        for entropy_attribute, _ in enumerate(class_type_search):
            entropy = 0
            for class_entropy in range(len(class_type_search[entropy_attribute])):
                val_instance = sum(
                    class_type_search[entropy_attribute][class_entropy])
                entropy = val_instance / \
                    sum(attribute_rows[entropy_attribute])
                if entropy != 0:
                    entropy += -1 * entropy * math.log(entropy, 2)
            E_new += entropy * \
                sum(attribute_rows[entropy_attribute]) / len(table)
        min_entropy.append(E_new)

    min_index = min_entropy.index(min(min_entropy))
    return attributes[min_index]


def partition_instances(instances, split_attribute, X_train):
    attribute_domains = {}
    for i, _ in enumerate(X_train[0]):
        unique_vals = []
        for row in X_train:
            if str(row[i]) not in unique_vals:
                unique_vals.append(str(row[i]))
        attribute_domains[i] = unique_vals
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


def majority_vote(att_partition, current_instances, value_subtree, tree):  # still check
    classifiers = []
    for value_class in att_partition:
        if value_class[-1] not in classifiers:
            classifiers.append(value_class[-1])

    find_majority = [[] for _ in classifiers]
    for value_class in att_partition:
        find_majority[classifiers.index(value_class[-1])].append(1)

    max_val = 0
    for count in find_majority:
        total_sum = sum(count)
        if total_sum > max_val:
            majority_rule = classifiers[find_majority.index(count)]

    leaf_node = ["Leaf", majority_rule, len(
        att_partition), len(current_instances)]
    value_subtree.append(leaf_node)
    tree.append(value_subtree)


def tdidt(current_instances, available_attributes, X_train):  # done#
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
            available_attributes_shallow = available_attributes.copy()
            subtree = tdidt(
                att_partition, available_attributes_shallow, X_train)
            if subtree is None:
                majority_vote(att_partition, current_instances,
                              value_subtree, tree)
            else:
                value_subtree.append(subtree)
                tree.append(value_subtree)
    return tree


def tdidt_predict(header, tree, instance):
    if tree[0] == "Leaf":
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


def prepend_attribute_label(table, header):
    for row in table:
        for i in range(len(row)):
            row[i] = header[i] + "=" + str(row[i])


def count_in_table(rule, table):
    count = 0
    for row in table:
        count_to_len = 0
        for item in row:
            if item in rule:
                count_to_len += 1
                if count_to_len == len(rule):
                    count += 1
                    break
    return count


def compute_rule_support(rule, table, minsup):
    Nboth = 0
    for row in table:
        j = 0
        for i in range(len(rule)):
            if rule[i] in row:
                j += 1
            else:
                break
        if j == len(rule):
            Nboth += 1
    if(Nboth / len(table)) >= minsup:
        return True
    else:
        return False


def make_Lnew(Ccurr, L_prev, k, table, minsup):
    L_new = []
    for i in range(0, len(Ccurr)):
        arr_count = 0
        subset_C = []
        subset_C.extend(itertools.combinations(Ccurr[i], k-1))
        for subset_C_item in subset_C:
            for L_prev_item in L_prev:
                if list(subset_C_item) == list(L_prev_item):
                    arr_count += 1
                    break
        if len(Ccurr[i]) == arr_count and compute_rule_support(Ccurr[i], table, minsup) is True:
            L_new.append(Ccurr[i])
    return L_new


def union(arr1, arr2):
    main_arr = arr1 + arr2
    seen_vars = []
    unique_val_arr = []
    i = len(main_arr)-1
    while i > -1:
        if seen_vars.count(main_arr[i]) == 0:
            unique_val_arr.append(main_arr[i])
            seen_vars.append(main_arr[i])
        i -= 1
    return sorted(unique_val_arr)


def unique_vals(arr):
    i = len(arr)-1
    while i > -1:
        if arr.count(arr[i]) > 1:
            del arr[i]
        i -= 1
    return arr


def make_Cnew(L_old):
    C_new = []
    L_old_copy = L_old
    for i in range(0, len(L_old)):
        for j in range(0, len(L_old_copy)):
            if L_old[i][:-1] == L_old_copy[j][:-1] and L_old[i] != L_old_copy[j]:
                C_new.append(union(L_old[i], L_old_copy[j]))
    return unique_vals(C_new)


def make_Cinit_and_Linit(_table, minsup):
    unique_items = []
    for row in _table:
        for item in row:
            if unique_items.count(item) == 0:
                unique_items.append(item)
    unique_items = sorted(unique_items)

    # making it into a list of lists
    Cinit = []
    for item in unique_items:
        new_list = item.split(', ')
        Cinit.append(new_list)

    # checking support of each item
    i = len(Cinit)-1
    Linit = Cinit.copy()
    while i > -1:
        if compute_rule_support(Linit[i], _table, minsup) is False:
            del Linit[i]
        i -= 1
    return Cinit, Linit


def join_supported_itemsets(L, k):
    total_union_set = []
    for i in range(1, k-2-1):
        total_union_set.extend(union(L[i], L[i+1]))
    return total_union_set


def difference(list1, list2):
    i = len(list2)-1
    list2_copy = list2.copy()
    while i > -1:
        if list1.count(list2_copy[i]) > 0:
            del list2_copy[i]
        i -= 1
    return list2_copy


def generate_rules(table, supported_itemset, minconf):
    confidence, rules, LHS, RHS = [], [], [], []
    j = 0
    for item_set in supported_itemset:
        for i in range(1, len(item_set)):
            _list = list(itertools.combinations(item_set, i))
            for list_item in _list:
                LHS.append(list((list_item)))
                RHS.append(difference(list_item, item_set))
    for i in range(0, len(LHS)):
        Nboth = count_in_table(union(LHS[i], RHS[i]), table)
        NLeft = count_in_table(LHS[i], table)
        confidence.append(Nboth/NLeft)
        # if confidence[-1] >= 0.8:
        if confidence[-1] >= minconf:
            rules.append([])
            rules[j].append(LHS[i])
            rules[j].append(RHS[i])
            j += 1
    return unique_vals(rules)  # this makes sure all of them are unique


def compute_stats(rule, table):
    Nleft, Nright, Nboth = 0, 0, 0
    Ntotal = len(table)

    left_target = rule[0]
    right_target = rule[1]
    both_target = union(rule[0], rule[1])

    for row in table:
        j = 0
        for i in range(0, len(left_target)):
            if left_target[i] in row:
                j += 1
            if j == len(left_target):
                Nleft += 1

        j = 0
        for i in range(0, len(right_target)):
            if right_target[i] in row:
                j += 1
            if j == len(right_target):
                Nright += 1

        j = 0
        for i in range(0, len(both_target)):
            if both_target[i] in row:
                j += 1
            if j == len(both_target):
                Nboth += 1
    if Nleft == 0:
        confidence = 0
    else:
        confidence = Nboth / Nleft

    support = Nboth / Ntotal

    if Nright == 0:
        lift = 0
    else:
        lift = Nboth / Nright

    return confidence, support, lift


def pretty_print_rules(rules, table):
    string_rules = []
    for i in range(0, len(rules)):
        string_rules.append([])
    i = 0
    for item in rules:
        string_rules[i] = "IF "
        for sub_item in item:
            if sub_item == item[-1]:
                string_rules[i] = (string_rules[i]) + " THEN "
            for sub_sub_item in sub_item:
                string_rules[i] = (string_rules[i]) + sub_sub_item
                if sub_sub_item != sub_item[-1]:
                    string_rules[i] = (string_rules[i]) + " AND "
        i += 1

    new_table = []
    for i in range(0, len(rules)):
        confidence, support, lift = compute_stats(rules[i], table)
        new_table.append(
            [str(i+1), string_rules[i], support, confidence, lift])

    print(tabulate(new_table, headers=[
          "#", "Rule", "Support", "Confidence", "Lift"]))


def apriori(table, minsup, minconf):
    C, L = [], []
    Cinit, Linit = make_Cinit_and_Linit(table, minsup)
    C.append(Cinit)
    L.append(Linit)
    k = 2
    keep_going = True
    for i in range(0, 999999):
        if len(L[-1]) == 0:  # if last element is empty set
            L.pop()
            break
        C.append(make_Cnew(L[i]))
        L.append(make_Lnew(C[i+1], L[-1], k, table, minsup))
        k += 1

    all_supported_itemsets = join_supported_itemsets(L, k)
    rules = generate_rules(table, all_supported_itemsets, minconf)
    return(rules)
