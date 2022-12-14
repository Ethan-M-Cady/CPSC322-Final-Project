from mysklearn.myclassifiers import MyKNeighborsClassifier
import numpy as np
from mysklearn.myclassifiers import MyNaiveBayesClassifier
from mysklearn import myevaluation, myutils
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.myclassifiers import MyAssociationRuleMiner

##################################################################
# NAIVE BAYES TESTS
##################################################################
# in-class Naive Bayes example (lab task #1)
header_inclass_example = ["att1", "att2"]
X_train_inclass_example = [
    [1, 5],  # yes
    [2, 6],  # yes
    [1, 5],  # no
    [1, 5],  # no
    [1, 6],  # yes
    [2, 6],  # no
    [1, 5],  # yes
    [1, 6]  # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

class_test = [
    [1, 5]
]
class_true = ["yes"]
class_priors_solutions = {'no': 0.375, 'yes': 0.625}
class_posteriors_solutions = {'no': [{1: 0.6666666666666666, 2: 0.3333333333333333}, {
    5: 0.6666666666666666, 6: 0.3333333333333333}], 'yes': [{1: 0.8, 2: 0.2}, {5: 0.4, 6: 0.6}]}


# RQ5 (fake) iPhone purchases dataset
header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
X_train_iphone = [
    [1, 3, "fair", "no"],
    [1, 3, "excellent", "no"],
    [2, 3, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [2, 1, "fair", "yes"],
    [2, 1, "excellent", "no"],
    [2, 1, "excellent", "yes"],
    [1, 2, "fair", "no"],
    [1, 1, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [1, 2, "excellent", "yes"],
    [2, 2, "excellent", "yes"],
    [2, 3, "fair", "yes"],
    [2, 2, "excellent", "no"],
    [2, 3, "fair", "yes"]

]
y_train_iphone = ["no", "no", "yes", "yes", "yes", "no",
                  "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
iphone_table_priors = {'no': 0.3333333333333333, 'yes': 0.6666666666666666}
iphone_table_posteriors = {'no': [{1: 0.6, 2: 0.4}, {1: 0.2, 2: 0.4, 3: 0.4}, {'excellent': 0.6, 'fair': 0.4}], 'yes': [
    {1: 0.2, 2: 0.8}, {1: 0.3, 2: 0.4, 3: 0.3}, {'excellent': 0.3, 'fair': 0.7}]}
iphone_test = [
    [2, 2, "fair"],
    [1, 1, "excellent"]
]
iphone_true = ["yes", "no"]

# Bramer 3.2 train dataset
header_train = ["day", "season", "wind", "rain", "class"]
X_train_train = [
    ["weekday", "spring", "none", "none", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "high", "heavy", "late"],
    ["saturday", "summer", "normal", "none", "on time"],
    ["weekday", "autumn", "normal", "none", "very late"],
    ["holiday", "summer", "high", "slight", "on time"],
    ["sunday", "summer", "normal", "none", "on time"],
    ["weekday", "winter", "high", "heavy", "very late"],
    ["weekday", "summer", "none", "slight", "on time"],
    ["saturday", "spring", "high", "heavy", "cancelled"],
    ["weekday", "summer", "high", "slight", "on time"],
    ["saturday", "winter", "normal", "none", "late"],
    ["weekday", "summer", "high", "none", "on time"],
    ["weekday", "winter", "normal", "heavy", "very late"],
    ["saturday", "autumn", "high", "slight", "on time"],
    ["weekday", "autumn", "none", "heavy", "on time"],
    ["holiday", "spring", "normal", "slight", "on time"],
    ["weekday", "spring", "normal", "none", "on time"],
    ["weekday", "spring", "normal", "slight", "on time"]
]
y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                 "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                 "very late", "on time", "on time", "on time", "on time", "on time"]
train_priors = {'cancelled': 0.05, 'late': 0.1,
                'on time': 0.7, 'very late': 0.15}

train_posteriors = {
    'on time': [{'weekday': 0.6428571428571429, 'saturday': 0.14285714285714285, 'holiday': 0.14285714285714285, 'sunday': 0.07142857142857142}, {'spring': 0.2857142857142857, 'winter': 0.14285714285714285, 'summer': 0.42857142857142855, 'autumn': 0.14285714285714285}, {'none': 0.35714285714285715, 'high': 0.2857142857142857, 'normal': 0.35714285714285715}, {'none': 0.35714285714285715, 'slight': 0.5714285714285714, 'heavy': 0.07142857142857142}],
    'late': [{'weekday': 0.5, 'saturday': 0.5, 'holiday': 0.0, 'sunday': 0.0}, {'spring': 0.0, 'winter': 1.0, 'summer': 0.0, 'autumn': 0.0}, {'none': 0.0, 'high': 0.5, 'normal': 0.5}, {'none': 0.5, 'slight': 0.0, 'heavy': 0.5}],
    'very late': [{'weekday': 1.0, 'saturday': 0.0, 'holiday': 0.0, 'sunday': 0.0}, {'spring': 0.0, 'winter': 0.6666666666666666, 'summer': 0.0, 'autumn': 0.3333333333333333}, {'none': 0.0, 'high': 0.3333333333333333, 'normal': 0.6666666666666666}, {'none': 0.3333333333333333, 'slight': 0.0, 'heavy': 0.6666666666666666}],
    'cancelled': [{'weekday': 0.0, 'saturday': 1.0, 'holiday': 0.0, 'sunday': 0.0}, {'spring': 1.0, 'winter': 0.0, 'summer': 0.0, 'autumn': 0.0}, {'none': 0.0, 'high': 1.0, 'normal': 0.0}, {'none': 0.0, 'slight': 0.0, 'heavy': 1.0}]
}

train_test = [
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "high", "heavy"],
    ["sunday", "summer", "normal", "sligh"]
]
train_true = ["very late", "on time", "on time"]


def test_naive_bayes_classifier_fit():
    test_fit = MyNaiveBayesClassifier()
    test_fit.fit(X_train_inclass_example, y_train_inclass_example)

    assert class_priors_solutions == test_fit.priors
    assert class_posteriors_solutions == test_fit.posteriors

    x_train = [x[0:3] for x in X_train_iphone]
    y_train = [x[-1] for x in X_train_iphone]
    test_fit.fit(x_train, y_train)

    assert iphone_table_priors == test_fit.priors
    assert iphone_table_posteriors == test_fit.posteriors

    x_train = [x[0:4] for x in X_train_train]
    y_train = [x[-1] for x in X_train_train]
    test_fit.fit(x_train, y_train)

    assert train_priors == test_fit.priors
    assert train_posteriors == test_fit.posteriors


def test_naive_bayes_classifier_predict():
    test_predict = MyNaiveBayesClassifier()
    test_predict.fit(X_train_inclass_example, y_train_inclass_example)
    class_predicted = test_predict.predict(class_test)
    assert class_true == class_predicted

    x_train = [x[0:3] for x in X_train_iphone]
    y_train = [x[-1] for x in X_train_iphone]
    test_predict.fit(x_train, y_train)
    iphone_predicted = test_predict.predict(iphone_test)

    assert iphone_true == iphone_predicted

    x_train = [x[0:4] for x in X_train_train]
    y_train = [x[-1] for x in X_train_train]
    test_predict.fit(x_train, y_train)
    train_predicted = test_predict.predict(train_test)

    assert train_true == train_predicted


##################################################################
# KNN TESTS #
##################################################################


def high_low_discretizer(value):
    if value <= 100:
        return "low"
    return "high"

# note: order is actual/received student value, expected/solution


def test_kneighbors_classifier_kneighbors():
    knn_clf = MyKNeighborsClassifier(n_neighbors=3)
    test_instance = [2, 3]

    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    knn_clf.fit(X_train=X_train_class_example1, y_train=y_train_class_example1)
    distances, indexes = knn_clf.kneighbors([test_instance])
    distanct_solution = [2.23606797749979,
                         3.1622776601683795, 3.433496759864497]
    indexes_solution = [[0, 1, 2]]
    assert np.allclose(distances, distanct_solution)
    assert np.allclose(indexes, indexes_solution)

    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    y_train = ["no", "yes", "no", "no", "yes",
               "no", "yes", "yes"]  # parallel to X_train
    knn_clf.fit(X_train=X_train, y_train=y_train)
    distances, indexes = knn_clf.kneighbors([test_instance])
    distanct_solution = [[1.41421356, 1.41421356, 2]]
    indexes_solution = [[0, 4, 6]]
    assert np.allclose(distances, distanct_solution)
    assert np.allclose(indexes, indexes_solution)

    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",
                              "-", "-", "+", "+", "+", "-", "+"]
    knn_clf.fit(X_train=X_train_bramer_example, y_train=y_train_bramer_example)
    distances, indexes = knn_clf.kneighbors([test_instance])
    distanct_solution = [[3.5114099732158874,
                          4.401136216933078, 5.1351728305871065]]
    indexes_solution = [[0, 2, 1]]
    assert np.allclose(distances, distanct_solution)
    assert np.allclose(indexes, indexes_solution)


def test_kneighbors_classifier_predict():
    knn_clf = MyKNeighborsClassifier(n_neighbors=3)
    test_instance = [2, 3]

    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    knn_clf.fit(X_train=X_train_class_example1, y_train=y_train_class_example1)
    knn_clf.kneighbors([test_instance])
    y_predict_solution = ["bad"]
    y_predict = knn_clf.predict([test_instance])
    assert y_predict == y_predict_solution

    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    y_train = ["no", "yes", "no", "no", "yes",
               "no", "yes", "yes"]  # parallel to X_train
    knn_clf.fit(X_train=X_train, y_train=y_train)
    y_predict_solution = ["yes"]
    y_predict = knn_clf.predict([test_instance])
    assert y_predict == y_predict_solution

    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",
                              "-", "-", "+", "+", "+", "-", "+"]
    knn_clf.fit(X_train=X_train_bramer_example, y_train=y_train_bramer_example)
    knn_clf.kneighbors([test_instance])
    y_predict_solution = ["-"]
    y_predict = knn_clf.predict([test_instance])
    assert y_predict == y_predict_solution

##################################################################
# DECISION TREE CLASSIFIER TESTS #
##################################################################


def test_decision_tree_classifier_fit():
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    # interview dataset
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True",
                         "False", "True", "True", "True", "True", "True", "False"]
    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_interview, y_train_interview)

    # attribute values are now sorted by index
    tree_interview = \
        ['Attribute', 'att0',
         ['Value', 'Senior',
          ['Attribute', 'att2',
           ['Value', 'no',
            ['Leaf', 'False', 3, 5]],
           ['Value', 'yes',
            ['Leaf', 'True', 2, 5]]]],
         ['Value', 'Mid',
          ['Leaf', 'True', 4, 14]],
         ['Value', 'Junior',
          ['Attribute', 'att3',
           ['Value', 'no',
            ['Leaf', 'True', 3, 5]],
           ['Value', 'yes',
            ['Leaf', 'False', 2, 5]]]]]
    assert tree_interview == decision_tree.tree

    X_train_iphone = [
        ["1", "3", "fair"],
        ["1", "3", "excellent"],
        ["2", "3", "fair"],
        ["2", "2", "fair"],
        ["2", "1", "fair"],
        ["2", "1", "excellent"],
        ["2", "1", "excellent"],
        ["1", "2", "fair"],
        ["1", "1", "fair"],
        ["2", "2", "fair"],
        ["1", "2", "excellent"],
        ["2", "2", "excellent"],
        ["2", "3", "fair"],
        ["2", "2", "excellent"],
        ["2", "3", "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no",
                      "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_iphone, y_train_iphone)

    tree_iphone = \
        ['Attribute', 'att0',
            ['Value', '1',
                ['Attribute', 'att1',
                    ['Value', '3',
                        ['Leaf', 'no', 2, 5]],
                    ['Value', '2',
                        ['Attribute', 'att2',
                            ['Value', 'fair',
                                ['Leaf', 'no', 1, 2]],
                            ['Value', 'excellent',
                                ['Leaf', 'yes', 1, 2]]]],
                    ['Value', '1',
                        ['Leaf', 'yes', 1, 5]]]],
            ['Value', '2',
                ['Attribute', 'att2',
                    ['Value', 'fair',
                        ['Leaf', 'yes', 6, 10]],
                    ['Value', 'excellent',
                        ['Leaf', 'no', 4, 10]]]]]
    assert tree_iphone == decision_tree.tree


def test_decision_tree_classifier_predict():
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    # interview dataset
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True",
                         "False", "True", "True", "True", "True", "True", "False"]
    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_interview, y_train_interview)
    X_test = [["Junior", "Java", "yes", "no"],
              ["Junior", "Java", "yes", "yes"]]
    expected_predict = ["True", "False"]
    predicted = decision_tree.predict(X_test)
    assert expected_predict == predicted

    X_train_iphone = [
        ["1", "3", "fair"],
        ["1", "3", "excellent"],
        ["2", "3", "fair"],
        ["2", "2", "fair"],
        ["2", "1", "fair"],
        ["2", "1", "excellent"],
        ["2", "1", "excellent"],
        ["1", "2", "fair"],
        ["1", "1", "fair"],
        ["2", "2", "fair"],
        ["1", "2", "excellent"],
        ["2", "2", "excellent"],
        ["2", "3", "fair"],
        ["2", "2", "excellent"],
        ["2", "3", "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no",
                      "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_iphone, y_train_iphone)
    X_test = [["2", "2", "fair"], ["1", "1", "excellent"]]
    expected_predict = ["yes", "yes"]
    predicted = decision_tree.predict(X_test)
    assert expected_predict == predicted


transactions = [
    ["b", "c", "m"],
    ["b", "c", "e", "m", "s"],
    ["b"],
    ["c", "e", "s"],
    ["c"],
    ["b", "c", "s"],
    ["c", "e", "s"],
    ["c", "e"]
]

# interview datasset
header = ["level", "lang", "tweets", "phd", "interviewed_well"]
table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
]


transactions_expected = [
                        [['m'], ['b', 'c']],
                        [['b', 'm'], ['c']],
                        [['c', 'm'], ['b']],
                        [['b', 's'], ['c']],
                        [['m'], ['b']],
                        [['e'], ['c']],
                        [['e', 's'], ['c']],
                        [['m'], ['c']],
                        [['s'], ['c']]
]

table_expected = [
    [['interviewed_well=False'], ['tweets=no']],
    [['level=Mid'], ['interviewed_well=True']],
    [['phd=no', 'tweets=yes'], ['interviewed_well=True']],
    [['tweets=yes'], ['interviewed_well=True']],
    [['lang=R'], ['tweets=yes']]
]

# table_expected

# note: order is actual/received student value, expected/solution
myutils.prepend_attribute_label(table, header)


def test_association_rule_miner_fit():
    test_fit = MyAssociationRuleMiner()

    test_fit.fit(transactions)
    print(test_fit.rules)
    assert len(test_fit.rules) == len(transactions_expected)
    for i in range(0, len(test_fit.rules)):
        assert transactions_expected.count(test_fit.rules[i]) == 1

    test_fit.fit(table)
    assert len(test_fit.rules) == len(table_expected)
    for i in range(0, len(test_fit.rules)):
        assert table_expected.count(test_fit.rules[i]) == 1
