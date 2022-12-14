import copy
from mysklearn import myutils

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """

    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        return_index = []
        return_dist = []
        for instance in X_test:
            row_index_distance = []
            for index, train_value in enumerate(self.X_train):
                dist = myutils.compute_euclidean_distance(
                    train_value, instance)
                row_index_distance.append((index, dist))
            row_index_distance.sort(key=operator.itemgetter(-1))
            top_k = row_index_distance[:self.n_neighbors]

            return_index.append([row[1] for row in top_k])
            return_dist.append([row[0] for row in top_k])

        return return_index, return_dist
class MyNaiveBayesClassifier:
    def __init__(self):
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        self.priors = {}
        self.posteriors = {}
        data_array = []
        for value in y_train:
            if value in self.priors:
                self.priors[value] += 1
            else:
                self.priors[value] = 1
        myutils.x_train_helper(X_train, data_array)
        for value in self.priors:
            self.posteriors[value] = copy.deepcopy(data_array)
        for row, x in zip(X_train, y_train):
            for i, j in enumerate(row):
                self.posteriors[x][i][j] += 1
        for value in self.posteriors:
            for i, row in enumerate(self.posteriors[value]):
                for data in row:
                    self.posteriors[value][i][data] /= self.priors[value]
        for value in self.priors:
            self.priors[value] /= len(y_train)

    def predict(self, X_test):
        y_predicted = []
        for row in X_test:
            probability = {}
            for value in self.posteriors:
                probability[value] = self.priors[value]
                for i, j in enumerate(row):
                    try:
                        probability[value] *= self.posteriors[value][i][j]
                    except KeyError:
                        pass
            max_str_value = ""
            max_val = -1
            for key, value in probability.items():
                if value > max_val:
                    max_str_value = key
                    max_val = value
            y_predicted.append(max_str_value)
        return y_predicted
    
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = list(range(len(train[0])-1))
        self.tree = myutils.tdidt(train, available_attributes, X_train)
        # print(self.tree)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        header = []
        for header_len in range(len(X_test[0])):
            header.append("att" + str(header_len))

        y_predicted = []
        for instance in X_test:
            predicted = myutils.tdidt_predict(header, self.tree, instance)
            y_predicted.append(predicted)

        return y_predicted
