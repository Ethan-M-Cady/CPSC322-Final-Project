import copy
from mysklearn import myutils


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
