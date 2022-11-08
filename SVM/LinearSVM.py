import numpy as np


class LinearSVM(object):
    def __init__(self, dataset_size, vector_size):
        """
        :param dataset_size: the number of dataset for initializing Lagrange Multipliers
        :param vector_size: dimmension of X for initializing W
        """
        self.__multipliers = np.zeros(dataset_size, np.float_)
        self.weight_vec = np.zeros(vector_size, np.float_)
        self.bias = 0

    def train(self, dataset, iteration_num):
        """
        :param dataset: train dataset for training SVM
        :param iteration_num: train epoch
        """
        dataset = np.array(dataset)
        for k in range(iteration_num):
            self.__update(dataset, k)

    def __update(self, dataset, k):
        for i in range(dataset.__len__() // 2):
            j = (dataset.__len__() // 2 + i + k) % dataset.__len__()
            X = dataset[i]
            Y = dataset[j]
            self.__sequential_minimal_optimization(dataset, X, Y, i, j)
            self.__update_weight_vec(dataset)
            self.__update_bias(dataset)

    def __sequential_minimal_optimization(self, train_dataset, X, Y, i, j):
        """
        :param train_dataset: train dataset for SVM
        :param X: sample1
        :param Y: sample2
        :param i: index of sample1
        :param j: index of sample2
        :return:
        """
        label_X = X[:-1]
        label_Y = Y[:-1]
        featureVec_X = np.array(X[0])
        featureVec_Y = np.array(Y[0])
        error_X = np.dot(self.weight_vec, featureVec_X) + self.bias - featureVec_X
        error_Y = np.dot(self.weight_vec, featureVec_Y) + self.bias - featureVec_Y
        eta = np.dot(featureVec_X - featureVec_Y, featureVec_X - featureVec_Y)
        unclipped_i = self.__multipliers[i] + featureVec_X * (error_Y - error_X) / eta

        constant = -self.__calculate_constant(train_dataset, i, j)
        multiplier = self.__quadratic_programming(unclipped_i, label_X, label_Y, i, j)
        if multiplier >= 0:
            self.__multipliers[i] = multiplier
        self.__multipliers[j] = (constant - multiplier * label_X) * label_Y

    def __update_bias(self, dataset):

        sum_bias = 0
        count = 0
        for k in range(self.__multipliers.__len__()):
            if self.__multipliers[k] != 0:
                label = dataset[k][-1]
                vector = np.array(dataset[k][0])
                sum_bias += 1 / label - np.dot(self.weight_vec, vector)
                count += 1
        if count == 0:
            self.bias = 0
        else:
            self.bias = sum_bias / count

    def __update_weight_vec(self, dataset):
        weight_vector = np.zeros(dataset[0][0].__len__())
        for k in range(dataset.__len__()):
            label = dataset[k][-1]
            vector = np.array(dataset[k][0])
            weight_vector += self.__multipliers[k] * label * vector
        self.weight_vec = weight_vector

    def __calculate_constant(self, dataset, i, j):
        label_i = dataset[i][-1]
        label_j = dataset[j][-1]
        dataset[i][-1] = 0
        dataset[j][-1] = 0
        sum_constant = 0
        for k in range(dataset.__len__()):
            label = dataset[k][-1]
            sum_constant += self.__multipliers[k] * label
        dataset[i][-1] = label_i
        dataset[j][-1] = label_j
        return sum_constant

    def __quadratic_programming(self, unclipped_i, label_i, label_j, i, j):
        multiplier = -1
        if label_i * label_j == 1:
            boundary = self.__multipliers[i] + self.__multipliers[j]
            if boundary >= 0:
                if unclipped_i <= 0:
                    multiplier = 0
                elif unclipped_i < boundary:
                    multiplier = unclipped_i
                else:
                    multiplier = boundary
        else:
            boundary = max(0, self.__multipliers[i] - self.__multipliers[j])
            if unclipped_i <= boundary:
                multiplier = boundary
            else:
                multiplier = unclipped_i
        return multiplier

    def predict(self, vector):
        result = np.dot(self.weight_vec, np.array(vector)) + self.bias
        if result >= 0:
            return 1
        else:
            return -1

    def __str__(self):
        return "multipliers:" + self.__multipliers.__str__() + '\n' + \
               "weight_vector:" + self.weight_vec.__str__() + '\n' + \
               "bias:" + self.bias.__str__()
