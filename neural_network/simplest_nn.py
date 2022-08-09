import numpy as np


class simplest_nn(object):
    def __init__(self, num_hidden_units, num_outputs, train_set, train_labels, test_set, test_labels):
        """
        :param train_set: numpy array of shape - The feature matrix where each row is a feature vector.
                          Shape is n feature vectors by m features.
        :param train_labels: numpy array - The labels corresponding to each feature vector.
                             Shape is 1 x n feature vectors.
        :param test_set: numpy array of shape - The feature matrix where each row is a feature vector.
                         Shape is n feature vectors by m features.
        :param test_labels: numpy array - The labels corresponding to each feature vector.
                            Shape is 1 x n feature vectors.
        """

        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs
        self.train_set = train_set
        self.train_labels = train_labels
        self.test_set = test_set
        self.test_labels = test_labels
        self.num_inputs = train_set.shape[1]

