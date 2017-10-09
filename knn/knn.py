"""
k-Nearest Neighbors
"""

import operator
from numpy import array, tile


def create_dataset():
    """
    Generate dataset
    """
    groups = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return groups, labels


def classify(data, dataset, labels, k):
    """
    give label to new data, learning from dataset and labels
    """
    # step 1. distance calculation
    dataset_size = dataset.shape[0]
    matrix_difference = tile(data, (dataset_size, 1)) - dataset
    sq_matrix_distance = matrix_difference ** 2
    sq_distance = sq_matrix_distance.sum(axis=1)
    distance = sq_distance ** 0.5

    sorted_distance_indicies = distance.argsort()
    class_count = {}

    # step 2. voting with lowest k distances
    for i in range(k):
        vote_i_label = labels[sorted_distance_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1

    # step 3. sort dictionary
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]