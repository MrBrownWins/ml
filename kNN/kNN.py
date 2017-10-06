import operator
from numpy import *


def create_dataset():
    groups = array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return groups, labels
