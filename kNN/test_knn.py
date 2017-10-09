"""
Test knn algorithm
"""
import knn


GROUPS, LABELS = knn.create_dataset()

if __name__ == '__main__':
    knn.classify([0, 0], GROUPS, LABELS, 3)
