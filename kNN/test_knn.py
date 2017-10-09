
import knn
import unittest


class KNNTestCase(unittest.TestCase):
    """
    Test knn algorithm
    """
    
    def setUp(self):
        self.groups, self.labels = knn.create_dataset()

    def test_knn_classify(self):
        test_point = [0, 0]
        another_test_point = [1, 0.9]

        classified_label = knn.classify(test_point, self.groups, self.labels, 3)
        another_classified_label = knn.classify(another_test_point, self.groups, self.labels, 3)

        self.assertEqual(classified_label, 'B')
        self.assertEqual(another_classified_label, 'A')


if __name__ == '__main__':
    unittest.main()
