import numpy as np


def calDist(x, y):
    return np.sum((x - y) ** 2)


def k_nearest_neighbors(p, points, k):
    """
    Finds the k nearest neighbors of point `p` in the dataset `points` using the specified distance metric.

    Parameters:
    - p (list): The point (e.g., [x1, y1, z1, ...]).
    - points (list of lists): The set of points (e.g., [[x1, y1, z1], [x2, y2, z2], ...]).
    - k (int): The number of nearest neighbors to find.

    Returns:
    - list: A list of the k nearest neighbors (each neighbor is a list representing a point)
            and their index in `points`.
    """
    dis_arr = [calDist(p, point) for point in points]
    sorted_args = np.argsort(dis_arr)
    return [points[i] for i in sorted_args[:k]], sorted_args[:k]


def knn_predict(X_train, y_train, testPoint, k):
    """
    Predicts labels for validation data using the K-Nearest Neighbors algorithm.

    Args:
      X_train: Training data features (numpy array).
      y_train: Training data labels (numpy array).
      testPoint: Validation or test data sample (numpy array).
      k: Number of nearest neighbors to consider.

    Returns:
      predictions: List of predicted labels for the validation data.
    """
    predictions = []
    for t in testPoint:
        _, b = k_nearest_neighbors(t, X_train, k)
        labels = [y_train[i] for i in b]
        predictions.append(max(set(labels), key=labels.count))
    return predictions
