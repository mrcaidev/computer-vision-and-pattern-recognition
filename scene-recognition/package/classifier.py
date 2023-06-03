from collections import Counter
from enum import Enum
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.svm import LinearSVC

__all__ = ["ClassifierType", "knn", "svm"]


class ClassifierType(Enum):
    KNN = "knn"
    SVM = "svm"


def knn(
    train_features: list[np.ndarray],
    train_labels: list[str],
    test_features: list[np.ndarray],
    k: int = 3,
):
    """
    K-Nearest Neighbors classifier.

    Parameters
    ----------
    train_features : The features of the training images.
    train_labels : The labels of the training images.
    test_features : The features of the test images.
    k : The number of nearest neighbors.

    Returns
    -------
    predictions : The predicted labels of the test images.
    """
    # Record the predictions.
    predictions = []

    # For every test feature:
    for test_feature in test_features:
        # Compute the distances between it and each train features.
        distances = [
            euclidean(test_feature, train_feature) for train_feature in train_features
        ]

        # Sort distances et the indices of the nearest neighbors.
        nearest_indices = np.argsort(distances)[:k]

        # Get the labels of the nearest neighbors.
        nearest_labels = np.array(train_labels)[nearest_indices]

        # Find the most common label among the nearest neighbors
        prediction = Counter(nearest_labels).most_common(1)[0][0]

        # Record the prediction.
        predictions.append(prediction)

    return predictions


def svm(
    train_features: list[np.ndarray],
    train_labels: list[str],
    test_features: list[np.ndarray],
):
    """
    Support Vector Machine classifier.

    Parameters
    ----------
    train_features : The features of the training images.
    train_labels : The labels of the training images.
    test_features : The features of the test images.

    Returns
    -------
    predictions : The predicted labels of the test images.
    """
    # Create a LinearSVC classifier.
    classifier = LinearSVC()

    # Train the classifier.
    classifier.fit(train_features, train_labels)

    # Predict the labels.
    predictions = classifier.predict(test_features)

    return list(predictions)
