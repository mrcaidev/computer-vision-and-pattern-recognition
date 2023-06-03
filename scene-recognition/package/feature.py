import os
from enum import Enum
import numpy as np
from scipy.spatial.distance import cdist
from skimage.feature import hog
from skimage.transform import resize
from sklearn.cluster import MiniBatchKMeans

__all__ = ["FeatureType", "get_tiny_image", "build_vocabulary", "get_bag_of_words"]


class FeatureType(Enum):
    TINY_IMAGE = "tiny image"
    BAG_OF_WORDS = "bag of words"


def get_tiny_image(image: np.ndarray):
    """
    Get the tiny version of an image.

    Parameters
    ----------
    image : The input image.

    Returns
    -------
    tiny_image : The tiny image.
    """
    # Resize the image.
    tiny_image = resize(image, (16, 16))

    # Flatten the image.
    tiny_image = tiny_image.flatten()

    # Normalize the image.
    tiny_image = (tiny_image - np.mean(tiny_image)) / np.std(tiny_image)

    return tiny_image


def describe_feature(image: np.ndarray):
    """
    Sample HOG descriptors from an image.

    Parameters
    ----------
    image : The input image.

    Returns
    -------
    feature : The HOG descriptors of the image.
    """
    return hog(
        resize(image, (256, 256)),
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
    ).reshape(-1, 2 * 2 * 9)


def build_vocabulary(images: list[np.ndarray], num: int = 200):
    """
    Build vocabulary from the images.

    Parameters
    ----------
    images : The input images.
    num : The number of vocabulary.

    Returns
    -------
    vocabulary : The cluster centers.
    """
    # Building vocabulary each time is a waste of time.
    # We can save the generated vocabulary to a file,
    # and load it next time to save time.
    VOCABULARY_PATH = "vocabulary.npy"

    # Use existing vocabulary if any.
    if os.path.exists(VOCABULARY_PATH):
        return np.load(VOCABULARY_PATH)

    # Sample HOG feature descriptors.
    features = [describe_feature(image) for image in images]
    features = np.concatenate(features)

    # Cluster features.
    k_means = MiniBatchKMeans(n_clusters=num, max_iter=300, n_init=3).fit(features)

    # Save cluster centers as vocabulary.
    np.save(VOCABULARY_PATH, k_means.cluster_centers_)

    return k_means.cluster_centers_


def get_bag_of_words(image: list[np.ndarray], vocabulary: np.ndarray):
    """
    Get the bag of words of an image.

    Parameters
    ----------
    image : The input image.
    vocabulary : The computed vocabulary.

    Returns
    -------
    histogram : The bag of words of the image.
    """
    # Sample HOG feature descriptors.
    feature = describe_feature(image)

    # Compute the distance between each feature and each cluster center.
    distance = cdist(feature, vocabulary)

    # Find the closest words for each feature.
    closest_words = np.argmin(distance, axis=1)

    # Compute the histogram of closest words.
    histogram = np.histogram(closest_words, bins=len(vocabulary))[0]
    histogram = histogram / np.linalg.norm(histogram)

    return histogram
