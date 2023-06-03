import cv2
import numpy as np
from skimage.feature import peak_local_max

__all__ = ["get_interest_points", "describe_features", "match_features"]


def get_interest_points(image: np.ndarray, feature_size: int = 16):
    """
    Get all interest points in an image.

    Parameters
    ----------
    image : The input image.
    feature_size : The width and height of each feature patch.

    Returns
    -------
    x : The x-coordinates of the interest points.
    y : The y-coordinates of the interest points.
    """
    # Compute the gradients in the x and y directions.
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the products of gradients at each pixel.
    gradient_xx = gradient_x * gradient_x
    gradient_xy = gradient_x * gradient_y
    gradient_yy = gradient_y * gradient_y

    # Smooth the products in a local neighborhood.
    kernel = cv2.getGaussianKernel(feature_size, -1)
    smoothed_xx = cv2.filter2D(gradient_xx, -1, kernel)
    smoothed_xy = cv2.filter2D(gradient_xy, -1, kernel)
    smoothed_yy = cv2.filter2D(gradient_yy, -1, kernel)

    # Compute the Harris corner response.
    k = 0.04
    corner_response = (
        smoothed_xx * smoothed_yy
        - smoothed_xy**2
        - k * (smoothed_xx + smoothed_yy) ** 2
    )

    # Find local maxima in the corner response function.
    coordinates = peak_local_max(
        corner_response,
        min_distance=feature_size,
        threshold_abs=0.01 * np.max(corner_response),
    )

    # Extract the x and y coordinates of the interest points.
    x = coordinates[:, 1]
    y = coordinates[:, 0]

    return x, y


def describe_features(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    feature_size: int = 16,
):
    """
    Get the feature descriptors of all interest points in an image.

    Parameters
    ----------
    image : The input image.
    x : The x-coordinates of the interest points.
    y : The y-coordinates of the interest points.
    feature_size : The width and height of each feature patch.

    Returns
    -------
    features : The feature descriptors.
    """
    feature_num = len(x)
    bin_num = 8
    features = np.zeros((feature_num, 4 * 4 * bin_num))

    # For each interest point:
    for i in range(feature_num):
        # Extract the local patch around the interest point.
        center_x = int(x[i])
        center_y = int(y[i])
        radius = feature_size // 2
        patch = image[
            center_y - radius : center_y + radius,
            center_x - radius : center_x + radius,
        ]

        # Compute the histogram of gradients in the patch.
        gradients_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        gradients_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        magnitudes = np.sqrt(gradients_x**2 + gradients_y**2)
        orientations = np.arctan2(gradients_y, gradients_x) % (2 * np.pi)

        # Divide the patch into a 4x4 grid and compute histograms for each cell.
        cell_width = feature_size // 4
        histogram = np.zeros((4, 4, bin_num))
        for row in range(4):
            for col in range(4):
                cell_magnitudes = magnitudes[
                    row * cell_width : (row + 1) * cell_width,
                    col * cell_width : (col + 1) * cell_width,
                ]
                cell_orientations = orientations[
                    row * cell_width : (row + 1) * cell_width,
                    col * cell_width : (col + 1) * cell_width,
                ]
                cell_histogram = np.histogram(
                    cell_orientations,
                    bins=bin_num,
                    range=(0, 2 * np.pi),
                    weights=cell_magnitudes,
                )[0]
                histogram[row, col] = cell_histogram

        # Flatten the histogram into a single feature.
        features[i] = histogram.flatten()

    return features


def match_features(image1_features: np.ndarray, image2_features: np.ndarray):
    """
    Match the features between two images.

    Parameters
    ----------
    image1_features : The feature descriptors of the first image.
    image2_features : The feature descriptors of the second image.

    Returns
    -------
    matches : The index pairs of matched features.
    """
    # Record the confidence and index pair of each match.
    matches = []

    # For each feature in image 1:
    for image1_index, image1_feature in enumerate(image1_features):
        # Initialize the nearest and the second nearest distances to infinity.
        first_nearest_distance = np.inf
        second_nearest_distance = np.inf

        # Initialize the index of the nearest feature to -1.
        match_index = -1

        # For each feature in image 2:
        for image2_index, image2_feature in enumerate(image2_features):
            # Compute the distance between the features.
            distance = np.linalg.norm(image1_feature - image2_feature)

            # Update the nearest and second nearest distances if necessary.
            if distance < first_nearest_distance:
                second_nearest_distance = first_nearest_distance
                first_nearest_distance = distance
                match_index = image2_index
            elif distance < second_nearest_distance:
                second_nearest_distance = distance

        # Compute the ratio of the nearest and second nearest distances.
        ratio = first_nearest_distance / second_nearest_distance

        # Record the confidence and the index pair.
        matches.append((1 - ratio, image1_index, match_index))

    # Sort the records by NNDR in descending order.
    matches = sorted(matches, key=lambda match: match[0], reverse=True)

    # Discard the confidences.
    matches = np.asarray([[match[1], match[2]] for match in matches])

    return matches
