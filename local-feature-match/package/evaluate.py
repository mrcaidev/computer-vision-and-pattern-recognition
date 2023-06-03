import numpy as np

__all__ = ["evaluate_accuracy"]


def evaluate_accuracy(
    truth_points: tuple[np.ndarray],
    estimated_points: tuple[np.ndarray],
    matches: np.ndarray,
):
    """
    Evaluate the accuracy of the estimated points
    by comparing them to the ground truth points.

    Parameters
    ----------
    truth_points : The ground truth points.
    estimated_points : The estimated points.
    matches : The index pairs of matched features.
    """
    # Get the coordinates of each ground truth point.
    x1, y1, x2, y2 = truth_points

    # Get the coordinates of each matched point.
    x1_est, y1_est, x2_est, y2_est = estimated_points
    x1_matched = np.asarray([x1_est[match[0]] for match in matches])
    y1_matched = np.asarray([y1_est[match[0]] for match in matches])
    x2_matched = np.asarray([x2_est[match[1]] for match in matches])
    y2_matched = np.asarray([y2_est[match[1]] for match in matches])

    # The rest of this function is given by the professor.
    # I'm too lazy to refactor it.

    uniqueness_dist = 150
    good_match_dist = 150

    bad_match_count = 0
    top_100_count = 0

    # Used to keep track of which TA points the student has matched
    # to so the student only gets credit for matching a TA point once.
    correct_matches = np.zeros(x2.shape[0])

    # For each ground truth point in image 1:
    for i in range(x1.shape[0]):
        # 1. find the student points within uniqueness_dist pixels of the ground truth point.
        x_dists = x1_matched - x1[i]
        y_dists = y1_matched - y1[i]

        # computes distances of each interest point to the ground truth point.
        dists = np.sqrt(np.power(x_dists, 2.0) + np.power(y_dists, 2.0))

        # get indices of points where distance is < uniqueness_dist.
        close_to_truth = dists < uniqueness_dist

        # 2. get the points in image1 and their corresponding matches in image2.
        image2_x = x2_matched[close_to_truth]
        image2_y = y2_matched[close_to_truth]

        # 3. compute the distance of the student's image2 matches to the ground truth match.
        x_dists_2 = image2_x - x2[i]
        y_dists_2 = image2_y - y2[i]

        dists_2 = np.sqrt(np.power(x_dists_2, 2.0) + np.power(y_dists_2, 2.0))

        # 4. matches within good_match_dist then count it as a correct match.
        good = dists_2 < good_match_dist
        if np.sum(good) >= 1.0:
            correct_matches[i] = 1
            if i < 100:
                top_100_count += 1
        else:
            bad_match_count += 1

    precision = (np.sum(correct_matches) / x2.shape[0]) * 100.0
    accuracy = min(top_100_count, 100)

    print(
        f"{str(int(np.sum(correct_matches)))} good matches, {str(bad_match_count)} bad matches."
    )
    print(f"Precision: {np.round(precision, 2)}%")
    print(f"Accuracy (top 100): {accuracy}%")
