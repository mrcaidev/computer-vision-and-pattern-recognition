import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import plot_matches

__all__ = ["show_interest_points", "show_matches"]


def show_interest_points(
    image1: np.ndarray,
    image2: np.ndarray,
    points: tuple[np.ndarray],
):
    """
    Show the interest points of the images.

    Parameters
    ----------
    image1 : The first image.
    image2 : The second image.
    points : The interest points. The order is (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = points

    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].axis("off")
    axes[0].imshow(image1, cmap="gray")
    axes[0].scatter(x1, y1, alpha=0.9, s=3)

    axes[1].axis("off")
    axes[1].imshow(image2, cmap="gray")
    axes[1].scatter(x2, y2, alpha=0.9, s=3)


def show_matches(
    image1: np.ndarray,
    image2: np.ndarray,
    points: tuple[np.ndarray],
    matches: np.ndarray,
    limit: int = 0,
):
    """
    Show the matches between the features.

    Parameters
    ----------
    image1 : The first image.
    image2 : The second image.
    points : The interest points. The order is (x1, y1, x2, y2).
    matches : The index pairs of matched features.
    limit : The maximum number of matches to show. If 0, show all matches.
    """
    x1, y1, x2, y2 = points

    keypoints1 = np.asarray(list(zip(y1, x1)))
    keypoints2 = np.asarray(list(zip(y2, x2)))

    if limit > 0:
        print(f"Showing the first {limit} matches.")
        matches = matches[:limit]

    _, ax = plt.subplots()
    plot_matches(ax, image1, image2, keypoints1, keypoints2, matches, only_matches=True)
    ax.axis("off")
    plt.show()
