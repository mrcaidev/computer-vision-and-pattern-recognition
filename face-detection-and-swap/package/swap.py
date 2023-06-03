import cv2
import numpy as np

__all__ = ["get_transformation_matrix", "warp_image"]


def get_transformation_matrix(source: np.ndarray, target: np.ndarray):
    """
    Get the affine transformation matrix.

    Parameters
    ----------
    source : The source matrix.
    target : The target matrix.

    Returns
    -------
    matrix : The affine transformation matrix.
    """
    # Convert the source and target matrix to float64 format.
    source = source.astype(np.float64)
    target = target.astype(np.float64)

    # Normalize the source matrix.
    source_center = np.mean(source, axis=0).reshape(1, 2)
    source_std = np.std(source)
    source = (source - source_center) / source_std

    # Normalize the target matrix.
    target_center = np.mean(target, axis=0).reshape(1, 2)
    target_std = np.std(target)
    target = (target - target_center) / target_std

    # Compute the scale factor.
    s = target_std / source_std

    # Compute the rotation matrix.
    U, _, Vt = np.linalg.svd(source.T @ target)
    R = U @ Vt

    # Compute the translation matrix.
    T = target_center - s * source_center @ R

    # Construct the transformation matrix [s * R | T].
    matrix = np.hstack([s * R, T.T])

    return matrix


def warp_image(image: np.ndarray, matrix: np.ndarray):
    """
    Warp an image with transformation matrix.

    Parameters
    ----------
    image : The input image.
    matrix : The affine transformation matrix.

    Returns
    -------
    warped_image : The warped image.
    """
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
