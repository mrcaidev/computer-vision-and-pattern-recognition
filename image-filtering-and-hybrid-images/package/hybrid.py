import numpy as np
from .filter import filter_image, create_gaussian_kernel

__all__ = ["hybridize_image"]


def hybridize_image(
    image1: np.ndarray,
    image2: np.ndarray,
    frequency_threshold: int = 7,
):
    """
    Hybridize two images.

    Mix the low frequency components of image 1
    and the high frequency components of image 2.

    Parameters
    ----------
    image1 : The first image.
    image2 : The second image.
    frequency_threshold : The threshold of low and high frequency.

    Returns
    -------
    low_frequencies : The low frequency components of image 1.
    high_frequencies : The high frequency components of image 2.
    hybrid_image : The hybrid image.

    Notes
    -----
    The returned high frequency components are added by 0.5
    to make them visible, and then clipped to [0, 1].

    The returned hybrid image is clipped to [0, 1].
    """
    # Make sure that the two images are of the same size.
    assert image1.shape == image2.shape, "The two images must be of the same size."

    # Create a Gaussian kernel.
    kernel = create_gaussian_kernel(frequency_threshold, frequency_threshold * 2)

    # Extract the low frequency components of image 1.
    low_frequencies = filter_image(image1, kernel)

    # Extract the high frequency components of image 2.
    high_frequencies = image2 - filter_image(image2, kernel)

    # Hybridize the extracted components.
    hybrid_image = low_frequencies + high_frequencies

    return (
        low_frequencies,
        np.clip(high_frequencies + 0.5, 0, 1),
        np.clip(hybrid_image, 0, 1),
    )
