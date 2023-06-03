import numpy as np

__all__ = ["filter_image", "create_gaussian_kernel"]


def filter_image(image: np.ndarray, kernel: np.ndarray):
    """
    Filter an image with a convolutional kernel.

    Parameters
    ----------
    image : The image to be filtered.
    kernel : The convolutional kernel.

    Returns
    -------
    filtered_image : The filtered image.
    """
    # Get relevant information about the image.
    image_height, image_width, channel_num = image.shape

    # Flip the kernel, since we are doing convolution.
    kernel = np.flip(kernel)

    # Get relevant information about the kernel.
    kernel_height, kernel_width = kernel.shape
    assert kernel_width % 2 == 1, "The kernel must have odd width."
    assert kernel_height % 2 == 1, "The kernel must have odd height."

    # Add paddings around the image, so that the kernel
    # can easily deal with pixels near the edges.
    padding_width = kernel_width // 2
    padding_height = kernel_height // 2
    padded_image = np.pad(
        image,
        (
            (padding_height, padding_height),
            (padding_width, padding_width),
            (0, 0),
        ),
    )

    # Create an empty filtered image.
    filtered_image = np.zeros_like(image)

    # For each channel:
    for channel in range(channel_num):
        # Apply convolution to each pixel.
        for y in range(image_height):
            for x in range(image_width):
                region = padded_image[
                    y : y + kernel_height,
                    x : x + kernel_width,
                    channel,
                ]
                filtered_image[y, x, channel] = np.sum(region * kernel)

    return filtered_image


def create_gaussian_kernel(sigma: int, radius: int):
    """
    Create a Gaussian kernel.

    Parameters
    ----------
    sigma : The standard deviation of the kernel.
    radius : The radius of the kernel.

    Returns
    -------
    kernel : The Gaussian kernel.

    References
    ----------
    https://stackoverflow.com/a/45764688
    """
    # Create a 1D Gaussian distribution.
    dist = np.asarray(
        [
            np.exp(-(z**2) / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
            for z in range(-radius, radius + 1)
        ]
    )

    # Expand it to 2D.
    kernel = np.outer(dist, dist)

    return kernel
