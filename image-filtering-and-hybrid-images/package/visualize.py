import numpy as np
import matplotlib.pyplot as plt

__all__ = ["show_images"]


def show_images(*images: np.ndarray, size: int = 5):
    """
    Show several images in a row.

    Parameters
    ----------
    images : The images to be shown.
    size : The width and height of each image.
    """
    image_num = len(images)

    _, axes = plt.subplots(1, image_num, figsize=(size * image_num, size))
    for ax, image in zip(axes, images):
        ax.axis("off")
        ax.imshow(image)
