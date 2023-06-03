import numpy as np
from PIL import Image

__all__ = ["load_image"]


def load_image(file_path: str):
    """
    Load an image.

    Parameters
    ----------
    file_path : The path of the image file.

    Returns
    -------
    image : The image.
    """
    image = Image.open(file_path)
    image = image.resize((300, 300))
    image = np.asarray(image)
    return image
