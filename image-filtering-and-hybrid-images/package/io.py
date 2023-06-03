from skimage.io import imread
from skimage.util import img_as_float

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
    image = imread(file_path)
    image = img_as_float(image)
    return image
