from scipy.io import loadmat
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.util import img_as_float
from skimage.io import imread

__all__ = ["load_image", "load_points"]

# Scale down the images for faster computation.
SCALE_FACTOR = 0.5


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
    image = rgb2gray(image)
    image = rescale(image, SCALE_FACTOR)
    return image


def load_points(file_path: str):
    """
    Load interest points.

    Parameters
    ----------
    file_path : The path of the interest points file.

    Returns
    -------
    points : The interest points. The order is (x1, y1, x2, y2).
    """
    points = loadmat(file_path)
    x1 = points["x1"] * SCALE_FACTOR
    y1 = points["y1"] * SCALE_FACTOR
    x2 = points["x2"] * SCALE_FACTOR
    y2 = points["y2"] * SCALE_FACTOR
    return x1, y1, x2, y2
