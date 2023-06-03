import os
import glob
from skimage.io import imread
from skimage.util import img_as_float

__all__ = ["load_dataset"]


def load_dataset(dir_path: str, category_names: list[str], limit: int = 100):
    """
    Load the images, image paths and labels in a dataset.

    Parameters
    ----------
    dir_path : The path to the dataset directory.
    category_names : The names of the categories in the dataset.
    limit : The maximum number of images to load per category.

    Returns
    -------
    images : The images in the dataset.
    image_paths : The paths of the images in the dataset.
    labels : The labels in the dataset.
    """
    # Record images, image paths and labels.
    images = []
    image_paths = []
    labels = []

    # For each category:
    for category_name in category_names:
        # Get the paths of the first `limit` images in the category.
        pattern = os.path.join(dir_path, category_name, "*.jpg")
        paths = glob.glob(pattern)[:limit]

        # Load these images.
        images.extend([load_image(path) for path in paths])

        # Record these image paths.
        image_paths.extend(paths)

        # These images all have the same label.
        labels.extend([category_name] * len(paths))

    return images, image_paths, labels


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
