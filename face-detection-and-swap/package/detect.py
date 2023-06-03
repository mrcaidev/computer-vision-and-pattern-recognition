import cv2
import numpy as np

__all__ = [
    "get_landmarks",
    "get_mask",
    "mask_image",
    "get_triangles",
    "analyze_image",
]

WHITE = (255, 255, 255)


def get_landmarks(image: np.ndarray, detector, predictor):
    """
    Detect all landmarks of the person.

    Parameters
    ----------
    image : The input image.
    detector : The face detector.
    predictor : The landmark predictor.

    Returns
    -------
    landmarks : The detected landmarks.
    """
    # Convert the image to grayscale.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces.
    faces = detector(image)

    # If there are more than one faces:
    if len(faces) > 1:
        raise Exception("Too many faces")

    # If there is no face:
    if len(faces) == 0:
        raise Exception("No face")

    # Detect landmarks.
    landmarks = predictor(image, faces[0])

    # Convert to numpy array.
    landmarks = np.asarray([(landmark.x, landmark.y) for landmark in landmarks.parts()])

    return landmarks


def get_mask(image: np.ndarray, landmarks: np.ndarray):
    """
    Get the face mask.

    Parameters
    ----------
    image : The input image.
    landmarks : The detected landmarks.

    Returns
    -------
    hull : The face convex hull.
    mask : The face mask.
    """
    # Get the convex hull of the face.
    hull = cv2.convexHull(landmarks)

    # Draw a filled polygon on the mask.
    mask = np.zeros_like(image)
    cv2.fillConvexPoly(mask, hull, WHITE)

    return hull, mask


def mask_image(image: np.ndarray, mask: np.ndarray):
    """
    Apply a mask to an image.

    Parameters
    ----------
    image : The input image.
    mask : The mask.

    Returns
    -------
    masked_image: The masked image.
    """
    return cv2.bitwise_and(image, mask)


def get_triangles(landmarks: np.ndarray, hull: np.ndarray):
    """
    Get the face mesh triangulation.

    Parameters
    ----------
    landmarks : The detected landmarks.
    hull : The face convex hull.

    Returns
    -------
    triangles : The face triangles.
    """
    # Create an instance of Subdiv2D.
    rect = cv2.boundingRect(hull)
    subdiv = cv2.Subdiv2D(rect)

    # Insert all landmarks into the subdiv.
    for x, y in landmarks:
        subdiv.insert((int(x), int(y)))

    # Get the triangulation.
    triangles = np.asarray(subdiv.getTriangleList())

    return triangles


def analyze_image(image: np.ndarray, landmarks: np.ndarray, triangles: np.ndarray):
    """
    Draw landmarks and Delaunay triangulation onto the image.

    Parameters
    ----------
    image : The input image.
    landmarks : The detected landmarks.
    triangles : The Delaunay triangulation.

    Returns
    -------
    marked_image : The marked image.
    """
    # Copy the image.
    marked_image = image.copy()

    # Draw the landmarks.
    for x, y in landmarks:
        cv2.circle(marked_image, (int(x), int(y)), 2, WHITE)

    # Draw the Delaunay triangulation.
    for triangle in triangles:
        cv2.polylines(
            marked_image,
            [np.asarray(triangle, np.int32).reshape((-1, 1, 2))],
            True,
            WHITE,
        )

    return marked_image
