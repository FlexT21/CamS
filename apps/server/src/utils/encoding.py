from typing import List

import cv2
import face_recognition
import numpy as np
from cv2.typing import MatLike

from src.dtypes import Arr


def preprocess_image_for_face_recognition(image: MatLike) -> MatLike:
    # Resize image for faster processing
    small_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    return small_image


def face_encodings(image: MatLike) -> List[Arr]:
    if image is None:
        return []

    processed_image = preprocess_image_for_face_recognition(image)
    encodings = face_recognition.face_encodings(processed_image)

    # L2-normalize embeddings to unit length for stable distance comparison
    encodings = [encoding / np.linalg.norm(encoding) for encoding in encodings]

    return encodings


def get_euclidian_distance(encoding1: Arr, encoding2: Arr) -> float:
    if len(encoding1) != len(encoding2):
        raise ValueError("Encodings must be of the same length")

    if isinstance(encoding1, list):
        encoding1 = np.array(encoding1)
    if isinstance(encoding2, list):
        encoding2 = np.array(encoding2)

    encoding1_np = encoding1.astype(np.float64)
    encoding2_np = encoding2.astype(np.float64)

    return np.linalg.norm(encoding1_np - encoding2_np)
