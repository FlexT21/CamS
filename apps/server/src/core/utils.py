from typing import List

import cv2
import numpy as np
from mediapipe.python.solutions import face_mesh


def load_image_file(file_path: str):
    with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as mp_face_mesh:
        image = cv2.imread(file_path)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return mp_face_mesh.process(image)


def face_encodings(image) -> List[List[float]]:
    if not image:
        return []

    face_landmarks = image.multi_face_landmarks
    encodings = []

    for landmarks in face_landmarks:
        encoding = [landmark.x for landmark in landmarks.landmark]
        encodings.append(encoding)

    return encodings


def get_euclidian_distance(encoding1: List[float], encoding2: List[float]) -> float:
    if len(encoding1) != len(encoding2):
        raise ValueError("Encodings must be of the same length")

    encoding1_np = np.array(encoding1)
    encoding2_np = np.array(encoding2)
    return np.linalg.norm(encoding1_np - encoding2_np)
