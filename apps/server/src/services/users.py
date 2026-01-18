from typing import List

from src.dtypes import Arr
from src.schemas import FaceEncoding, RecognizeUserResponse
from src.utils import get_euclidian_distance


def recognize_user(
    known_users: List[FaceEncoding],
    current_encoding: Arr,
    threshold: float,
) -> RecognizeUserResponse:
    best_user = "Unknown"
    success = False
    best_distance = float("inf")

    for user in known_users:
        for encoding in user.encodings:
            d = get_euclidian_distance(encoding, current_encoding)
            if d < best_distance:
                best_distance = d
                best_user = user.user
                success = True

    if best_distance > threshold:
        best_user = "Unknown"
        success = False

    return RecognizeUserResponse(
        user=best_user, success=success, distance=best_distance
    )
