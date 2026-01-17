from typing import List

from src.core.constants import IMAGE_EXTENSIONS
from src.core.utils import face_encodings, get_euclidian_distance, load_image_file
from src.dtypes import Arr
from src.schemas import FaceEncoding
from src.utils.rootdir import USERSDIR


def load_users() -> List[FaceEncoding]:
    users = []
    for user in USERSDIR.iterdir():
        if not user.is_dir():
            continue

        photos_path = [
            user_photo
            for user_photo in user.iterdir()
            if user_photo.suffix.lower() in IMAGE_EXTENSIONS
        ]

        # Assume one face per photo
        # It's unrelated, but Python's `map` is awful to use compared to other languages.
        encodings = []
        for photo in photos_path:
            image = load_image_file(str(photo))
            photo_encodings = face_encodings(image)
            if photo_encodings:
                encodings.append(photo_encodings[0])

        users.append(FaceEncoding(user=user.name, encodings=encodings))

    # TODO: Implement KNN algorithm to optimize the search for large user bases.

    return users


def recognize_user(
    known_users: List[FaceEncoding],
    encoding_actual: Arr,
    threshold: float,
) -> tuple[str, float]:
    best_user = "Unknown"
    best_distance = float("inf")

    for user in known_users:
        for encoding in user["encodings"]:
            d = get_euclidian_distance(encoding, encoding_actual)
            if d < best_distance:
                best_distance = d
                best_user = user["user"]

    if best_distance > threshold:
        best_user = "Unknown"

    return best_user, best_distance


if __name__ == "__main__":
    from pprint import pprint

    pprint(load_users())
