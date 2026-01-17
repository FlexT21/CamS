from typing import List

from src.core.constants import IMAGE_EXTENSIONS
from src.core.utils import face_encodings, get_euclidian_distance, load_image_file
from src.schemas import FaceEncoding
from src.utils import USERSDIR


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
        encodings = list(
            map(
                lambda photo: face_encodings(load_image_file(str(photo)))[0],
                photos_path,
            )
        )

        users.append(FaceEncoding(user=user.name, encodings=encodings))

    return users


def recognize_user(
    known_users: List[FaceEncoding],
    encoding_actual: List[float],
    threshold: float = 1.5,
) -> tuple[str, float]:
    min_distance = float("inf")
    recognized_user = "Unknown"

    for user in known_users:
        for encoding in user["encodings"]:
            distance = get_euclidian_distance(encoding, encoding_actual)
            if distance > min_distance:
                continue

            min_distance = distance
            if min_distance < threshold:
                recognized_user = user["user"]

    return recognized_user, min_distance


if __name__ == "__main__":
    load_users()
