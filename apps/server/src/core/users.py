from typing import List

from sklearn.cluster import KMeans

from src.core.constants import IMAGE_EXTENSIONS, K_MEANS_CLUSTERS
from src.schemas import FaceEncoding
from src.utils import USERSDIR, face_encodings, load_image_file


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

        # Implement KMeans algorithm to optimize the search for large user bases.
        if len(encodings) >= K_MEANS_CLUSTERS:
            kmeans = KMeans(n_clusters=K_MEANS_CLUSTERS, random_state=37)
            kmeans.fit(encodings)
            encodings = kmeans.cluster_centers_.tolist()

        users.append(FaceEncoding(user=user.name, encodings=encodings))

    return users


# I'll use an in-memory cache for known users to avoid reloading them on each recognition.
known_users = load_users()


if __name__ == "__main__":
    from pprint import pprint

    pprint(known_users)
