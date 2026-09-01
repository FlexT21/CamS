import numpy as np
from sklearn.cluster import KMeans

from src.core.config import settings
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
            if user_photo.suffix.lower() in settings.VALID_IMAGE_EXTENSIONS
        ]

        encodings = []
        for photo in photos_path:
            image = load_image_file(str(photo))
            photo_encodings = face_encodings(image)
            if photo_encodings:
                encodings.append(photo_encodings[0])

        if len(encodings) >= settings.K_MEANS_CLUSTERS:
            kmeans = KMeans(n_clusters=settings.K_MEANS_CLUSTERS, random_state=37)
            kmeans.fit(encodings)
            encodings = kmeans.cluster_centers_.tolist()

        users.append(FaceEncoding(user=user.name, encodings=encodings))

    return users
