import cv2
from cv2.typing import MatLike


def load_image_file(file_path: str) -> MatLike:
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {file_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
