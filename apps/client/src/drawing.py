from mediapipe.python.solutions import drawing_styles, drawing_utils, face_mesh
import math
from dataclasses import dataclass
import cv2

def draw_face_mesh(image, results) -> None:
    for face_landmarks in results.multi_face_landmarks:
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
        drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

def get_face_centroid(face_landmarks, image_width: int, image_height: int) -> tuple[float, float]:
    xs = [lm.x * image_width for lm in face_landmarks.landmark]
    ys = [lm.y * image_height for lm in face_landmarks.landmark]
    return sum(xs) / len(xs), sum(ys) / len(ys)

@dataclass
class DirectionVector:
    origin_x: int
    origin_y: int
    dx: float
    dy: float

    @property
    def magnitude(self) -> float:
        return math.hypot(self.dx, self.dy)

    @property
    def angle_degrees(self) -> float:
        return math.degrees(math.atan2(self.dy, self.dx))

    @property
    def tip(self) -> tuple[int, int]:
        return int(self.origin_x + self.dx), int(self.origin_y + self.dy)


def compute_display_vector(centroid_x: float, centroid_y: float, frame_width: int, frame_height: int) -> DirectionVector:
    """Vector para MOSTRAR en pantalla, sobre el frame ya espejado (cv2.flip)."""
    origin_x = frame_width // 2
    origin_y = frame_height // 2
    mirrored_centroid_x = frame_width - centroid_x  # solo el eje X se invierte con el flip horizontal
    return DirectionVector(
        origin_x=origin_x,
        origin_y=origin_y,
        dx=mirrored_centroid_x - origin_x,
        dy=centroid_y - origin_y,
    )


def draw_direction_vector(image, vector: DirectionVector) -> None:
    origin = (vector.origin_x, vector.origin_y)
    cv2.arrowedLine(image, origin, vector.tip, (0, 255, 255), 2, tipLength=0.15)
    cv2.circle(image, origin, 4, (255, 0, 0), -1)
    cv2.circle(image, vector.tip, 6, (0, 0, 255), -1)  # el centroide real, en punta de flecha

    label = f"|v|={vector.magnitude:.0f}px  ang={vector.angle_degrees:.0f}deg"
    cv2.putText(image, label, (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)