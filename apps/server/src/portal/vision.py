"""Adapters for the existing MediaPipe and face-recognition/dlib engines."""

import base64
import binascii
import importlib.util
import io
import math
import threading
from pathlib import Path

from fastapi import HTTPException


def normalize(values):
    values = [float(v) for v in values]
    norm = math.sqrt(sum(v * v for v in values))
    if not values or not math.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid face embedding")
    return [v / norm for v in values]


def cosine(a, b):
    if len(a) != len(b) or not a:
        return -1.0
    return sum(x * y for x, y in zip(a, b))


def distance(a, b):
    if len(a) != len(b) or not a:
        return float("inf")
    from src.utils import get_euclidian_distance

    return float(get_euclidian_distance(a, b))


def match(embedding, persons, threshold):
    # Reuse the server's existing Euclidean recognizer, without normalizing
    # its dlib embeddings or substituting cosine/SFace identification.
    from src.schemas import FaceEncoding
    from src.services.users import recognize_user

    compatible = [p for p in persons if len(p["encoding"]) == len(embedding)]
    result = recognize_user(
        [FaceEncoding(user=p["id"], encodings=[p["encoding"]]) for p in compatible],
        embedding,
        threshold,
    )
    if not result.success:
        return None
    person = next(p for p in compatible if p["id"] == result.user)
    return {"id": person["id"], "name": person["name"]}


def decode_image(encoded, config):
    if encoded.startswith("data:image/jpeg;base64,"):
        encoded = encoded.split(",", 1)[1]
    if len(encoded) > ((config.max_image_bytes + 2) // 3) * 4:
        raise HTTPException(413, "JPEG exceeds byte limit")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise HTTPException(422, "Invalid base64 JPEG") from None
    if len(data) > config.max_image_bytes:
        raise HTTPException(413, "JPEG exceeds byte limit")
    try:
        import cv2
        import numpy as np
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        raise HTTPException(
            503, "Image processing requires Pillow, numpy and OpenCV"
        ) from None
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG":
                raise HTTPException(422, "Only JPEG images are accepted")
            width, height = image.size
            if (
                width < 32
                or height < 32
                or width * height > config.max_pixels
                or max(width, height) > 4096
            ):
                raise HTTPException(413, "JPEG dimensions outside allowed limits")
            image.verify()
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        # OpenCV honors JPEG EXIF orientation; 90-degree rotations swap dimensions.
        if frame is None or frame.shape[:2] not in ((height, width), (width, height)):
            raise ValueError("JPEG decode failed")
    except HTTPException:
        raise
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise HTTPException(422, "Invalid JPEG image") from None
    return data, frame


class FaceEngine:
    def __init__(self, config):
        self.config = config
        self.recognizer = None
        self.lock = threading.Lock()
        self.error = None
        self.mesh = None

    @property
    def paths(self):
        spec = importlib.util.find_spec("face_recognition_models")
        if spec is None or not spec.origin:
            return []
        folder = Path(spec.origin).parent / "models"
        return [
            folder / "dlib_face_recognition_resnet_model_v1.dat",
            folder / "shape_predictor_5_face_landmarks.dat",
            folder / "shape_predictor_68_face_landmarks.dat",
        ]

    def status(self):
        missing = [
            name
            for name in ("dlib", "face_recognition", "face_recognition_models")
            if importlib.util.find_spec(name) is None
        ]
        models = [path.name for path in self.paths if not path.is_file()]
        state = (
            "missing_dependency"
            if missing
            else "missing_models"
            if models
            else "not_loaded"
        )
        if self.error:
            state = "error"
        elif self.recognizer is not None:
            state = "ready"
        return {
            "backend": "face_recognition_dlib",
            "state": state,
            "missing_models": models,
            "missing_dependencies": missing,
        }

    def _load(self):
        if self.recognizer is not None:
            return
        status = self.status()
        if status["missing_dependencies"] or status["missing_models"]:
            raise HTTPException(
                503, "Existing face-recognition/dlib engine is unavailable"
            )
        try:
            from src.utils import face_encodings

            self.recognizer = face_encodings
            self.error = None
        except Exception:
            self.error = "Existing face-recognition/dlib engine could not be loaded"
            raise HTTPException(503, self.error) from None

    def embeddings(self, frame):
        with self.lock:
            self._load()
            try:
                import cv2

                return [
                    value.tolist()
                    for value in self.recognizer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ]
            except Exception:
                raise HTTPException(503, "Face inference failed") from None

    def landmarks(self, frame):
        # Preserve the old client's MediaPipe FaceMesh visualisation.
        with self.lock:
            try:
                import cv2
                from mediapipe.python.solutions import face_mesh

                if self.mesh is None:
                    self.mesh = face_mesh.FaceMesh(
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                result = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                height, width = frame.shape[:2]
                output = []
                for face in result.multi_face_landmarks or []:
                    points = [
                        {"x": width - 1 - point.x * width, "y": point.y * height}
                        for point in face.landmark
                    ]
                    xs, ys = [p["x"] for p in points], [p["y"] for p in points]
                    connections = sorted(
                        face_mesh.FACEMESH_TESSELATION
                        | face_mesh.FACEMESH_CONTOURS
                        | face_mesh.FACEMESH_IRISES
                    )
                    output.append(
                        {
                            "x": min(xs),
                            "y": min(ys),
                            "width": max(xs) - min(xs),
                            "height": max(ys) - min(ys),
                            "points": [points[i] for i in (33, 263, 1, 61, 291)],
                            "mesh_points": points,
                            "mesh_connections": connections,
                        }
                    )
                return output
            except Exception:
                raise HTTPException(503, "MediaPipe face mesh failed") from None

    def close(self):
        if self.mesh is not None:
            self.mesh.close()


class MediaPipeTracker:
    """Single-person face tracking, matching the existing camera client.

    Detection is MediaPipe FaceDetection. ID continuity survives brief missed
    frames; multiple faces clear association instead of choosing one arbitrarily.
    """

    def __init__(self, config):
        try:
            from mediapipe.python.solutions import face_detection

            self.detector = face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        except Exception:
            raise HTTPException(503, "MediaPipe detector is unavailable") from None
        self.track_id = 0
        self.last_seen = None
        self.max_gap = config.track_ttl

    def detect(self, frame):
        import time

        import cv2

        try:
            now = time.monotonic()
            result = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            faces = result.detections or []
            if len(faces) != 1:
                if len(faces) > 1:
                    self.last_seen = None
                return []
            if self.last_seen is None or now - self.last_seen > self.max_gap:
                self.track_id += 1
            self.last_seen = now
            box = faces[0].location_data.relative_bounding_box
            height, width = frame.shape[:2]
            source_left = max(0, int(box.xmin * width))
            source_right = min(width, int((box.xmin + box.width) * width))
            return [
                (
                    self.track_id,
                    width - source_right,
                    max(0, int(box.ymin * height)),
                    width - source_left,
                    min(height, int((box.ymin + box.height) * height)),
                )
            ]
        except Exception:
            raise HTTPException(503, "MediaPipe tracking failed") from None

    def close(self):
        self.detector.close()


def tracker_status(config):
    missing = [
        name for name in ("mediapipe", "cv2") if importlib.util.find_spec(name) is None
    ]
    return {
        "backend": "mediapipe_face_detection",
        "state": "missing_dependency" if missing else "not_loaded",
        "missing_dependencies": missing,
        "single_person": True,
    }
