import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.main import app
from src.schemas import FaceEncoding
from src.services.users import recognize_user


class RecognitionTests(unittest.TestCase):
    def test_empty_database_is_json_serializable(self):
        result = recognize_user([], np.zeros(128), 0.5)
        self.assertFalse(result.success)
        self.assertIsNone(result.distance)
        self.assertIn("null", result.model_dump_json())

    def test_nearest_identity_and_threshold(self):
        users = [FaceEncoding(user="ana", encodings=[np.zeros(128)])]
        self.assertTrue(recognize_user(users, np.zeros(128), 0.5).success)
        self.assertFalse(recognize_user(users, np.ones(128), 0.5).success)

    def test_websocket_rgb_invalid_image_multiple_faces_and_empty_users(self):
        image = np.full((16, 16, 3), (10, 20, 200), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", image)
        from src.api.routes import websocket as route

        with TestClient(app) as client, patch.object(route, "known_users", []):
            with client.websocket_connect("/api/ws/") as socket:

                def request(data, frame):
                    socket.send_json(
                        dict(type="face_image", frame_id=frame, device_id="test")
                    )
                    socket.send_bytes(data)
                    return socket.receive_json()

                self.assertEqual(request(b"invalid", 1)["status"], "no_face")
                self.assertEqual(request(b"", 4)["status"], "no_face")
                with patch.object(
                    route, "face_encodings", return_value=[np.zeros(128), np.zeros(128)]
                ):
                    self.assertFalse(request(jpeg.tobytes(), 2)["success"])
                with patch.object(
                    route, "face_encodings", return_value=[np.zeros(128)]
                ) as encoder:
                    response = request(jpeg.tobytes(), 3)
                    self.assertFalse(response["success"])
                    self.assertIsNone(response["distance"])
                    rgb = encoder.call_args.args[0]
                    self.assertGreater(int(rgb[0, 0, 0]), int(rgb[0, 0, 2]))

    def test_registration_and_reload_protocol(self):
        from src.api.routes import websocket as route

        image = np.full((32, 32, 3), (10, 20, 200), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", image)
        with tempfile.TemporaryDirectory() as tmp, TestClient(app) as client:
            with (
                patch.object(route, "USERSDIR", Path(tmp)),
                patch.object(route, "face_encodings", return_value=[np.zeros(128)]),
                patch.object(route, "reload_known_users") as reload_users,
            ):
                with client.websocket_connect("/api/ws/") as socket:
                    socket.send_json(
                        dict(
                            type="register_face",
                            frame_id=1,
                            device_id="test",
                            username="ana",
                        )
                    )
                    socket.send_bytes(jpeg.tobytes())
                    self.assertTrue(socket.receive_json()["success"])
                    saved = cv2.imread(str(Path(tmp) / "ana" / "capture_0001.jpg"))
                    self.assertGreater(int(saved[0, 0, 2]), int(saved[0, 0, 0]))
                    socket.send_json(
                        dict(
                            type="register_complete",
                            frame_id=2,
                            device_id="test",
                            username="ana",
                        )
                    )
                    self.assertTrue(socket.receive_json()["success"])
                    reload_users.assert_called_once()
                    socket.send_json(
                        dict(
                            type="register_complete",
                            frame_id=3,
                            device_id="test",
                            username="../escape",
                        )
                    )
                    self.assertFalse(socket.receive_json()["success"])


if __name__ == "__main__":
    unittest.main()
