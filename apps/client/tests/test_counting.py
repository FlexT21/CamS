import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from websockets.asyncio.server import serve

from src.arduino import Arduino
from src.connection import ServerConnection
from src.direction_tracker import Direction, PersonTrack
from src.occupancy import Occupancy
from src.request import send_image_to_server


class CountingTests(unittest.TestCase):
    def test_left_entry_right_exit_and_dead_zone(self):
        track = PersonTrack(line_x=100, dead_zone=10)
        self.assertIsNone(track.update(150))
        for x in (105, 99, 102, 95):
            self.assertIsNone(track.update(x))
        self.assertEqual(track.update(80), Direction.ENTRADA)
        self.assertIsNone(track.update(70))
        self.assertEqual(track.update(120), Direction.SALIDA)
        self.assertEqual(track.pending_crossings, [Direction.ENTRADA, Direction.SALIDA])

    def test_new_track_and_loss_do_not_count(self):
        track = PersonTrack(line_x=100)
        self.assertIsNone(track.update(10))
        self.assertEqual([track.mark_missed() for _ in range(9)], [False] * 8 + [True])
        self.assertEqual(track.pending_crossings, [])

    def test_persistence_duplicates_unknown_and_zero_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "count.db")
            ledger = Occupancy(path)
            self.assertIsNone(ledger.record("Unknown", Direction.ENTRADA))
            self.assertIsNone(ledger.record("ana", Direction.SALIDA))
            event = ledger.record("ana", Direction.ENTRADA)
            self.assertIsNotNone(event)
            self.assertIsNone(ledger.record("ana", Direction.ENTRADA))
            ledger.record("juan", Direction.ENTRADA)
            ledger.signal_result(event, "acknowledged")
            ledger.close()
            ledger = Occupancy(path)
            self.assertEqual(ledger.totals(), (2, 0, 2))
            ledger.record("ana", Direction.SALIDA)
            self.assertIsNone(ledger.record("ana", Direction.SALIDA))
            ledger.record("juan", Direction.SALIDA)
            self.assertEqual(ledger.totals(), (2, 2, 0))
            ledger.record("ana", Direction.ENTRADA)
            self.assertEqual(ledger.totals(), (3, 2, 1))
            ledger.close()

    def test_arduino_zero_and_ack(self):
        with patch("serial.Serial") as serial, patch("src.arduino.time.sleep"):
            serial.return_value.write.return_value = 1
            serial.return_value.readline.return_value = b"LED_ON\r\n"
            board = Arduino("COM_TEST")
            board.entry()
            serial.return_value.write.assert_called_once_with(b"0")
            serial.return_value.readline.return_value = b""
            with self.assertRaises(OSError):
                board.entry()
            board.close()
            serial.return_value.close.assert_called_once()


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_websocket_transaction(self):
        import json

        async def handler(socket):
            metadata = json.loads(await socket.recv())
            self.assertEqual(await socket.recv(), b"jpeg")
            await socket.send(
                json.dumps(
                    dict(
                        type="recognition_result",
                        frame_id=metadata["frame_id"],
                        status="ok",
                        success=True,
                        user="ana",
                    )
                )
            )

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            connection = ServerConnection(f"ws://127.0.0.1:{port}")
            response = await send_image_to_server(connection, frame_id=7, image=b"jpeg")
            self.assertTrue(response["success"])
            await connection.close()

    async def test_cancelled_request_drops_socket(self):
        received = asyncio.Event()

        async def handler(socket):
            await socket.recv()
            await socket.recv()
            received.set()
            await socket.wait_closed()

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            connection = ServerConnection(f"ws://127.0.0.1:{port}")
            task = asyncio.create_task(connection.request({"frame_id": 1}, b"jpeg"))
            await asyncio.wait_for(received.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertIsNone(connection.websocket)


class CameraFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_recognition_entry_exit_and_only_one_zero(self):
        from types import SimpleNamespace

        import numpy as np

        from src import main as camera

        # Coordinates are in the mirrored display: right -> left, absence, left -> right.
        positions = [0.8] * 4 + [0.1] * 6 + [None] * 10 + [0.1] * 4 + [0.8] * 6
        detections = []
        for x in positions:
            bbox = SimpleNamespace(
                xmin=1 - (x or 0) - 0.05, ymin=0.4, width=0.1, height=0.2
            )
            detection = SimpleNamespace(
                location_data=SimpleNamespace(relative_bounding_box=bbox)
            )
            detections.append(
                SimpleNamespace(detections=[detection] if x is not None else [])
            )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, frame.copy()) for _ in positions] + [
            (False, None)
        ]
        detector = MagicMock()
        crossed = asyncio.Event()
        detection_index = 0

        def detect(_image):
            nonlocal detection_index
            result = detections[detection_index]
            detection_index += 1
            if detection_index >= 6:
                crossed.set()
            return result

        detector.__enter__.return_value.process.side_effect = detect
        mesh = MagicMock()
        mesh.__enter__.return_value.process.return_value.multi_face_landmarks = None
        board = MagicMock()

        async def recognize(*args, **kwargs):
            await crossed.wait()
            return dict(success=True, user="ana")

        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "occupancy.db")
            with (
                patch.object(camera.cv2, "VideoCapture", return_value=cap),
                patch.object(camera.cv2, "imshow"),
                patch.object(camera.cv2, "waitKey", return_value=0),
                patch.object(camera.cv2, "destroyAllWindows"),
                patch.object(
                    camera.face_detection, "FaceDetection", return_value=detector
                ),
                patch.object(camera.face_mesh, "FaceMesh", return_value=mesh),
                patch.object(camera, "Arduino", return_value=board),
                patch.object(camera, "send_image_to_server", side_effect=recognize),
            ):
                await camera.main(
                    0,
                    server_url="ws://test",
                    recognition_interval=0.01,
                    arduino_port="COM_TEST",
                    database=database,
                )
            ledger = Occupancy(database)
            self.assertEqual(ledger.totals(), (1, 1, 0))
            self.assertEqual(
                ledger.db.execute(
                    "SELECT signal FROM events WHERE direction='entrada'"
                ).fetchone()[0],
                "acknowledged",
            )
            ledger.close()
            board.entry.assert_called_once()
            board.close.assert_called_once()
            cap.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
