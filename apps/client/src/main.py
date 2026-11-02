import argparse
import asyncio
import time
from pathlib import Path

import cv2
from mediapipe.python.solutions import face_detection, face_mesh

from src.arduino import Arduino
from src.connection import ServerConnection
from src.direction_tracker import Direction, PersonTrack
from src.drawing import (
    compute_display_vector,
    draw_counter,
    draw_dashed_line,
    draw_direction_vector,
    draw_face_mesh,
)
from src.occupancy import Occupancy
from src.request import send_image_to_server


async def recognize_track(connection, track, image_bytes, frame_id):
    try:
        response = await asyncio.wait_for(
            send_image_to_server(connection, frame_id=frame_id, image=image_bytes),
            timeout=5,
        )
        if response.get("success") and response.get("user") not in (None, "Unknown"):
            track.recognized_user = response["user"]
    except Exception as error:
        print(f"Reconocimiento omitido: {error}")
    finally:
        track.recognition_pending = False


async def main(
    cam,
    *,
    server_url,
    recognition_interval,
    arduino_port=None,
    line_position=0.25,
    database=None,
):
    connection = ServerConnection(server_url)
    cap = cv2.VideoCapture(cam)
    task = None
    arduino = None
    ledger = None
    track = None
    frame_id = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara o vídeo: {cam}")
        ledger = Occupancy(
            database
            or str(Path(__file__).resolve().parents[1] / "data/occupancy.sqlite3")
        )
        if arduino_port:
            arduino = await asyncio.to_thread(Arduino, arduino_port)
        else:
            print(
                "Arduino desactivado: especifica --arduino-port COM3 (o el puerto de tu placa)."
            )
        with (
            face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            ) as detector,
            face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as mesh,
        ):
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Fin de vídeo o error de captura.")
                    break
                h, w = image.shape[:2]
                line_x = int((w - 1) * line_position)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)
                detections = results.detections or []
                display = cv2.flip(image, 1)
                # A single identity is supported. Never assign the first of several faces.
                if len(detections) > 1:
                    track = None
                    if task is not None and not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    cv2.putText(
                        display,
                        "Una persona a la vez",
                        (18, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                elif len(detections) == 1:
                    bbox = detections[0].location_data.relative_bounding_box
                    cx = (bbox.xmin + bbox.width / 2) * w
                    cy = (bbox.ymin + bbox.height / 2) * h
                    x = w - 1 - cx
                    if track is None:
                        track = PersonTrack(
                            line_x=line_x, dead_zone=max(10, int(w * 0.015))
                        )
                    track.update(x)
                    now = time.monotonic()
                    # Start recognition before crossing; keep retrying at the requested interval.
                    if (
                        track.recognized_user is None
                        and not track.recognition_pending
                        and (task is None or task.done())
                        and now - track.last_recognition_at >= recognition_interval
                    ):
                        encoded, jpeg = cv2.imencode(".jpg", image)
                        if encoded:
                            track.recognition_pending = True
                            track.last_recognition_at = now
                            frame_id += 1
                            task = asyncio.create_task(
                                recognize_track(
                                    connection, track, jpeg.tobytes(), frame_id
                                )
                            )
                    if track.recognized_user:
                        while track.pending_crossings:
                            event = track.pending_crossings.pop(0)
                            event_id = ledger.record(track.recognized_user, event)
                            if event_id is not None:
                                signal = "disabled"
                                if event is Direction.ENTRADA and arduino:
                                    try:
                                        await asyncio.to_thread(arduino.entry)
                                        signal = "acknowledged"
                                    except Exception as error:
                                        signal = "failed"
                                        print(
                                            f"Fallo de señal Arduino (sin reintento automático): {error}"
                                        )
                                if event is Direction.ENTRADA:
                                    ledger.signal_result(event_id, signal)
                                print(
                                    f"{event.value.upper()}: {track.recognized_user}; dentro: {ledger.inside}"
                                )
                    if track.is_in_recognition_zone(x):
                        mesh_results = mesh.process(rgb)
                        if mesh_results.multi_face_landmarks:
                            draw_face_mesh(image, mesh_results)
                            display = cv2.flip(image, 1)
                    draw_direction_vector(display, compute_display_vector(cx, cy, w, h))
                    cv2.putText(
                        display,
                        track.recognized_user or "Reconociendo...",
                        (18, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                elif track is not None and track.mark_missed():
                    track = None
                    if task is not None and not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                draw_dashed_line(display, line_x, (255, 0, 0))
                draw_counter(display, *ledger.totals())
                cv2.imshow("CamS", display)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                # OpenCV is synchronous: explicitly let recognition/network tasks run.
                await asyncio.sleep(0.001)
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        cap.release()
        cv2.destroyAllWindows()
        await connection.close()
        if arduino:
            arduino.close()
        if ledger:
            ledger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cam", type=str, nargs="?", default="0")
    parser.add_argument(
        "--server",
        "-s",
        type=str,
        default="ws://localhost:8765/api/ws/",
        help="Websocket server URL for face recognition.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between recognition attempts sent to the server.",
    )
    parser.add_argument("--arduino-port", help="USB serial port, e.g. COM3")
    parser.add_argument(
        "--line-position",
        type=float,
        default=0.25,
        help="Counting line as fraction of displayed width (0 to 1).",
    )
    parser.add_argument(
        "--database", help="Path to persistent SQLite occupancy database"
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval debe ser mayor que cero")
    if not 0.05 <= args.line_position <= 0.95:
        parser.error("--line-position debe estar entre 0.05 y 0.95")

    try:
        cam_arg = int(args.cam)
    except ValueError:
        cam_arg = args.cam

    asyncio.run(
        main(
            cam_arg,
            server_url=args.server,
            recognition_interval=args.interval,
            arduino_port=args.arduino_port,
            line_position=args.line_position,
            database=args.database,
        )
    )
