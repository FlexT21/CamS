import argparse
import asyncio
import time
from typing import TypeVar

import cv2
from mediapipe.python.solutions import face_detection, face_mesh as mp_face_mesh_module

from src.direction_tracker import Direction, PersonTrack
from src.connection import ServerConnection
from src.drawing import (
    compute_display_vector,
    draw_counter,
    draw_dashed_line,
    draw_face_mesh,
    draw_direction_vector,
)
from src.request import send_image_to_server

Cam = TypeVar("Cam", int, str)


async def recognize_track(connection, track: PersonTrack, image_bytes: bytes) -> None:
    """Recognize a track without blocking the camera loop indefinitely."""
    try:
        response = await asyncio.wait_for(
            send_image_to_server(connection, frame_id=0, image=image_bytes),
            timeout=2.5,
        )
        if response.get("success"):
            track.recognized_user = response.get("user")
        else:
            print(
                "Reconocimiento sin coincidencia: "
                f"estado={response.get('status')}, "
                f"distancia={response.get('distance')}"
            )
    except Exception as error:
        print(f"Reconocimiento omitido: {error}")
    finally:
        track.recognition_pending = False


async def main(cam, *, server_url: str, recognition_interval: float) -> None:
    cap = cv2.VideoCapture(cam)
    connection = ServerConnection(server_url)
    await connection.connect()

    track: PersonTrack | None = None
    entered = 0
    exited = 0

    with (
        face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector,
        mp_face_mesh_module.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) as mesh_detector,
    ):
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            h, w = image.shape[:2]
            # La línea se dibuja en el frame original y luego se espeja para
            # que aparezca en el lado derecho de la ventana.
            line_x_display = int(w * 0.82)
            line_x_source = w - line_x_display
            draw_dashed_line(image, line_x_source, (255, 0, 0))

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 1. Tracking continuo, siempre activo
            detection_results = face_detector.process(rgb_image)
            centroid_x = None
            centroid_y = None
            if detection_results.detections:
                bbox = detection_results.detections[0].location_data.relative_bounding_box
                centroid_x = (bbox.xmin + bbox.width / 2) * w
                centroid_y = (bbox.ymin + bbox.height / 2) * h

            flipped_image = cv2.flip(image, 1)
            if centroid_x is not None:
                display_centroid_x = w - centroid_x
                vector = compute_display_vector(centroid_x, centroid_y, w, h)
                draw_direction_vector(flipped_image, vector)
                if track is None:
                    track = PersonTrack(
                        line_x=line_x_display,
                        dead_zone=max(10, int(w * 0.015)),
                    )

                # 2. Solo dentro de la zona de reconocimiento se corre FaceMesh (visual) + reconocimiento
                if track.is_in_recognition_zone(display_centroid_x):
                    mesh_results = mesh_detector.process(rgb_image)
                    if mesh_results.multi_face_landmarks:
                        draw_face_mesh(image, mesh_results)

                    if (
                        track.recognized_user is None
                        and not track.recognition_pending
                    ):
                        track.recognition_pending = True
                        _, img_encoded = cv2.imencode(".jpg", image)
                        await recognize_track(
                            connection,
                            track,
                            img_encoded.tobytes(),
                        )
                else:
                    cv2.circle(image, (int(centroid_x), int(bbox.ymin * h)), 6, (0, 0, 255), -1)

                event = track.update(display_centroid_x)
                if event is not None:
                    if event is Direction.ENTRADA:
                        entered += 1
                    else:
                        exited += 1
                    inside = max(0, entered - exited)
                    user = track.recognized_user or "Desconocido"
                    print(
                        f"Evento: {event.value.upper()} — usuario: {user} — "
                        f"dentro: {inside}"
                    )
                    track = None

            elif track is not None and track.mark_missed():
                track = None

            draw_counter(flipped_image, entered, exited, max(0, entered - exited))
            cv2.imshow("CamS", flipped_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    await connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cam", type=str, nargs="?", default="0")
    parser.add_argument(
        "--server", "-s", type=str,
        default="ws://localhost:8765/api/ws/",
        help="Websocket server URL for face recognition.",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Seconds between recognition attempts sent to the server.",
    )
    args = parser.parse_args()

    try:
        cam_arg = int(args.cam)
    except ValueError:
        cam_arg = args.cam

    asyncio.run(main(cam_arg, server_url=args.server, recognition_interval=args.interval))
