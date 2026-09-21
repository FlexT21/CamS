import argparse
import asyncio
import time
from typing import TypeVar

import cv2
from mediapipe.python.solutions import face_mesh
from mediapipe.python.solutions import face_detection

from src.direction_tracker import Direction, PersonTrack
from src.connection import ServerConnection
from src.drawing import draw_face_mesh
from src.request import send_image_to_server

Cam = TypeVar("Cam", int, str)


async def main(cam, *, server_url: str, recognition_interval: float) -> None:
    cap = cv2.VideoCapture(cam)
    connection = ServerConnection(server_url)
    await connection.connect()

    track: PersonTrack | None = None

    with face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            h, w = image.shape[:2]
            cv2.line(image, (w // 2, 0), (w // 2, h), (255, 0, 0), 2)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_image)

            centroid_x = None
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                centroid_x = (bbox.xmin + bbox.width / 2) * w
                cv2.circle(image, (int(centroid_x), int(bbox.ymin * h)), 6, (0, 0, 255), -1)

            if centroid_x is not None:
                if track is None:
                    margin = int(w * 0.08)
                    track = PersonTrack(center_x=w // 2, left_zone_x=w // 2 - margin, right_zone_x=w // 2 + margin)

                event = track.update(centroid_x)

                # dispara reconocimiento una sola vez por track, cuando pasa por la zona frontal
                if (
                    track.recognized_user is None
                    and not track.recognition_pending
                    and track.is_in_recognition_zone(centroid_x)
                ):
                    track.recognition_pending = True
                    _, img_encoded = cv2.imencode(".jpg", image)
                    response = await send_image_to_server(connection, frame_id=0, image=img_encoded.tobytes())
                    if response.get("success"):
                        track.recognized_user = response.get("user")
                    track.recognition_pending = False

                if event is not None:
                    user = track.recognized_user or "Desconocido"
                    print(f"Evento: {event.value.upper()} — usuario: {user}")
                    # aquí publicas/registras el evento con user + event.value
                    track = None  # el cruce ya se cerró, se libera el track

            elif track is not None and track.mark_missed():
                # se perdió de vista sin llegar a cruzar del todo -> se descarta
                track = None

            cv2.imshow("CamS", cv2.flip(image, 1))
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