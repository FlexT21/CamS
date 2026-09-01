import argparse
import asyncio
import time
from typing import TypeVar

import cv2
from mediapipe.python.solutions import face_mesh

from src.connection import ServerConnection
from src.drawing import draw_face_mesh
from src.request import send_image_to_server

Cam = TypeVar("Cam", int, str)


async def main(cam: Cam, *, server_url: str, recognition_interval: float) -> None:
    cap = cv2.VideoCapture(cam)

    connection = ServerConnection(server_url)
    await connection.connect()

    last_recognition = 0.0
    current_user = "Unknown"
    current_distance = None

    with face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as mp_face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                if isinstance(cam, int):
                    continue
                break

            image.flags.writeable = False
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb_image)
            image.flags.writeable = True
            image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                draw_face_mesh(image, results)

                current_time = time.monotonic()
                if current_time - last_recognition >= recognition_interval:
                    last_recognition = current_time

                    _, img_encoded = cv2.imencode(".jpg", image)
                    response = await send_image_to_server(
                        connection,
                        frame_id=0,
                        image=img_encoded.tobytes(),
                    )

                    if response.get("type") == "recognition_result":
                        current_user = response.get("user", "Unknown")
                        current_distance = response.get("distance")
                    print(f"Server response: {response}")
            else:
                current_user = "Unknown"
                current_distance = None

            flipped_image = cv2.flip(image, 1)

            label = f"{current_user}"
            if current_distance is not None:
                label += f" - {current_distance:.2f}"

            cv2.putText(
                flipped_image, label, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )

            cv2.imshow("MediaPipe Face Mesh", flipped_image)
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