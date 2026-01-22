import argparse
import asyncio
from typing import TypeVar

import cv2
from mediapipe.python.solutions import face_mesh

from src.connection import ServerConnection
from src.drawing import draw_face_mesh
from src.request import send_image_to_server

Cam = TypeVar("Cam", int, str)


async def main(cam: Cam, *, server_url: str) -> None:
    cap = cv2.VideoCapture(cam)

    connection = ServerConnection(server_url)
    await connection.connect()

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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = mp_face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                draw_face_mesh(image, results)

            flipped_image = cv2.flip(image, 1)

            # Send image to server if face is detected
            # TODO: Optimize to send at intervals (e.g., 5 FPS), JUST IF liveness probes are previously passed.
            if results.multi_face_landmarks:
                _, img_encoded = cv2.imencode(".jpg", image)
                img_bytes = img_encoded.tobytes()
                response = await send_image_to_server(
                    server_url,
                    frame_id=0,
                    image=img_bytes,
                )
                print(f"Server response: {response}")

            cv2.imshow("MediaPipe Face Mesh", flipped_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cam",
        type=str,
        default="0",
        help="Device index of the camera.",
    )
    parser.add_argument(
        "--server",
        "-s",
        type=str,
        default="ws://localhost:8765/",
        help="Websocket server URL for face recognition.",
    )
    args = parser.parse_args()

    try:
        cam_arg = int(args.cam)
    except ValueError:
        cam_arg = args.cam

    server_url = args.server

    asyncio.run(main(cam_arg, server_url=server_url))
