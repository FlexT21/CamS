import argparse
from typing import List
import requests

import cv2
from mediapipe.python.solutions import face_mesh

from server.src.core.users import load_users
from server.src.dtypes import Cam
from server.src.schemas import FaceEncoding
from server.src.utils.drawing import draw_face_mesh


def main(cam: Cam, threshold: float, *, known_users: List[FaceEncoding], server_url: str) -> None:
    cap = cv2.VideoCapture(cam)
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

            if results.multi_face_landmarks:
                # Llamar al servidor para el reconocimiento facial
                try:
                    # Convertir imagen a bytes
                    _, img_encoded = cv2.imencode('.jpg', image)
                    img_bytes = img_encoded.tobytes()
                    
                    # Enviar al servidor
                    response = requests.post(
                        f"{server_url}/recognize",
                        json={
                            "image": img_bytes.hex(),
                            "threshold": threshold
                        },
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        user = data.get("user", "Unknown")
                        distance = data.get("distance", 0.0)
                        
                        cv2.putText(
                            flipped_image,
                            f"{user}-{distance:.2f}",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            bottomLeftOrigin=False,
                        )

                        if user != "Unknown":
                            print(f"Recognized user: {user} with distance: {distance:.2f}")
                    else:
                        print(f"Server error: {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    print(f"Error connecting to server: {e}")

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
        "--threshold",
        "-t",
        type=float,
        default=0.55,
        help="Threshold for face recognition.",
    )
    parser.add_argument(
        "--server",
        "-s",
        type=str,
        default="http://localhost:8000",
        help="Server URL for face recognition.",
    )
    args = parser.parse_args()

    try:
        cam_arg = int(args.cam)
    except ValueError:
        cam_arg = args.cam

    threshold = args.threshold
    server_url = args.server

    users = load_users()

    main(cam_arg, threshold, known_users=users, server_url=server_url)