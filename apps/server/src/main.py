import argparse
from typing import List

import cv2
from mediapipe.python.solutions import face_mesh

from src.core.users import load_users, recognize_user
from src.core.utils import face_encodings
from src.dtypes import Cam
from src.schemas import FaceEncoding
from src.utils.drawing import draw_face_mesh


def main(cam: Cam, threshold: float, *, known_users: List[FaceEncoding]) -> None:
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

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image in RGB format
            results = mp_face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                # Draw the face mesh annotations on the image.
                draw_face_mesh(image, results)

            # Flip the image horizontally for a selfie-view display.
            flipped_image = cv2.flip(image, 1)

            if results.multi_face_landmarks:
                # TODO: Implement logic to handle when the face has closed eyes or is not fully visible
                # and all that kind of stuff to prevent false positives.

                # Face recognition logic
                encodings = face_encodings(image)
                if encodings:
                    user, distance = recognize_user(
                        known_users, encodings[0], threshold
                    )
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
    args = parser.parse_args()

    try:
        cam_arg = int(args.cam)
    except ValueError:
        cam_arg = args.cam

    threshold = args.threshold

    # Load known users
    users = load_users()

    main(cam_arg, threshold, known_users=users)
