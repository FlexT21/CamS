import argparse
import asyncio
import time
from typing import TypeVar

import cv2
from mediapipe.python.solutions import face_detection
from mediapipe.python.solutions import face_mesh as mp_face_mesh_module

from src.connection import ServerConnection
from src.direction_tracker import Direction, PersonTrack
from src.drawing import (
    compute_display_vector,
    draw_access_status,
    draw_counter,
    draw_dashed_line,
    draw_direction_vector,
    draw_face_mesh,
)
from src.request import complete_registration, register_face_image, send_image_to_server

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
            track.access_granted = True
            track.access_message = f"ACCESO ACEPTADO: {track.recognized_user}"
            print(track.access_message)
        else:
            track.access_granted = False
            track.access_message = (
                "ACCESO DENEGADO: reconocimiento sin coincidencia: "
                f"estado={response.get('status')}, "
                f"distancia={response.get('distance')}"
            )
            print(track.access_message)
    except Exception as error:
        track.access_granted = False
        track.access_message = f"ACCESO DENEGADO: reconocimiento omitido: {error}"
        print(track.access_message)
    finally:
        track.access_message_until = time.monotonic() + 3.0
        track.recognition_pending = False


async def register_user(
    connection: ServerConnection,
    cap: cv2.VideoCapture,
    username: str,
    required_photos: int,
) -> None:
    """Capture guided registration photos from the same camera window."""
    captured = 0
    frame_id = 0
    window_name = "CamS - Registro"

    while captured < required_photos:
        success, image = cap.read()
        if not success:
            continue

        display = cv2.flip(image, 1)
        message = (
            f"Registro: {username} | Foto {captured + 1}/{required_photos} | "
            "ESPACIO captura - ESC cancela"
        )
        cv2.putText(
            display, message, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            display, message, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(5) & 0xFF

        if key == 27:
            print("Registro cancelado.")
            return
        if key != ord(" "):
            continue

        _, encoded_image = cv2.imencode(".jpg", image)
        response = await register_face_image(
            connection,
            frame_id=frame_id,
            username=username,
            image=encoded_image.tobytes(),
        )
        frame_id += 1
        if response.get("success"):
            captured += 1
            print(f"Foto {captured}/{required_photos} registrada.")
        else:
            print(response.get("message", "No se pudo registrar la foto."))

    response = await complete_registration(
        connection, frame_id=frame_id, username=username
    )
    if response.get("success"):
        print(f"Usuario '{username}' registrado correctamente.")
    else:
        print(response.get("message", "No se pudo completar el registro."))


async def main(
    cam,
    *,
    server_url: str,
    recognition_interval: float,
    registration_username: str | None = None,
    registration_photos: int = 3,
) -> None:
    cap = cv2.VideoCapture(cam)
    connection = ServerConnection(server_url)
    await connection.connect()

    if registration_username is not None:
        await register_user(
            connection, cap, registration_username, registration_photos
        )
        cap.release()
        cv2.destroyAllWindows()
        await connection.close()
        return

    tracks: list[PersonTrack] = []
    entered = 0
    exited = 0

    with (
        face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector,
        mp_face_mesh_module.FaceMesh(
            max_num_faces=5, refine_landmarks=True,
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
            line_x_display = int(w * 0.82) - 15
            line_x_source = w - line_x_display
            draw_dashed_line(image, line_x_source, (255, 0, 0))

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 1. Tracking continuo, siempre activo
            detection_results = face_detector.process(rgb_image)
            detections = []
            for detection in detection_results.detections or []:
                bbox = detection.location_data.relative_bounding_box
                centroid_x = (bbox.xmin + bbox.width / 2) * w
                centroid_y = (bbox.ymin + bbox.height / 2) * h
                detections.append((bbox, centroid_x, centroid_y, w - centroid_x))

            # Asociar cada detección con el track más cercano para no mezclar personas.
            unmatched_tracks = set(range(len(tracks)))
            matched_tracks = []
            max_match_distance = max(80, int(w * 0.12))
            for bbox, centroid_x, centroid_y, display_centroid_x in detections:
                best_index = None
                best_distance = float("inf")
                for index in unmatched_tracks:
                    track = tracks[index]
                    distance = (
                        (display_centroid_x - track.last_centroid_x) ** 2
                        + (centroid_y - track.last_centroid_y) ** 2
                    ) ** 0.5
                    if distance < best_distance:
                        best_index = index
                        best_distance = distance

                if best_index is not None and best_distance <= max_match_distance:
                    unmatched_tracks.remove(best_index)
                    track = tracks[best_index]
                else:
                    track = PersonTrack(
                        line_x=line_x_display,
                        recognition_x=w // 2,
                        dead_zone=max(10, int(w * 0.015)),
                    )
                    tracks.append(track)
                matched_tracks.append((track, bbox, centroid_x, centroid_y, display_centroid_x))

            for index in sorted(unmatched_tracks, reverse=True):
                if tracks[index].mark_missed():
                    tracks.pop(index)

            mesh_results = mesh_detector.process(rgb_image) if detections else None
            if mesh_results and mesh_results.multi_face_landmarks:
                draw_face_mesh(image, mesh_results)

            flipped_image = cv2.flip(image, 1)
            completed_tracks = []
            for track, bbox, centroid_x, centroid_y, display_centroid_x in matched_tracks:
                vector = compute_display_vector(centroid_x, centroid_y, w, h)
                draw_direction_vector(flipped_image, vector)

                if (
                    track.is_in_recognition_zone(display_centroid_x)
                    and track.last_side != "right"
                ):
                    if track.recognized_user is None and not track.recognition_pending:
                        x1 = max(0, int(bbox.xmin * w))
                        y1 = max(0, int(bbox.ymin * h))
                        x2 = min(w, int((bbox.xmin + bbox.width) * w))
                        y2 = min(h, int((bbox.ymin + bbox.height) * h))
                        face_image = image[y1:y2, x1:x2]
                        if face_image.size:
                            track.recognition_pending = True
                            _, img_encoded = cv2.imencode(".jpg", face_image)
                            await recognize_track(
                                connection,
                                track,
                                img_encoded.tobytes(),
                            )
                else:
                    cv2.circle(
                        image,
                        (int(centroid_x), int(bbox.ymin * h)),
                        6,
                        (0, 0, 255),
                        -1,
                    )

                event = track.update(display_centroid_x, centroid_y)
                if event is not None:
                    if event is Direction.ENTRADA and track.access_granted:
                        entered += 1
                    elif event is Direction.SALIDA:
                        exited += 1
                    inside = max(0, entered - exited)
                    user = track.recognized_user or "Desconocido"
                    print(
                        f"Evento: {event.value.upper()} — usuario: {user} — "
                        f"dentro: {inside}"
                    )
                    completed_tracks.append(track)

            tracks = [track for track in tracks if track not in completed_tracks]

            draw_counter(flipped_image, entered, exited, max(0, entered - exited))
            for track in matched_tracks:
                if track[0].access_message_until > time.monotonic():
                    draw_access_status(flipped_image, track[0].access_message)
                    break
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
    parser.add_argument(
        "--register", metavar="USERNAME", nargs="?", const="",
        help="Open the camera registration interface for a user.",
    )
    parser.add_argument(
        "--photos", type=int, default=3,
        help="Number of valid photos required during registration.",
    )
    args = parser.parse_args()

    try:
        cam_arg = int(args.cam)
    except ValueError:
        cam_arg = args.cam

    registration_username = args.register
    if registration_username == "":
        registration_username = input("Nombre de usuario: ").strip()
    if registration_username is not None and not registration_username:
        parser.error("El nombre de usuario no puede estar vacío.")
    if args.photos < 1:
        parser.error("--photos debe ser mayor que cero.")

    asyncio.run(
        main(
            cam_arg,
            server_url=args.server,
            recognition_interval=args.interval,
            registration_username=registration_username,
            registration_photos=args.photos,
        )
    )
