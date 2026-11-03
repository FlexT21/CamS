import argparse
import asyncio
import time

import cv2
from mediapipe.python.solutions import face_detection, face_mesh

from src.arduino import ArduinoSignal
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
from src.occupancy import Occupancy
from src.request import complete_registration, register_face_image, send_image_to_server


async def recognize_track(
    connection: ServerConnection,
    track: PersonTrack,
    image_bytes: bytes,
    frame_id: int,
    arduino: ArduinoSignal | None,
) -> None:
    """Recognize a person and notify the Arduino only after camera approval."""
    try:
        response = await asyncio.wait_for(
            send_image_to_server(connection, frame_id=frame_id, image=image_bytes),
            timeout=5,
        )
        if response.get("success"):
            track.recognized_user = response.get("user")
            track.access_granted = True
            track.access_message = f"ACCESO ACEPTADO: {track.recognized_user}"
            print(track.access_message)
            if arduino is not None:
                arduino.send(True)
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
        track.access_message_until = time.monotonic() + 3
        track.recognition_pending = False


async def register_user(
    connection: ServerConnection,
    cap: cv2.VideoCapture,
    username: str,
    required_photos: int,
) -> None:
    """Capture guided registration photos from the camera."""
    captured = 0
    frame_id = 0

    while captured < required_photos:
        success, image = cap.read()
        if not success:
            break

        display = cv2.flip(image, 1)
        message = (
            f"Registro: {username} | Foto {captured + 1}/{required_photos} | "
            "ESPACIO captura - ESC cancela"
        )
        cv2.putText(
            display, message, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow("CamS - Registro", display)
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
    arduino_port: str | None = None,
    arduino_baudrate: int = 9600,
    line_position: float = 0.25,
    database: str = "data/occupancy.sqlite3",
) -> None:
    cap = cv2.VideoCapture(cam)
    connection = ServerConnection(server_url)
    arduino = (
        ArduinoSignal(arduino_port, arduino_baudrate)
        if arduino_port
        else None
    )
    ledger = Occupancy(database)

    try:
        await connection.connect()
        if registration_username is not None:
            await register_user(
                connection, cap, registration_username, registration_photos
            )
            return

        tracks: list[PersonTrack] = []
        entered, exited, frame_id = 0, 0, 0
        with (
            face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            ) as face_detector,
            face_mesh.FaceMesh(
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as mesh_detector,
        ):
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                height, width = image.shape[:2]
                line_x_display = int(width * line_position)
                draw_dashed_line(image, width - line_x_display, (255, 0, 0))
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detector.process(rgb_image)
                detections = []
                for detection in results.detections or []:
                    bbox = detection.location_data.relative_bounding_box
                    source_x = (bbox.xmin + bbox.width / 2) * width
                    source_y = (bbox.ymin + bbox.height / 2) * height
                    detections.append((bbox, source_x, source_y, width - source_x))

                unmatched = set(range(len(tracks)))
                matched = []
                for bbox, source_x, source_y, display_x in detections:
                    best_index = None
                    best_distance = float("inf")
                    for index in unmatched:
                        track = tracks[index]
                        distance = (
                            (display_x - track.last_centroid_x) ** 2
                            + (source_y - track.last_centroid_y) ** 2
                        ) ** 0.5
                        if distance < best_distance:
                            best_index, best_distance = index, distance
                    if best_index is not None and best_distance <= max(80, int(width * 0.12)):
                        unmatched.remove(best_index)
                        track = tracks[best_index]
                    else:
                        track = PersonTrack(
                            line_x=line_x_display,
                            dead_zone=max(10, int(width * 0.015)),
                        )
                        tracks.append(track)
                    matched.append((track, bbox, source_x, source_y, display_x))

                for index in sorted(unmatched, reverse=True):
                    if tracks[index].mark_missed():
                        tracks.pop(index)

                mesh_results = mesh_detector.process(rgb_image) if detections else None
                if mesh_results and mesh_results.multi_face_landmarks:
                    draw_face_mesh(image, mesh_results)

                displayed = cv2.flip(image, 1)
                completed = []
                for track, bbox, source_x, source_y, display_x in matched:
                    draw_direction_vector(
                        displayed, compute_display_vector(source_x, source_y, width, height)
                    )
                    event = track.update(display_x)
                    if (
                        track.side_of(display_x) == "left"
                        and track.is_in_recognition_zone(display_x)
                        and track.recognized_user is None
                        and not track.recognition_pending
                        and time.monotonic() - track.last_recognition_at
                        >= recognition_interval
                    ):
                        x1 = max(0, int(bbox.xmin * width))
                        y1 = max(0, int(bbox.ymin * height))
                        x2 = min(width, int((bbox.xmin + bbox.width) * width))
                        y2 = min(height, int((bbox.ymin + bbox.height) * height))
                        face_image = image[y1:y2, x1:x2]
                        if face_image.size:
                            track.recognition_pending = True
                            track.last_recognition_at = time.monotonic()
                            _, encoded = cv2.imencode(".jpg", face_image)
                            await recognize_track(
                                connection, track, encoded.tobytes(), frame_id, arduino
                            )
                            frame_id += 1

                    if event is not None:
                        user = track.recognized_user or "Desconocido"
                        event_id = None
                        if event is Direction.ENTRADA and track.access_granted:
                            event_id = ledger.record(user, event)
                            if event_id is not None:
                                entered += 1
                        elif event is Direction.SALIDA:
                            event_id = ledger.record(user, event)
                            if event_id is not None:
                                exited += 1
                        if event_id is not None:
                            ledger.signal_result(event_id, "camera_accepted")
                        print(
                            f"Evento: {event.value.upper()} — usuario: {user} — "
                            f"dentro: {ledger.inside}"
                        )
                        completed.append(track)

                tracks = [track for track in tracks if track not in completed]
                draw_counter(displayed, entered, exited, ledger.inside)
                for track, *_ in matched:
                    if track.access_message_until > time.monotonic():
                        draw_access_status(displayed, track.access_message)
                        break
                cv2.imshow("CamS", displayed)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    finally:
        ledger.close()
        cap.release()
        cv2.destroyAllWindows()
        await connection.close()
        if arduino is not None:
            arduino.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cam", type=str, nargs="?", default="0")
    parser.add_argument("--server", "-s", default="ws://localhost:8765/api/ws/")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--arduino-port", default="COM3")
    parser.add_argument("--arduino-baudrate", type=int, default=9600)
    parser.add_argument("--line-position", type=float, default=0.25)
    parser.add_argument("--database", default="data/occupancy.sqlite3")
    parser.add_argument("--register", metavar="USERNAME", nargs="?", const="")
    parser.add_argument("--photos", type=int, default=3)
    parser.add_argument("--admin", action="store_true")
    args = parser.parse_args()

    if args.admin:
        from src.admin import run_admin_interface

        run_admin_interface(
            args.server,
            arduino_port=args.arduino_port,
            arduino_baudrate=args.arduino_baudrate,
        )
        raise SystemExit(0)
    if args.register == "":
        args.register = input("Nombre de usuario: ").strip()
    if args.register is not None and not args.register:
        parser.error("--register requiere un nombre.")
    if args.interval <= 0 or args.photos < 1:
        parser.error("--interval debe ser mayor que cero y --photos debe ser válido.")
    if not 0.05 <= args.line_position <= 0.95:
        parser.error("--line-position debe estar entre 0.05 y 0.95.")

    try:
        camera = int(args.cam)
    except ValueError:
        camera = args.cam
    asyncio.run(
        main(
            camera,
            server_url=args.server,
            recognition_interval=args.interval,
            registration_username=args.register,
            registration_photos=args.photos,
            arduino_port=args.arduino_port,
            arduino_baudrate=args.arduino_baudrate,
            line_position=args.line_position,
            database=args.database,
        )
    )
