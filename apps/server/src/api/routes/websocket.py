import cv2
import numpy as np
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from src.api.deps import MQTTPublisherDep
from src.core.config import settings
from src.core.users import ensure_known_users_loaded, known_users, reload_known_users
from src.schemas import WebSocketMessage
from src.services.users import recognize_user
from src.utils import USERSDIR, face_encodings

router = APIRouter()


def _safe_username(username: str | None) -> str | None:
    if username is None:
        return None
    cleaned = username.strip()
    if not cleaned or cleaned in {".", ".."}:
        return None
    if any(character in cleaned for character in '/\\'):
        return None
    return cleaned


@router.websocket("/")
async def recognize_user_endpoint(websocket: WebSocket, publisher: MQTTPublisherDep):
    await websocket.accept()
    try:
        while True:
            metadata = await websocket.receive_json()
            message = WebSocketMessage(**metadata)

            image_data = await websocket.receive_bytes()
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            encodings = face_encodings(image)

            if message.type == "register_face":
                username = _safe_username(message.username)
                if username is None:
                    await websocket.send_json({
                        "type": "registration_result",
                        "frame_id": message.frame_id,
                        "status": "invalid_username",
                        "success": False,
                        "message": "El nombre de usuario no es válido.",
                    })
                    continue
                if len(encodings) != 1 or image is None:
                    await websocket.send_json({
                        "type": "registration_result",
                        "frame_id": message.frame_id,
                        "status": "face_required",
                        "success": False,
                        "message": "La foto debe contener exactamente un rostro.",
                    })
                    continue

                user_directory = USERSDIR / username
                user_directory.mkdir(parents=True, exist_ok=True)
                photo_path = user_directory / f"capture_{message.frame_id:04d}.jpg"
                if not cv2.imwrite(str(photo_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    await websocket.send_json({
                        "type": "registration_result",
                        "frame_id": message.frame_id,
                        "status": "save_failed",
                        "success": False,
                        "message": "No se pudo guardar la foto.",
                    })
                    continue

                reload_known_users()
                await websocket.send_json({
                    "type": "registration_result",
                    "frame_id": message.frame_id,
                    "status": "ok",
                    "success": True,
                    "message": f"Foto guardada para {username}.",
                })
                continue

            if not encodings:
                await websocket.send_json({
                    "type": "recognition_result",
                    "frame_id": message.frame_id,
                    "status": "no_face",
                    "user": "Unknown",
                    "success": False,
                    "distance": None,
                })
                continue

            ensure_known_users_loaded()
            result = recognize_user(
                known_users=known_users,
                current_encoding=encodings[0],
                threshold=settings.THRESHOLD_DISTANCE,
            )

            if result.success:
                publisher.publish(
                    topic="user/recognized",
                    message=f"User {result.user} recognized with distance {result.distance:.4f}",
                )

            await websocket.send_json({
                "type": "recognition_result",
                "frame_id": message.frame_id,
                "status": "ok",
                "user": result.user,
                "success": result.success,
                "distance": result.distance,
            })
    except WebSocketDisconnect:
        print(f"Client disconnected: {websocket.client}")