import cv2
import numpy as np
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from src.api.deps import MQTTPublisherDep
from src.core.config import settings
from src.core.users import known_users
from src.schemas import WebSocketMessage
from src.services.users import recognize_user
from src.utils import face_encodings

router = APIRouter()


@router.websocket("/")
async def recognize_user_endpoint(websocket: WebSocket, publisher: MQTTPublisherDep):
    await websocket.accept()
    try:
        while True:
            metadata = await websocket.receive_json()
            message = WebSocketMessage(**metadata)

            image_data = await websocket.receive_bytes()
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

            encodings = face_encodings(image)
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