import cv2
import numpy as np
from fastapi import APIRouter, WebSocket

from src.api.deps import MQTTPublisherDep
from src.core.config import settings
from src.core.users import known_users
from src.schemas import WebSocketMessage, WebSocketResponse
from src.services.users import recognize_user
from src.utils import face_encodings

router = APIRouter()


@router.websocket("/")
async def recognize_user_endpoint(websocket: WebSocket, publisher: MQTTPublisherDep):
    await websocket.accept()
    while True:
        metadata = await websocket.receive_json()
        message = WebSocketMessage(**metadata)

        image_data = await websocket.receive_bytes()
        message.image = image_data

        # Send acknowledgment response
        response = WebSocketResponse(frame_id=message.frame_id)
        await websocket.send_json(response.model_dump())

        # Get image from bytes
        image_bytes = message.image
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Extract face encodings
        encodings = face_encodings(image)
        if not encodings:
            continue  # No faces detected

        # Recognize user based on the received embedding
        result = recognize_user(
            known_users=known_users,
            current_encoding=encodings,
            threshold=settings.THRESHOLD_DISTANCE,
        )

        if result.success:
            publisher.publish(
                topic="user/recognized",
                message=(
                    f"User {result.user} recognized with distance {result.distance:.4f}"
                ),
            )
