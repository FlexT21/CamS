from fastapi import APIRouter, WebSocket

from src.core.constants import THRESHOLD_DISTANCE
from src.core.users import known_users
from src.schemas import WebSocketMessage, WebSocketResponse
from src.services.users import recognize_user

router = APIRouter()


@router.websocket("/")
async def recognize_user_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        message = WebSocketMessage(**data)

        # Send acknowledgment response
        response = WebSocketResponse(frame_id=message.frame_id)
        await websocket.send_json(response.model_dump())

        # Recognize user based on the received embedding
        result = recognize_user(
            known_users=known_users,
            current_encoding=message.embedding,
            threshold=THRESHOLD_DISTANCE,
        )

        if result.success:
            # TODO: Implement queue broker to send recognized user events to other services.
            print(f"User {result.user} recognized with distance {result.distance}")
            pass
