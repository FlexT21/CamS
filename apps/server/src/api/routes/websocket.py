from fastapi import APIRouter, WebSocket

from src.schemas import WebSocketMessage, WebSocketResponse

router = APIRouter()


@router.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        message = WebSocketMessage(**data)
        
        # TODO: Logic to process the embedding goes here.

        response = WebSocketResponse(frame_id=message.frame_id)
        await websocket.send_json(response.model_dump())
