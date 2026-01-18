import json
from typing import Dict

from websockets.asyncio.client import connect


async def send_image_to_server(
    server_url: str, *, frame_id: int, image: bytes
) -> Dict[str, str]:
    error = False
    try:
        async with connect(server_url) as websocket:
            metadata_message = {
                "type": "face_image",
                "frame_id": frame_id,
                # This device_id is intended to be unique per client
                # using .env or other configuration methods in a real application.
                "device_id": "client_1",
            }

            await websocket.send(json.dumps(metadata_message), text=True)
            await websocket.send(image, text=False)

            response = await websocket.recv()
            response = json.loads(response)
    except Exception as e:
        print(f"Error sending image to server: {e}")
        error = True

    if response.get("type", "") != "ack" or error:
        response = {
            "type": "error",
            "status": "failed",
            "message": "Failed to send image to server or invalid response.",
        }

    return response
