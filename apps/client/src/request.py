import json

from src.connection import ServerConnection
from src.schemas import RequestMetadata, ServerResponse


async def send_image_to_server(
    connection: ServerConnection, *, frame_id: int, image: bytes
) -> ServerResponse:
    try:
        metadata_message: RequestMetadata = {
            "type": "face_image",
            "frame_id": frame_id,
            # This device_id is intended to be unique per client
            # using .env or other configuration methods in a real application.
            "device_id": "client_1",
        }

        await connection.send_message(json.dumps(metadata_message), text=True)
        await connection.send_message(image, text=False)
        response_message = await connection.receive_message()
        response: ServerResponse = json.loads(response_message)
    except Exception as e:
        print(f"Error sending image to server: {e}")
        return {
            "type": "error",
            "status": "failed",
            "message": f"Error sending image to server: {e}",
        }

    if response.get("type", "") != "ack":
        return {
            "type": "error",
            "status": "failed",
            "message": "Failed to send image to server or invalid response.",
        }

    return response
