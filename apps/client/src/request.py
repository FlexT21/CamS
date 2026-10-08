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

        await connection.send_message(json.dumps(metadata_message))
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

    if response.get("type", "") != "recognition_result":
        return {
            "type": "error",
            "status": "failed",
            "message": "Failed to send image to server or invalid response.",
        }

    return response


async def register_face_image(
    connection: ServerConnection,
    *,
    frame_id: int,
    username: str,
    image: bytes,
) -> ServerResponse:
    metadata: RequestMetadata = {
        "type": "register_face",
        "frame_id": frame_id,
        "device_id": "client_1",
        "username": username,
    }

    try:
        await connection.send_message(json.dumps(metadata))
        await connection.send_message(image, text=False)
        response: ServerResponse = json.loads(await connection.receive_message())
    except Exception as error:
        return {
            "type": "error",
            "status": "failed",
            "message": f"Error registering image: {error}",
        }

    return response


async def complete_registration(
    connection: ServerConnection,
    *,
    frame_id: int,
    username: str,
) -> ServerResponse:
    metadata: RequestMetadata = {
        "type": "register_complete",
        "frame_id": frame_id,
        "device_id": "client_1",
        "username": username,
    }

    try:
        await connection.send_message(json.dumps(metadata))
        response: ServerResponse = json.loads(await connection.receive_message())
    except Exception as error:
        return {
            "type": "error",
            "status": "failed",
            "message": f"Error completing registration: {error}",
        }

    return response
