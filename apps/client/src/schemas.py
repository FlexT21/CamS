from typing import TypedDict


class RequestMetadata(TypedDict):
    type: str
    frame_id: int
    device_id: str


class ServerResponse(TypedDict):
    type: str
    status: str
    message: str
