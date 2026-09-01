from typing import NotRequired, Optional, TypedDict


class RequestMetadata(TypedDict):
    type: str
    frame_id: int
    device_id: str


class ServerResponse(TypedDict):
    type: str
    status: str
    message: NotRequired[str]
    user: NotRequired[str]
    success: NotRequired[bool]
    distance: NotRequired[Optional[float]]