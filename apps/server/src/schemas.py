from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict

from src.dtypes import Arr


class FaceEncoding(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user: str
    encodings: List[Arr]


class RecognizeUserResponse(BaseModel):
    user: str
    success: bool
    distance: float


class HealthcheckResponse(BaseModel):
    status: str
    message: str
    extra: Dict[str, Any]


class WebSocketMessage(BaseModel):
    type: str
    frame_id: int
    device_id: str
    embedding: List[float]


class WebSocketResponse(BaseModel):
    type: str = "ack"
    frame_id: int
    status: str = "accepted"
