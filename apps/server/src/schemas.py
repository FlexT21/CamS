from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from src.dtypes import Arr


class FaceEncoding(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user: str
    encodings: List[Arr]


class RecognizeUserResponse(BaseModel):
    user: str
    success: bool
    distance: Optional[float] = None


class HealthcheckResponse(BaseModel):
    status: str
    message: str
    extra: Dict[str, Any]
