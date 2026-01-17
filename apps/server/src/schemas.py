from typing import List, TypedDict

from src.dtypes import Arr


class FaceEncoding(TypedDict):
    user: str
    encodings: List[Arr]
