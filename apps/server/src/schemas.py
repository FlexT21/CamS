from typing import List, TypedDict


class FaceEncoding(TypedDict):
    user: str
    encodings: List[List[float]]
