from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Direction(Enum):
    ENTRADA = "entrada"
    SALIDA = "salida"


@dataclass
class PersonTrack:
    line_x: int
    recognition_x: int | None = None
    dead_zone: int = 12
    max_missed_frames: int = 8

    last_centroid_x: float = 0.0
    last_centroid_y: float = 0.0
    last_side: Optional[str] = None
    start_side: Optional[str] = None
    missed_frames: int = 0
    recognized_user: Optional[str] = None
    access_granted: Optional[bool] = None
    access_message: Optional[str] = None
    access_message_until: float = 0.0
    recognition_pending: bool = False

    def side_of(self, centroid_x: float) -> Optional[str]:
        """Return the side of the counting line in the displayed image.

        The client mirrors the camera image before showing it. Tracking uses
        that same displayed coordinate system so the meaning of the movement
        matches what the operator sees.
        """
        if centroid_x <= self.line_x - self.dead_zone:
            return "left"
        if centroid_x >= self.line_x + self.dead_zone:
            return "right"
        return None

    def is_in_recognition_zone(self, centroid_x: float) -> bool:
        # La persona se reconoce en el centro antes de llegar a la línea.
        recognition_center = self.recognition_x or self.line_x
        margin = max(self.dead_zone * 3, 80)
        return (
            recognition_center - margin
            <= centroid_x
            <= recognition_center + margin
        )

    def update(self, centroid_x: float, centroid_y: float = 0.0) -> Optional[Direction]:
        self.last_centroid_x = centroid_x
        self.last_centroid_y = centroid_y
        self.missed_frames = 0

        side = self.side_of(centroid_x)
        if side is None:
            return None  # zona muerta, no confirma nada

        if self.start_side is None:
            self.start_side = side

        event = None
        if self.last_side is not None and side != self.last_side:
            event = Direction.SALIDA if side == "left" else Direction.ENTRADA

        self.last_side = side
        return event

    def mark_missed(self) -> bool:
        """Devuelve True si el track debe darse por perdido (persona salió de cuadro)."""
        self.missed_frames += 1
        return self.missed_frames > self.max_missed_frames
