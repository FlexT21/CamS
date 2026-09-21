import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Direction(Enum):
    ENTRADA = "entrada"
    SALIDA = "salida"


@dataclass
class PersonTrack:
    center_x: int
    left_zone_x: int
    right_zone_x: int
    max_missed_frames: int = 8

    last_centroid_x: float = 0.0
    last_side: Optional[str] = None
    start_side: Optional[str] = None
    missed_frames: int = 0
    recognized_user: Optional[str] = None
    recognition_pending: bool = False

    def side_of(self, centroid_x: float) -> Optional[str]:
        if centroid_x <= self.left_zone_x:
            return "left"
        if centroid_x >= self.right_zone_x:
            return "right"
        return None

    def is_in_recognition_zone(self, centroid_x: float) -> bool:
        # zona amplia alrededor del centro, donde asumimos que la persona
        # queda razonablemente de frente a la cámara durante el cruce
        margin = (self.right_zone_x - self.left_zone_x) * 1.5
        return (self.center_x - margin) <= centroid_x <= (self.center_x + margin)

    def update(self, centroid_x: float) -> Optional[Direction]:
        self.last_centroid_x = centroid_x
        self.missed_frames = 0

        side = self.side_of(centroid_x)
        if side is None:
            return None  # zona muerta, no confirma nada

        if self.start_side is None:
            self.start_side = side

        event = None
        if self.last_side is not None and side != self.last_side and side != self.start_side:
            # cruzó al lado contrario de donde empezó -> cruce confirmado
            event = Direction.ENTRADA if side == "right" else Direction.SALIDA

        self.last_side = side
        return event

    def mark_missed(self) -> bool:
        """Devuelve True si el track debe darse por perdido (persona salió de cuadro)."""
        self.missed_frames += 1
        return self.missed_frames > self.max_missed_frames