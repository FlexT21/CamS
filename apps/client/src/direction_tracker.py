import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Direction(Enum):
    ENTRADA = "entrada"
    SALIDA = "salida"


@dataclass
class PersonTrack:
    line_x: int
    dead_zone: int = 12
    max_missed_frames: int = 8

    last_centroid_x: float = 0.0
    last_side: Optional[str] = None
    start_side: Optional[str] = None
    missed_frames: int = 0
    recognized_user: Optional[str] = None
    recognition_pending: bool = False

    def side_of(self, centroid_x: float) -> Optional[str]:
        """Return the side of the counting line in the displayed image.

        The client mirrors the camera image before showing it. Tracking uses
        that same displayed coordinate system so the meaning of the movement
        matches what the operator sees: right -> left is an exit.
        """
        if centroid_x <= self.line_x - self.dead_zone:
            return "left"
        if centroid_x >= self.line_x + self.dead_zone:
            return "right"
        return None

    def is_in_recognition_zone(self, centroid_x: float) -> bool:
        # zona amplia alrededor del centro, donde asumimos que la persona
        # queda razonablemente de frente a la cámara durante el cruce
        margin = max(self.dead_zone * 3, 80)
        return (self.line_x - margin) <= centroid_x <= (self.line_x + margin)

    def update(self, centroid_x: float) -> Optional[Direction]:
        self.last_centroid_x = centroid_x
        self.missed_frames = 0

        side = self.side_of(centroid_x)
        if side is None:
            return None  # zona muerta, no confirma nada

        if self.start_side is None:
            self.start_side = side

        event = None
        if self.last_side is not None and side != self.last_side:
            # En la imagen mostrada: derecha -> izquierda = salida;
            # izquierda -> derecha = entrada.
            event = Direction.SALIDA if side == "left" else Direction.ENTRADA

        self.last_side = side
        return event

    def mark_missed(self) -> bool:
        """Devuelve True si el track debe darse por perdido (persona salió de cuadro)."""
        self.missed_frames += 1
        return self.missed_frames > self.max_missed_frames
