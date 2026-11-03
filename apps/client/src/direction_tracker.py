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
    last_centroid_y: float = 0.0
    last_side: Optional[str] = None
    start_side: Optional[str] = None
    missed_frames: int = 0
    recognized_user: Optional[str] = None
    access_granted: Optional[bool] = None
    access_message: Optional[str] = None
    access_message_until: float = 0.0
    recognition_pending: bool = False
    last_recognition_at: float = float("-inf")
    pending_crossings: list[Direction] = field(default_factory=list)

    def side_of(self, centroid_x: float) -> Optional[str]:
        """Return the side of the counting line in the displayed image.

        The client mirrors the camera image before showing it. Tracking uses
        that same displayed coordinate system so the meaning of the movement
        matches what the operator sees: right -> left is an entry.
        """
        if centroid_x <= self.line_x - self.dead_zone:
            return "left"
        if centroid_x >= self.line_x + self.dead_zone:
            return "right"
        return None

    def is_in_recognition_zone(self, centroid_x: float) -> bool:
        # zona amplia alrededor del centro, donde asumimos que la persona
        # queda razonablemente de frente a la cámara durante el cruce
        margin = max(self.dead_zone * 3, 100)
        return (self.line_x - margin) <= centroid_x <= (self.line_x + margin)

    def update(
        self, centroid_x: float, centroid_y: float = 0.0
    ) -> Optional[Direction]:
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
            event = Direction.ENTRADA if side == "left" else Direction.SALIDA
            self.pending_crossings.append(event)

        self.last_side = side
        return event

    def mark_missed(self) -> bool:
        """Devuelve True si el track debe darse por perdido (persona salió de cuadro)."""
        self.missed_frames += 1
        return self.missed_frames > self.max_missed_frames
