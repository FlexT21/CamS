import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from fastapi import HTTPException

from .vision import MediaPipeTracker, match


@dataclass
class Track:
    seen: float = 0
    # Use the same mirrored coordinates as the displayed video and JPEG.
    # Keep the last stable side and identity across crossings and returns.
    start_side: int = 0
    last_side: int = 0
    sequence: int = 0
    generation: str = field(default_factory=lambda: uuid.uuid4().hex)
    user: dict | None = None
    identity_candidate: dict | None = None
    identity_count: int = 0
    recognition_at: float = -1
    pending_crossings: list = field(default_factory=list)

    def crossing(self, center_x, line, dead_zone, confirmations=None):
        side = (
            -1
            if center_x < line - dead_zone
            else (1 if center_x > line + dead_zone else 0)
        )
        if side == 0:
            return None
        if not self.start_side:
            self.start_side = side
        event = None
        if self.last_side and side != self.last_side:
            self.sequence += 1
            event = "entrada" if side == 1 else "salida"
        self.last_side = side
        return event


@dataclass
class MonitorSession:
    touched: float
    lock: object = field(default_factory=threading.Lock)
    tracker: object = None
    tracks: dict = field(default_factory=dict)
    generation: str = field(default_factory=lambda: uuid.uuid4().hex)
    last_frame: float = -1
    dimensions: tuple | None = None
    pending: list = field(default_factory=list)
    pending_access: list = field(default_factory=list)


class Monitor:
    def __init__(self, config, faces, tracker_factory=MediaPipeTracker):
        self.config, self.faces, self.tracker_factory = config, faces, tracker_factory
        self.lock = threading.Lock()
        self.sessions = {}

    @contextmanager
    def session(self, owner, session_id):
        now = time.monotonic()
        key = (owner, session_id)
        with self.lock:
            for expired_key, value in list(self.sessions.items()):
                if now - value.touched > self.config.monitor_ttl and value.lock.acquire(
                    blocking=False
                ):
                    del self.sessions[expired_key]
                    value.lock.release()
            session = self.sessions.get(key)
            if session is None:
                if (
                    len(self.sessions) >= self.config.max_sessions
                    or sum(k[0] == owner for k in self.sessions)
                    >= self.config.max_sessions_per_owner
                ):
                    raise HTTPException(
                        429, "Monitor session limit reached; wait for session TTL"
                    )
                session = MonitorSession(now)
                self.sessions[key] = session
            if not session.lock.acquire(blocking=False):
                raise HTTPException(
                    409, "A frame is already processing for this session"
                )
            session.touched = now
        try:
            if (
                session.last_frame >= 0
                and now - session.last_frame < self.config.frame_interval
            ):
                raise HTTPException(429, "Frames arriving too quickly")
            session.last_frame = now
            yield session
        finally:
            session.touched = time.monotonic()
            session.lock.release()

    def process(self, session, session_id, frame, persons):
        config = self.config
        now = time.monotonic()
        height, width = frame.shape[:2]
        if session.dimensions and session.dimensions != (width, height):
            raise HTTPException(409, "Frame dimensions changed; use a new session_id")
        # Check the existing dlib engine before opening the MediaPipe detector.
        with self.faces.lock:
            self.faces._load()
        if session.tracker is None:
            session.tracker = self.tracker_factory(config)
        session.dimensions = (width, height)
        detections = session.tracker.detect(frame)
        # A single full-frame YuNet pass gives the visual overlay its actual
        # facial keypoints. Recognition itself remains scoped to each person
        # box below, as before.
        line_x = width * config.line_fraction
        dead_zone = max(10, width * config.dead_zone_fraction)
        mesh_margin = max(dead_zone * 3, 80)
        near_line = any(
            abs((x1 + x2) / 2 - line_x) <= mesh_margin for _, x1, _, x2, _ in detections
        )
        face_landmarks = (
            self.faces.landmarks(frame)
            if near_line and hasattr(self.faces, "landmarks")
            else []
        )
        session.tracks = {
            key: value
            for key, value in session.tracks.items()
            if now - value.seen <= config.track_ttl
        }
        output = []
        seen_ids = set()
        for track_id, x1, y1, x2, y2 in detections[: config.max_tracks]:
            if track_id in seen_ids:
                continue
            seen_ids.add(track_id)
            x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
            y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            if (
                track_id not in session.tracks
                and len(session.tracks) >= config.max_tracks
            ):
                continue
            track = session.tracks.setdefault(track_id, Track())
            track.seen = now
            if (
                track.user is None
                and now - track.recognition_at >= config.recognition_interval
                and persons
            ):
                track.recognition_at = now
                embeddings = self.faces.embeddings(frame)
                user = (
                    match(embeddings[0], persons, config.face_threshold)
                    if len(embeddings) == 1
                    else None
                )
                if user:
                    track.user = user
                    session.pending_access.append(user["name"])
            center = (x1 + x2) / 2
            line_x = width * config.line_fraction
            direction = track.crossing(center, line_x, dead_zone)
            if direction:
                track.pending_crossings.append((direction, track.sequence))
            # Same rule as the existing client: wait for a recognized identity.
            # A crossing pending identification survives while the track lives.
            if track.user:
                for saved_direction, sequence in track.pending_crossings:
                    session.pending.append(
                        {
                            "session_id": session_id,
                            "track_id": track_id,
                            "direction": saved_direction,
                            "person_id": track.user["id"],
                            "person_name": track.user["name"],
                            "crossing_key": f"{session.generation}:{track.generation}:{sequence}",
                        }
                    )
                track.pending_crossings.clear()
            face = self._face_in_person_box(face_landmarks, x1, y1, x2, y2)
            output.append(
                {
                    "id": track_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": center,
                    "user": track.user["name"] if track.user else None,
                    "distance_to_line": abs(center - line_x),
                    "face": face,
                }
            )
            # Retain the stable side and recognized identity. Deleting this
            # state here loses quick returns and forces recognition again.
        return output

    @staticmethod
    def _face_in_person_box(faces, x1, y1, x2, y2):
        """Choose measured facial landmarks inside the tracked face box."""
        candidates = []
        for face in faces:
            center_x = face["x"] + face["width"] / 2
            center_y = face["y"] + face["height"] / 2
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                candidates.append(face)
        return max(
            candidates, key=lambda item: item["width"] * item["height"], default=None
        )

    def count(self, owner):
        with self.lock:
            return sum(
                key[0] == owner
                and time.monotonic() - value.touched <= self.config.monitor_ttl
                for key, value in self.sessions.items()
            )

    def loaded(self, owner):
        with self.lock:
            return any(
                key[0] == owner
                and value.tracker is not None
                and time.monotonic() - value.touched <= self.config.monitor_ttl
                for key, value in self.sessions.items()
            )
