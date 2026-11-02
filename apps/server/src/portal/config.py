"""Portal configuration; no model, database or hardware is opened on import."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[4]


def database_url():
    value = os.getenv("SUPABASE_DB_URL", "").strip()
    placeholders = (
        "your-",
        "your_",
        "placeholder",
        "[password]",
        "[your",
        "example.com",
        "<",
        ">",
    )
    if not value or any(marker in value.lower() for marker in placeholders):
        return ""
    parsed = urlsplit(value)
    if parsed.scheme not in ("postgres", "postgresql") or not parsed.hostname:
        raise ValueError("SUPABASE_DB_URL must be a PostgreSQL connection URL")
    return value


def models_dir():
    return Path(
        os.getenv(
            "CAMS_MODELS_DIR",
            str(Path(__file__).resolve().parents[4] / ".runtime" / "models"),
        )
    )


@dataclass(frozen=True)
class Config:
    database_url: str = field(default_factory=database_url)
    sqlite_path: str = field(
        default_factory=lambda: os.getenv(
            "CAMS_SQLITE_PATH", str(REPO_ROOT / ".runtime" / "portal.sqlite3")
        )
    )
    session_seconds: int = 86400
    max_image_bytes: int = 2_000_000
    max_pixels: int = 4_000_000
    max_body_bytes: int = 2_800_000
    monitor_ttl: int = 300
    max_sessions: int = 8
    max_sessions_per_owner: int = 2
    max_tracks: int = 128
    max_persons: int = 1000
    track_ttl: float = 3.0
    frame_interval: float = 0.15
    face_threshold: float = field(
        default_factory=lambda: float(os.getenv("THRESHOLD_DISTANCE", "0.52"))
    )
    recognition_interval: float = field(
        default_factory=lambda: float(os.getenv("CAMS_RECOGNITION_INTERVAL", "1.0"))
    )
    # Coordinates match the mirrored browser video. Moving left to right
    # is entry; the reverse is exit. A narrow neutral band filters jitter.
    line_fraction: float = field(
        default_factory=lambda: float(os.getenv("CAMS_LINE_FRACTION", "0.25"))
    )
    dead_zone_fraction: float = field(
        default_factory=lambda: float(
            os.getenv("CAMS_LINE_DEAD_ZONE_FRACTION", "0.015")
        )
    )
    # Retained as a configuration compatibility field for existing deployments.
    # The imported desktop trajectory rule confirms a crossing on the first
    # stable-side transition instead.
    confirmation_frames: int = 1
    models_dir: Path = field(default_factory=models_dir)
    serial_port: str = field(default_factory=lambda: os.getenv("SERIAL_PORT", ""))
    serial_baud: int = field(
        default_factory=lambda: int(
            os.getenv("CAMS_SERIAL_BAUD", os.getenv("SERIAL_BAUD", "9600"))
        )
    )
    mqtt_host: str = field(
        default_factory=lambda: os.getenv(
            "CAMS_MQTT_HOST", os.getenv("MQTT_BROKER_ADDRESS", "127.0.0.1")
        )
    )
    mqtt_port: int = field(
        default_factory=lambda: int(
            os.getenv("CAMS_MQTT_PORT", os.getenv("MQTT_BROKER_PORT", "1883"))
        )
    )
    mqtt_username: str = field(
        default_factory=lambda: os.getenv("CAMS_MQTT_USERNAME", "")
    )
    mqtt_password: str = field(
        default_factory=lambda: os.getenv("CAMS_MQTT_PASSWORD", "")
    )
    mqtt_tls: bool = field(
        default_factory=lambda: os.getenv("CAMS_MQTT_TLS", "0") == "1"
    )

    def __post_init__(self):
        object.__setattr__(self, "models_dir", Path(self.models_dir))
        if not 0 < self.face_threshold <= 1:
            raise ValueError(
                "THRESHOLD_DISTANCE must be greater than zero and at most one"
            )
        if self.recognition_interval <= 0:
            raise ValueError("CAMS_RECOGNITION_INTERVAL must be positive")
        if (
            not 0 < self.line_fraction < 1
            or not 0 < self.dead_zone_fraction < 0.5
            or self.confirmation_frames < 1
        ):
            raise ValueError("Invalid crossing confirmation configuration")
