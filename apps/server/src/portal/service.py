import json
import secrets
import threading
import time
import uuid

from fastapi import HTTPException

from .bridge import EventBridge
from .config import Config
from .database import Database, decode_encoding
from .monitor import Monitor
from .security import RateLimiter, check_password, hash_password, token_hash
from .vision import FaceEngine, decode_image, distance


class Portal:
    def __init__(self, config=None):
        self.config = config or Config()
        self.db = Database(self.config)
        self.faces = FaceEngine(self.config)
        self.monitor = Monitor(self.config, self.faces)
        self.bridge = EventBridge(self.config)
        self.limiter = RateLimiter()
        self.dummy_hash = hash_password(secrets.token_urlsafe(32))

    def issue_token(self, conn, owner):
        self.db.owner_lock(conn, owner)
        now = time.time()
        self.db.execute(conn, "DELETE FROM cams_sessions WHERE expires_at<=?", (now,))
        count = self.db.execute(
            conn, "SELECT COUNT(*) AS n FROM cams_sessions WHERE owner_id=?", (owner,)
        ).fetchone()["n"]
        if count >= 16:
            raise HTTPException(
                429, "Active login session limit reached; log out an existing session"
            )
        token = secrets.token_urlsafe(32)
        expires = now + self.config.session_seconds
        self.db.execute(
            conn,
            "INSERT INTO cams_sessions(token_hash,owner_id,expires_at) VALUES (?,?,?)",
            (token_hash(token), owner, expires),
        )
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_at": expires,
            "expires_in": self.config.session_seconds,
        }

    def register(self, email, password, name):
        password_hash = hash_password(password)
        owner = str(uuid.uuid4())
        try:
            with self.db.connect(write=True) as conn:
                self.db.execute(
                    conn,
                    "INSERT INTO cams_users(id,email,name,password_hash,created_at) VALUES (?,?,?,?,?)",
                    (owner, email, name, password_hash, time.time()),
                )
                token = self.issue_token(conn, owner)
        except Exception as exc:
            if self.db.is_integrity_error(exc):
                raise HTTPException(409, "Account already exists") from None
            raise
        return dict(token, user={"id": owner, "email": email, "name": name})

    def login(self, email, password):
        with self.db.connect() as conn:
            row = self.db.execute(
                conn,
                "SELECT id,email,name,password_hash FROM cams_users WHERE email=?",
                (email,),
            ).fetchone()
        if (
            not check_password(
                password, row["password_hash"] if row else self.dummy_hash
            )
            or row is None
        ):
            raise HTTPException(
                401, "Invalid email or password", headers={"WWW-Authenticate": "Bearer"}
            )
        with self.db.connect(write=True) as conn:
            token = self.issue_token(conn, row["id"])
        return dict(token, user={key: row[key] for key in ("id", "email", "name")})

    def authenticate(self, token):
        with self.db.connect() as conn:
            row = self.db.execute(
                conn,
                """SELECT u.id,u.email,u.name FROM cams_sessions s
                JOIN cams_users u ON s.owner_id=u.id WHERE s.token_hash=? AND s.expires_at>?""",
                (token_hash(token), time.time()),
            ).fetchone()
        if row is None:
            raise HTTPException(
                401,
                "Invalid or expired session",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return dict(row)

    def enroll(self, owner, name, external_id, image):
        data, frame = decode_image(image, self.config)
        encodings = self.faces.embeddings(frame)
        if len(encodings) != 1:
            raise HTTPException(422, "Enrollment requires exactly one face")
        encoding = encodings[0]
        person = {
            "id": str(uuid.uuid4()),
            "name": name,
            "external_id": external_id,
            "created_at": time.time(),
        }
        try:
            with self.db.connect(write=True) as conn:
                self.db.owner_lock(conn, owner)
                persons = self.db.persons(owner, conn, private=True)
                if len(persons) >= self.config.max_persons:
                    raise HTTPException(409, "Enrollment limit reached")
                if any(
                    p["encoding_engine"] == "face_recognition_dlib"
                    and distance(encoding, decode_encoding(p["encoding"]))
                    <= self.config.face_threshold
                    for p in persons
                ):
                    raise HTTPException(409, "Face is already enrolled")
                self.db.execute(
                    conn,
                    """INSERT INTO cams_persons(id,owner_id,name,external_id,image,encoding,encoding_engine,created_at)
                    VALUES (?,?,?,?,?,?,'face_recognition_dlib',?)""",
                    (
                        person["id"],
                        owner,
                        name,
                        external_id,
                        data,
                        json.dumps(encoding, allow_nan=False),
                        person["created_at"],
                    ),
                )
        except Exception as exc:
            if self.db.is_integrity_error(exc):
                raise HTTPException(409, "External ID is already enrolled") from None
            raise
        return person


_portal = None
_initialize_lock = threading.Lock()


def initialize(config=None):
    """Initialize storage once. Call after loading the parent application's env."""
    global _portal
    with _initialize_lock:
        if _portal is None:
            portal = Portal(config)
            portal.db.initialize()
            _portal = portal
        return _portal


def get_portal():
    return initialize()
