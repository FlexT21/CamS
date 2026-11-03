import base64
import io
import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from .security import token_hash
from .service import get_portal
from .vision import decode_image, tracker_status

logger = logging.getLogger(__name__)


class SafeRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def safe_handler(request):
            try:
                # Read incrementally before JSON parsing, including chunked bodies.
                # All CPU/inference work remains in synchronous route functions.
                portal = await run_in_threadpool(get_portal)
                maximum = portal.config.max_body_bytes
                length = request.headers.get("content-length")
                if length is not None:
                    try:
                        if int(length) < 0:
                            raise ValueError
                        if int(length) > maximum:
                            raise HTTPException(413, "Request body exceeds limit")
                    except ValueError:
                        raise HTTPException(400, "Invalid Content-Length") from None
                body = bytearray()
                async for chunk in request.stream():
                    if len(body) + len(chunk) > maximum:
                        raise HTTPException(413, "Request body exceeds limit")
                    body.extend(chunk)
                request._body = bytes(body)
                response = await handler(request)
            except RequestValidationError as exc:
                # Pydantic's default errors echo inputs, including passwords/images.
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": [
                            {
                                "loc": list(error["loc"]),
                                "type": error["type"],
                                "msg": error["msg"],
                            }
                            for error in exc.errors()
                        ]
                    },
                    headers={"Cache-Control": "no-store"},
                )
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail},
                    headers={"Cache-Control": "no-store", **(exc.headers or {})},
                )
            except Exception as exc:
                # Don't leak SQL, database credentials, hardware paths or biometric data.
                logger.error("Portal request failed (%s)", type(exc).__name__)
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Portal service temporarily unavailable"},
                    headers={"Cache-Control": "no-store"},
                )
            response.headers["Cache-Control"] = "no-store"
            return response

        return safe_handler


router = APIRouter(prefix="/portal", tags=["Portal"], route_class=SafeRoute)
bearer = HTTPBearer(auto_error=False)


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Login(Input):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def email_format(cls, value):
        value = value.strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("Invalid email address")
        return value


class Register(Login):
    password: str = Field(min_length=10, max_length=128)
    name: str = Field(min_length=1, max_length=120)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        if not value.strip():
            raise ValueError("Name cannot be blank")
        return value.strip()


class Person(Input):
    name: str = Field(min_length=1, max_length=120)
    external_id: str = Field(min_length=1, max_length=120)
    image: str = Field(min_length=1, max_length=2_700_000)

    @field_validator("name", "external_id")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Value cannot be blank")
        return value.strip()


class Frame(Input):
    image: str = Field(min_length=1, max_length=2_700_000)
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


def authenticated(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or len(credentials.credentials) > 128
    ):
        raise HTTPException(
            401, "Bearer session required", headers={"WWW-Authenticate": "Bearer"}
        )
    portal = get_portal()
    user = portal.authenticate(credentials.credentials)
    portal.limiter.check(("api", user["id"]), 600, 60)
    return user


def auth_rate(request, email):
    portal = get_portal()
    # Do not trust X-Forwarded-For. Configure trusted proxy IP forwarding in Uvicorn.
    host = request.client.host if request.client else "unknown"
    portal.limiter.check(("auth-ip", host), 20, 60)
    portal.limiter.check(("auth-email", email), 10, 60)


def thumbnail(data):
    from PIL import Image

    with Image.open(io.BytesIO(bytes(data))) as image:
        image.thumbnail((160, 160))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


@router.post("/auth/register", status_code=201)
def register(payload: Register, request: Request):
    auth_rate(request, payload.email)
    return get_portal().register(payload.email, payload.password, payload.name)


@router.post("/auth/login")
def login(payload: Login, request: Request):
    auth_rate(request, payload.email)
    return get_portal().login(payload.email, payload.password)


@router.post("/auth/logout")
def logout(user=Depends(authenticated), credentials=Depends(bearer)):
    portal = get_portal()
    with portal.db.connect(write=True) as conn:
        portal.db.execute(
            conn,
            "DELETE FROM cams_sessions WHERE token_hash=? AND owner_id=?",
            (token_hash(credentials.credentials), user["id"]),
        )
    return {"message": "Logged out"}


@router.get("/me")
def me(user=Depends(authenticated)):
    return user


@router.get("/persons")
def persons(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0, le=100_000),
    user=Depends(authenticated),
):
    portal = get_portal()
    with portal.db.connect() as conn:
        rows = portal.db.execute(
            conn,
            """SELECT id,name,external_id,created_at,image FROM cams_persons
            WHERE owner_id=? ORDER BY created_at,id LIMIT ? OFFSET ?""",
            (user["id"], limit, offset),
        ).fetchall()
        total = portal.db.execute(
            conn,
            "SELECT COUNT(*) AS n FROM cams_persons WHERE owner_id=?",
            (user["id"],),
        ).fetchone()["n"]
    result = []
    for row in rows:
        person = {key: row[key] for key in ("id", "name", "external_id", "created_at")}
        person["image"] = thumbnail(row["image"])
        result.append(person)
    return {"persons": result, "total": total, "limit": limit, "offset": offset}


@router.post("/persons", status_code=201)
def enroll(payload: Person, user=Depends(authenticated)):
    portal = get_portal()
    portal.limiter.check(("enroll", user["id"]), 10, 60)
    person = portal.enroll(user["id"], payload.name, payload.external_id, payload.image)
    # Same image representation as GET /persons; originals stay in database.
    data, _ = decode_image(payload.image, portal.config)
    person["image"] = thumbnail(data)
    return person


@router.get("/events")
def events(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0, le=100_000),
    user=Depends(authenticated),
):
    portal = get_portal()
    with portal.db.connect() as conn:
        rows = portal.db.execute(
            conn,
            """SELECT id,session_id,track_id,direction,person_id,person_name,created_at
            FROM cams_events WHERE owner_id=? ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?""",
            (user["id"], limit, offset),
        ).fetchall()
    return {"events": [dict(row) for row in rows], "limit": limit, "offset": offset}


@router.get("/stats")
def stats(user=Depends(authenticated)):
    portal = get_portal()
    with portal.db.connect() as conn:
        # Single statement gives a consistent snapshot across all counters.
        row = portal.db.execute(
            conn,
            """SELECT
            COALESCE(SUM(CASE WHEN direction='entrada' THEN 1 ELSE 0 END),0) AS entries,
            COALESCE(SUM(CASE WHEN direction='salida' THEN 1 ELSE 0 END),0) AS exits,
            (SELECT COUNT(*) FROM cams_persons WHERE owner_id=?) AS persons
            FROM cams_events WHERE owner_id=?""",
            (user["id"], user["id"]),
        ).fetchone()
    result = {key: int(row[key]) for key in ("entries", "exits", "persons")}
    result["inside"] = max(0, result["entries"] - result["exits"])
    return result


@router.get("/status")
def status(user=Depends(authenticated)):
    portal = get_portal()
    with portal.db.connect() as conn:
        portal.db.execute(conn, "SELECT 1").fetchone()
    return {
        "database": {
            "backend": "postgresql" if portal.db.postgres else "sqlite",
            "mode": "remote" if portal.db.postgres else "local",
            "state": "connected",
        },
        "face": portal.faces.status(),
        "tracker": dict(tracker_status(portal.config), state="ready")
        if portal.monitor.loaded(user["id"])
        else tracker_status(portal.config),
        "monitor_sessions": portal.monitor.count(user["id"]),
        **portal.bridge.status(),
    }


@router.post("/monitor/frame")
def frame(payload: Frame, user=Depends(authenticated)):
    portal = get_portal()
    portal.limiter.check(("frames", user["id"]), 360, 60)
    with portal.monitor.session(user["id"], payload.session_id) as session:
        # Retry an uncommitted batch before consuming another tracker frame.
        # Crossing keys make retries safe if the prior commit outcome was unknown.
        saved = portal.db.save_events(user["id"], session.pending)
        session.pending.clear()
        if saved:
            portal.bridge.publish_saved(tuple(saved))
        _, image = decode_image(payload.image, portal.config)
        tracks = portal.monitor.process(
            session, payload.session_id, image, portal.db.encodings(user["id"])
        )
        for person_name in session.pending_access:
            portal.bridge.publish_access(person_name)
        session.pending_access.clear()
        new_events = portal.db.save_events(user["id"], session.pending)
        session.pending.clear()
        if new_events:
            portal.bridge.publish_saved(tuple(new_events))
        saved.extend(new_events)
    height, width = image.shape[:2]
    return {
        "tracks": tracks,
        "events": saved,
        "message": "Crossing confirmed" if saved else "Frame processed",
        "width": width,
        "height": height,
        "line_x": width * portal.config.line_fraction,
    }
