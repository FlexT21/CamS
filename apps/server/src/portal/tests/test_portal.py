import base64
import io
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from src.portal import router, service
from src.portal.bridge import EventBridge
from src.portal.config import Config
from src.portal.monitor import Monitor, Track
from src.portal.security import RateLimiter, check_password, hash_password, token_hash
from src.portal.vision import (
    FaceEngine,
    MediaPipeTracker,
    cosine,
    decode_image,
    match,
    normalize,
    tracker_status,
)


def jpeg():
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), color="white").save(buffer, "JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeFaces:
    def __init__(self):
        self.lock = threading.Lock()
        self.values = [[1.0, 0.0]]

    def _load(self):
        return None

    def embeddings(self, frame):
        return self.values


class FakeTracker:
    def __init__(self, config):
        self.center = 10
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [(7, self.center - 5, 0, self.center + 5, 90)]


class PortalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = Config(
            database_url="",
            sqlite_path=str(Path(self.temp.name) / "portal.db"),
            models_dir=Path(self.temp.name),
            mqtt_host="",
            serial_port="",
            frame_interval=0,
        )
        self.portal = service.Portal(self.config)
        self.portal.db.initialize()
        self.previous = service._portal
        service._portal = self.portal
        app = FastAPI()
        app.include_router(router, prefix="/api")
        self.client = TestClient(app)
        self.image = jpeg()

    def tearDown(self):
        self.client.close()
        self.portal.bridge.close()
        service._portal = self.previous
        self.temp.cleanup()

    def account(self, email="owner@example.com"):
        response = self.client.post(
            "/api/portal/auth/register",
            json={"email": email, "password": "long-password", "name": "Owner"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def headers(self, account):
        return {"Authorization": "Bearer " + account["access_token"]}

    def enroll(self, account, external_id="p1"):
        self.portal.faces = FakeFaces()
        return self.client.post(
            "/api/portal/persons",
            headers=self.headers(account),
            json={"name": "Person", "external_id": external_id, "image": self.image},
        )

    def test_auth_hashes_expiry_logout_and_no_private_fields(self):
        account = self.account()
        self.assertNotIn("password", json.dumps(account))
        with self.portal.db.connect(write=True) as conn:
            user = self.portal.db.execute(conn, "SELECT * FROM cams_users").fetchone()
            session = self.portal.db.execute(
                conn, "SELECT * FROM cams_sessions"
            ).fetchone()
            self.assertNotEqual(user["password_hash"], "long-password")
            self.assertEqual(session["token_hash"], token_hash(account["access_token"]))
            self.assertNotEqual(session["token_hash"], account["access_token"])
        me = self.client.get("/api/portal/me", headers=self.headers(account))
        self.assertEqual(set(me.json()), {"id", "email", "name"})
        login = self.client.post(
            "/api/portal/auth/login",
            json={"email": "OWNER@example.com", "password": "long-password"},
        )
        self.assertEqual(login.status_code, 200)
        self.assertEqual(
            self.client.post(
                "/api/portal/auth/login",
                json={"email": "owner@example.com", "password": "wrong"},
            ).status_code,
            401,
        )
        self.assertEqual(
            self.client.post(
                "/api/portal/auth/logout", headers=self.headers(account)
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.get(
                "/api/portal/me", headers=self.headers(account)
            ).status_code,
            401,
        )
        with self.portal.db.connect(write=True) as conn:
            self.portal.db.execute(conn, "UPDATE cams_sessions SET expires_at=0")
        self.assertEqual(
            self.client.get(
                "/api/portal/me", headers=self.headers(login.json())
            ).status_code,
            401,
        )

    def test_all_private_endpoints_require_auth(self):
        for path in ("me", "persons", "events", "stats", "status"):
            self.assertEqual(self.client.get("/api/portal/" + path).status_code, 401)
        for path, body in (
            ("persons", {"name": "X", "external_id": "x", "image": self.image}),
            ("monitor/frame", {"image": self.image, "session_id": "browser"}),
        ):
            self.assertEqual(
                self.client.post("/api/portal/" + path, json=body).status_code, 401
            )

    def test_duplicate_account_normalized(self):
        self.account()
        response = self.client.post(
            "/api/portal/auth/register",
            json={
                "email": " OWNER@EXAMPLE.COM ",
                "password": "long-password",
                "name": "Owner",
            },
        )
        self.assertEqual(response.status_code, 409)

    def test_validation_does_not_echo_password_or_image(self):
        response = self.client.post(
            "/api/portal/auth/register",
            json={"email": "bad", "password": "SECRET", "name": ""},
        )
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("SECRET", response.text)
        self.assertNotIn('"input"', response.text)

    def test_tenant_isolation_images_and_duplicates(self):
        a, b = self.account(), self.account("other@example.com")
        response = self.enroll(a)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertTrue(response.json()["image"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(self.enroll(a, "another").status_code, 409)
        self.assertEqual(
            self.enroll(b).status_code, 201
        )  # independent enrollment namespace
        own = self.client.get("/api/portal/persons", headers=self.headers(a)).json()
        other = self.client.get("/api/portal/persons", headers=self.headers(b)).json()
        self.assertEqual(own["total"], 1)
        self.assertNotEqual(own["persons"][0]["id"], other["persons"][0]["id"])
        self.assertNotIn("encoding", json.dumps(own))
        self.assertNotIn("owner_id", json.dumps(own))
        with self.portal.db.connect() as conn:
            row = self.portal.db.execute(
                conn, "SELECT image,encoding FROM cams_persons LIMIT 1"
            ).fetchone()
            self.assertIsInstance(row["image"], bytes)
            self.assertEqual(json.loads(row["encoding"]), [1.0, 0.0])

    def test_enrollment_requires_exactly_one_face(self):
        account = self.account()
        faces = FakeFaces()
        self.portal.faces = faces
        for values in ([], [[1.0, 0.0], [0.0, 1.0]]):
            faces.values = values
            response = self.client.post(
                "/api/portal/persons",
                headers=self.headers(account),
                json={"name": "Person", "external_id": "p1", "image": self.image},
            )
            self.assertEqual(response.status_code, 422)

    def test_missing_existing_engine_explicit_503_and_local_status(self):
        account = self.account()
        unavailable = {
            "backend": "face_recognition_dlib",
            "state": "missing_dependency",
            "missing_models": [],
            "missing_dependencies": ["dlib"],
        }
        with patch.object(self.portal.faces, "status", return_value=unavailable):
            response = self.client.post(
                "/api/portal/persons",
                headers=self.headers(account),
                json={"name": "Person", "external_id": "p1", "image": self.image},
            )
            self.assertEqual(response.status_code, 503)
            self.assertIn("face-recognition/dlib", response.text)
            status = self.client.get(
                "/api/portal/status", headers=self.headers(account)
            ).json()
            self.assertEqual(status["database"]["mode"], "local")
            self.assertEqual(status["face"]["backend"], "face_recognition_dlib")
            self.assertEqual(status["face"]["state"], "missing_dependency")
            self.assertEqual(status["tracker"]["backend"], "mediapipe_face_detection")
            self.assertEqual(status["serial"], "disabled")

    def test_corrupt_jpeg_base64_and_dimensions(self):
        for image in ("!invalid!", base64.b64encode(b"not jpeg").decode()):
            with self.assertRaises(HTTPException) as raised:
                decode_image(image, self.config)
            self.assertEqual(raised.exception.status_code, 422)
        with self.assertRaises(HTTPException) as raised:
            decode_image(self.image, replace(self.config, max_pixels=100))
        self.assertEqual(raised.exception.status_code, 413)

    def test_foreign_engine_encodings_are_never_used_as_dlib(self):
        account = self.account()
        person = self.enroll(account).json()
        self.assertEqual(len(self.portal.db.encodings(account["user"]["id"])), 1)
        with self.portal.db.connect(write=True) as conn:
            self.portal.db.execute(
                conn,
                "UPDATE cams_persons SET encoding_engine='unknown' WHERE id=?",
                (person["id"],),
            )
        self.assertEqual(self.portal.db.encodings(account["user"]["id"]), [])
        self.assertEqual(
            self.client.get(
                "/api/portal/persons", headers=self.headers(account)
            ).json()["total"],
            1,
        )

    def test_body_limit_even_without_content_length(self):
        self.portal.config = replace(self.config, max_body_bytes=30)
        response = self.client.post(
            "/api/portal/auth/login",
            content=iter([b"x" * 20, b"y" * 20]),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 413)

    def test_event_persistence_is_idempotent_and_tenant_filtered(self):
        a, b = self.account(), self.account("other@example.com")
        event = {
            "session_id": "browser",
            "track_id": 7,
            "direction": "entrada",
            "person_id": None,
            "person_name": None,
            "crossing_key": "one-crossing",
        }
        self.assertEqual(len(self.portal.db.save_events(a["user"]["id"], [event])), 1)
        self.assertEqual(self.portal.db.save_events(a["user"]["id"], [event]), [])
        self.assertEqual(
            len(
                self.client.get("/api/portal/events", headers=self.headers(a)).json()[
                    "events"
                ]
            ),
            1,
        )
        self.assertEqual(
            self.client.get("/api/portal/events", headers=self.headers(b)).json()[
                "events"
            ],
            [],
        )
        self.assertEqual(
            self.client.get("/api/portal/stats", headers=self.headers(a)).json(),
            {"entries": 1, "exits": 0, "inside": 1, "persons": 0},
        )
        self.assertEqual(
            self.client.get("/api/portal/stats", headers=self.headers(b)).json()[
                "entries"
            ],
            0,
        )

    def test_frame_crossing_and_output_only_after_commit(self):
        account = self.account()
        self.assertEqual(self.enroll(account).status_code, 201)
        self.portal.monitor = Monitor(self.config, FakeFaces(), FakeTracker)
        with patch.object(self.portal.bridge, "publish_saved") as publish:
            response = None
            observed = []
            for center in (10, 10, 10, 50, 90, 90, 90):
                sessions = self.portal.monitor.sessions
                if sessions:
                    next(iter(sessions.values())).tracker.center = center
                response = self.client.post(
                    "/api/portal/monitor/frame",
                    headers=self.headers(account),
                    json={"session_id": "browser", "image": self.image},
                )
                self.assertEqual(response.status_code, 200, response.text)
                observed.extend(response.json()["events"])
            body = response.json()
            self.assertEqual(observed[0]["direction"], "entrada")
            self.assertEqual(body["tracks"][0]["id"], 7)
            self.assertEqual(
                set(body["tracks"][0]),
                {
                    "id",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "center_x",
                    "user",
                    "distance_to_line",
                    "face",
                },
            )
            self.assertIsNone(body["tracks"][0]["face"])
            self.assertEqual((body["width"], body["height"]), (100, 100))
            self.assertEqual(body["line_x"], 25.0)
            delivered = [
                event for call in publish.call_args_list for event in call.args[0]
            ]
            self.assertEqual(len(delivered), 1)
            self.assertIn("id", delivered[0])
            self.assertEqual(
                self.client.get(
                    "/api/portal/stats", headers=self.headers(account)
                ).json()["entries"],
                1,
            )

    def test_server_errors_do_not_expose_credentials(self):
        account = self.account()
        with patch.object(
            self.portal.db, "connect", side_effect=RuntimeError("postgres://secret")
        ):
            response = self.client.get(
                "/api/portal/stats", headers=self.headers(account)
            )
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret", response.text)

    def test_failed_commit_retains_crossing_and_does_not_publish(self):
        account = self.account()
        self.portal.monitor = Monitor(self.config, FakeFaces(), FakeTracker)
        with self.portal.monitor.session(account["user"]["id"], "browser") as session:
            session.pending.append(
                {
                    "session_id": "browser",
                    "track_id": 7,
                    "direction": "entrada",
                    "person_id": None,
                    "person_name": None,
                    "crossing_key": "pending",
                }
            )
        with (
            patch.object(self.portal.db, "save_events", side_effect=OSError("offline")),
            patch.object(self.portal.bridge, "publish_saved") as publish,
        ):
            response = self.client.post(
                "/api/portal/monitor/frame",
                headers=self.headers(account),
                json={"session_id": "browser", "image": self.image},
            )
            self.assertEqual(response.status_code, 503)
            publish.assert_not_called()
            self.assertEqual(len(session.pending), 1)
        response = self.client.post(
            "/api/portal/monitor/frame",
            headers=self.headers(account),
            json={"session_id": "browser", "image": self.image},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["events"]), 1)
        self.assertEqual(session.pending, [])

    def test_event_cannot_reference_another_tenants_person(self):
        a, b = self.account(), self.account("other@example.com")
        person = self.enroll(b).json()
        event = {
            "session_id": "browser",
            "track_id": 7,
            "direction": "entrada",
            "person_id": person["id"],
            "person_name": person["name"],
            "crossing_key": "cross-tenant",
        }
        with self.assertRaises(ValueError):
            self.portal.db.save_events(a["user"]["id"], [event])
        self.assertEqual(
            self.client.get("/api/portal/events", headers=self.headers(a)).json()[
                "events"
            ],
            [],
        )

    def test_concurrent_duplicate_enrollment_is_serialized(self):
        from concurrent.futures import ThreadPoolExecutor

        account = self.account()
        self.portal.faces = FakeFaces()

        def enroll(index):
            try:
                self.portal.enroll(
                    account["user"]["id"], "Person", str(index), self.image
                )
                return 201
            except HTTPException as exc:
                return exc.status_code

        with ThreadPoolExecutor(max_workers=2) as executor:
            self.assertEqual(sorted(executor.map(enroll, range(2))), [201, 409])

    def test_pagination_and_additional_payload_fields_are_rejected(self):
        account = self.account()
        self.assertEqual(
            self.client.get(
                "/api/portal/persons?limit=99999", headers=self.headers(account)
            ).status_code,
            422,
        )
        response = self.client.post(
            "/api/portal/monitor/frame",
            headers=self.headers(account),
            json={"session_id": "browser", "image": self.image, "direction": "entrada"},
        )
        self.assertEqual(response.status_code, 422)

    def test_wal_and_foreign_keys(self):
        with self.portal.db.connect() as conn:
            self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_return_crossing_is_saved_and_updates_occupancy(self):
        account = self.account()
        self.assertEqual(self.enroll(account).status_code, 201)
        self.portal.monitor = Monitor(self.config, FakeFaces(), FakeTracker)
        directions = []
        with patch.object(self.portal.bridge, "publish_saved") as publish:
            for center in (10, 50, 50, 10, 10):
                sessions = self.portal.monitor.sessions
                if sessions:
                    next(iter(sessions.values())).tracker.center = center
                response = self.client.post(
                    "/api/portal/monitor/frame",
                    headers=self.headers(account),
                    json={"session_id": "browser", "image": self.image},
                )
                self.assertEqual(response.status_code, 200, response.text)
                directions.extend(e["direction"] for e in response.json()["events"])
            stats = self.client.get(
                "/api/portal/stats", headers=self.headers(account)
            ).json()
            self.assertEqual(directions, ["entrada", "salida"])
            self.assertEqual(
                (stats["entries"], stats["exits"], stats["inside"]), (1, 1, 0)
            )
            self.assertEqual(
                sum(len(call.args[0]) for call in publish.call_args_list), 2
            )


class PureTests(unittest.TestCase):
    def test_existing_identity_confirms_once_and_survives_track(self):
        config = Config(database_url="", mqtt_host="", frame_interval=0)
        monitor = Monitor(config, FakeFaces(), FakeTracker)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        persons = [{"id": "p1", "name": "Person", "encoding": [1.0, 0.0]}]
        with monitor.session("a", "browser") as session:
            first = monitor.process(session, "browser", image, persons)
            self.assertEqual(first[0]["user"], "Person")
            session.tracks[7].recognition_at = -1
            second = monitor.process(session, "browser", image, persons)
            self.assertEqual(second[0]["user"], "Person")
            self.assertEqual(session.tracker.calls, 2)
            # A track that expired must not inherit a prior side or identity.
            session.tracks[7].seen = time.monotonic() - config.track_ttl - 1
            session.tracker.center = 90
            third = monitor.process(session, "browser", image, persons)
            self.assertEqual(third[0]["user"], "Person")
            self.assertEqual(session.pending, [])

    def test_same_tracked_person_enters_exits_and_reenters(self):
        config = Config(database_url="", mqtt_host="", frame_interval=0)
        monitor = Monitor(config, FakeFaces(), FakeTracker)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        persons = [{"id": "p1", "name": "Person", "encoding": [1.0, 0.0]}]
        with monitor.session("owner", "browser") as session:
            monitor.process(session, "browser", image, persons)
            session.tracks[7].recognition_at = -1
            monitor.process(session, "browser", image, persons)
            identity = session.tracks[7]
            for center in (35, 50, 50, 35, 10, 10, 35, 50):
                session.tracker.center = center
                output = monitor.process(session, "browser", image, persons)
                self.assertEqual(output[0]["user"], "Person")
                self.assertIs(session.tracks[7], identity)
            self.assertEqual(
                [e["direction"] for e in session.pending],
                ["entrada", "salida", "entrada"],
            )
            self.assertTrue(all(e["person_id"] == "p1" for e in session.pending))
            self.assertEqual(len({e["crossing_key"] for e in session.pending}), 3)

    def test_neutral_band_and_stationary_person_do_not_duplicate_crossings(self):
        track = Track()
        for x in (10, 33.5, 35, 36.5, 35):
            self.assertIsNone(track.crossing(x, 35, 2))
        self.assertEqual(track.crossing(38, 35, 2), "entrada")
        for x in (39, 45, 36, 34, 35):
            self.assertIsNone(track.crossing(x, 35, 2))
        self.assertEqual(track.crossing(32, 35, 2), "salida")
        self.assertEqual(track.crossing(38, 35, 2), "entrada")

    def test_crossing_waits_for_existing_recognizer_before_counting(self):
        config = Config(database_url="", mqtt_host="", frame_interval=0)
        faces = FakeFaces()
        faces.values = []
        monitor = Monitor(config, faces, FakeTracker)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        persons = [{"id": "p1", "name": "Person", "encoding": [1.0, 0.0]}]
        with monitor.session("a", "browser") as session:
            monitor.process(session, "browser", image, persons)
            session.tracker.center = 60
            monitor.process(session, "browser", image, persons)
            self.assertEqual(session.pending, [])
            self.assertEqual(len(session.tracks[7].pending_crossings), 1)
            faces.values = [[1.0, 0.0]]
            session.tracks[7].recognition_at = -1
            monitor.process(session, "browser", image, persons)
            self.assertEqual([e["direction"] for e in session.pending], ["entrada"])
            self.assertEqual(session.pending[0]["person_id"], "p1")

    def test_frame_interval_and_dimension_changes(self):
        config = Config(database_url="", mqtt_host="", frame_interval=5)
        monitor = Monitor(config, FakeFaces(), FakeTracker)
        with monitor.session("a", "browser") as session:
            monitor.process(
                session, "browser", np.zeros((100, 100, 3), dtype=np.uint8), []
            )
            with self.assertRaises(HTTPException) as raised:
                monitor.process(
                    session, "browser", np.zeros((200, 200, 3), dtype=np.uint8), []
                )
            self.assertEqual(raised.exception.status_code, 409)
        with self.assertRaises(HTTPException) as raised:
            with monitor.session("a", "browser"):
                pass
        self.assertEqual(raised.exception.status_code, 429)

    def test_jsonb_encoding_already_decoded(self):
        from src.portal.database import decode_encoding

        self.assertEqual(decode_encoding([1.0, 0.0]), [1.0, 0.0])
        self.assertEqual(decode_encoding("[1.0,0.0]"), [1.0, 0.0])

    def test_scrypt(self):
        stored = hash_password("long-password")
        self.assertTrue(check_password("long-password", stored))
        self.assertFalse(check_password("wrong", stored))
        self.assertFalse(check_password("wrong", "malformed"))

    def test_existing_euclidean_threshold(self):
        vector = normalize([3, 4])
        self.assertAlmostEqual(cosine(vector, vector), 1)
        person = {"id": "a", "name": "A", "encoding": vector}
        self.assertEqual(match(vector, [person], 0.45)["id"], "a")
        self.assertEqual(match(vector, [person, dict(person, id="b")], 0.45)["id"], "a")
        self.assertIsNone(match([-1, 0], [person], 0.45))

    def test_desktop_trajectory_uses_left_line_centroid_and_one_crossing(self):
        track = Track()
        # Width 100, line at 35 and an 8-pixel neutral zone, exactly as the
        # browser portal now adapts the desktop direction tracker.
        self.assertIsNone(track.crossing(10, 35, 8))
        self.assertIsNone(track.crossing(35, 35, 8))  # neutral area
        self.assertEqual(track.crossing(80, 35, 8), "entrada")
        # A different person first seen on the right must start without an
        # event, then produce an exit when moving to the left.
        fresh = Track()
        self.assertIsNone(fresh.crossing(80, 35, 8))
        self.assertEqual(fresh.crossing(10, 35, 8), "salida")

    def test_sessions_owner_scope_busy_limit_and_ttl(self):
        config = Config(
            database_url="",
            mqtt_host="",
            max_sessions=2,
            max_sessions_per_owner=1,
            frame_interval=0,
        )
        monitor = Monitor(config, FakeFaces(), FakeTracker)
        with monitor.session("a", "same-id") as a:
            with self.assertRaises(HTTPException) as raised:
                with monitor.session("a", "same-id"):
                    pass
            self.assertEqual(raised.exception.status_code, 409)
            with monitor.session("b", "same-id") as b:
                self.assertIsNot(a, b)
        with self.assertRaises(HTTPException) as raised:
            with monitor.session("a", "new-id"):
                pass
        self.assertEqual(raised.exception.status_code, 429)
        a.touched = time.monotonic() - config.monitor_ttl - 1
        with monitor.session("a", "new-id") as new:
            self.assertIsNot(new, a)
        self.assertEqual(len(monitor.sessions), 2)

    def test_rate_limits(self):
        limiter = RateLimiter(capacity=1)
        limiter.check("a", 1, 60)
        for key in ("a", "b"):
            with self.assertRaises(HTTPException) as raised:
                limiter.check(key, 1, 60)
            self.assertEqual(raised.exception.status_code, 429)

    def test_configuration_aliases_and_placeholder_database(self):
        with patch.dict(
            "os.environ",
            {
                "SUPABASE_DB_URL": "postgresql://postgres:[YOUR-PASSWORD]@db.example.com/postgres",
                "MQTT_BROKER_ADDRESS": "test-broker",
                "MQTT_BROKER_PORT": "1884",
                "SERIAL_BAUD": "115200",
            },
            clear=True,
        ):
            config = Config()
            self.assertEqual(config.database_url, "")
            self.assertEqual(config.mqtt_host, "test-broker")
            self.assertEqual(config.mqtt_port, 1884)
            self.assertEqual(config.serial_baud, 115200)
            self.assertEqual(Path(config.sqlite_path).name, "portal.sqlite3")

    def test_pg_migration_security_and_pooler_contract(self):
        sql = (
            Path(service.__file__)
            .with_name("migrations")
            .joinpath("postgres.sql")
            .read_text()
        )
        for table in ("cams_users", "cams_sessions", "cams_persons", "cams_events"):
            self.assertIn(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY", sql)
        for role in ("PUBLIC", "anon", "authenticated"):
            self.assertIn(f"FROM {role}", sql)
        self.assertIn("image BYTEA", sql)
        from unittest.mock import MagicMock

        from src.portal.database import Database

        fake_conn = MagicMock()
        with patch("psycopg.connect", return_value=fake_conn) as connect:
            db = Database(Config(database_url="postgresql://backend@localhost/test"))
            with db.connect():
                pass
            self.assertIsNone(connect.call_args.kwargs["prepare_threshold"])
            fake_conn.commit.assert_called_once()
            fake_conn.close.assert_called_once()

    def test_hardware_failures_are_nonfatal_and_status_disconnected(self):
        bridge = EventBridge(
            Config(database_url="", serial_port="INVALID", mqtt_host="127.0.0.1")
        )
        event = {"id": "event", "direction": "entrada"}
        with patch.dict("sys.modules", {"serial": None, "paho.mqtt.client": None}):
            bridge._serial_event(event)
            bridge._mqtt_event(event)
        self.assertEqual(bridge.status()["serial"], "disconnected")
        self.assertEqual(bridge.status()["mqtt"], "disconnected")
        bridge.close()

    def test_bridge_publication_does_not_start_worker(self):
        bridge = EventBridge(
            Config(database_url="", serial_port="test", mqtt_host="test")
        )
        with patch("src.portal.bridge.threading.Thread") as worker:
            bridge.publish_saved([{"id": "committed", "direction": "entrada"}])
            worker.assert_not_called()
        self.assertIsNone(bridge.thread)
        self.assertEqual(bridge.queue.qsize(), 1)
        bridge.close()

    def test_bridge_start_connects_without_publication_and_can_restart(self):
        from unittest.mock import MagicMock

        bridge = EventBridge(
            Config(database_url="", serial_port="test", mqtt_host="test")
        )
        serial_opened, mqtt_opened = threading.Event(), threading.Event()
        device, client = MagicMock(), MagicMock()
        device.is_open = True
        client.is_connected.return_value = False

        def open_serial(*args, **kwargs):
            serial_opened.set()
            return device

        def loop_start():
            client.is_connected.return_value = True
            client.on_connect(client, None, {}, 0, None)
            mqtt_opened.set()
            return 0

        client.loop_start.side_effect = loop_start
        with (
            patch("serial.Serial", side_effect=open_serial) as serial_factory,
            patch("paho.mqtt.client.Client", return_value=client) as mqtt_factory,
        ):
            try:
                self.assertTrue(bridge.start())
                self.assertTrue(mqtt_opened.wait(2))
                self.assertTrue(serial_opened.wait(2))
                worker = bridge.thread
                self.assertTrue(bridge.start())
                self.assertIs(bridge.thread, worker)
                self.assertEqual(bridge.status()["mqtt"], "connected")
                self.assertEqual(bridge.status()["serial"], "unverified")
                client.publish.assert_not_called()
                device.write.assert_not_called()
                serial_factory.assert_called_once()
                mqtt_factory.assert_called_once()
                client.connect_async.assert_called_once_with(
                    "test", bridge.config.mqtt_port, keepalive=30
                )
                client.reconnect_delay_set.assert_called_once_with(
                    min_delay=2, max_delay=5
                )
                client.on_disconnect(client, None, {}, 1, None)
                client.is_connected.return_value = False
                self.assertEqual(bridge.status()["mqtt"], "disconnected")
                client.on_connect(client, None, {}, 5, None)
                self.assertEqual(bridge.status()["mqtt"], "disconnected")
                bridge.close()
                self.assertFalse(worker.is_alive())
                serial_opened.clear()
                mqtt_opened.clear()
                self.assertTrue(bridge.start())
                self.assertTrue(mqtt_opened.wait(2))
                self.assertTrue(serial_opened.wait(2))
                self.assertIsNot(bridge.thread, worker)
                self.assertEqual(serial_factory.call_count, 2)
                self.assertEqual(mqtt_factory.call_count, 2)
                client.publish.assert_not_called()
                device.write.assert_not_called()
            finally:
                bridge.close()

    def test_serial_boot_wait_precedes_first_wire_event(self):
        from unittest.mock import MagicMock

        bridge = EventBridge(Config(database_url="", serial_port="test", mqtt_host=""))
        device = MagicMock()
        device.is_open = True
        device.write.return_value = 2
        device.readline.return_value = b"ACK 0\r\n"
        with (
            patch("serial.Serial", return_value=device),
            patch("src.portal.bridge.time.monotonic", return_value=100),
            patch.object(bridge.stopping, "wait", return_value=False) as wait,
        ):
            self.assertTrue(bridge._ensure_serial())
            device.write.assert_not_called()
            bridge._serial_event({"direction": "entrada"})
            wait.assert_called_once_with(2)
            device.write.assert_called_once_with(b"0\n")
        bridge.close()

    def test_connection_failures_throttle_retry(self):
        bridge = EventBridge(
            Config(database_url="", serial_port="test", mqtt_host="test")
        )
        with (
            patch(
                "serial.Serial", side_effect=OSError("unavailable")
            ) as serial_factory,
            patch(
                "paho.mqtt.client.Client", side_effect=OSError("unavailable")
            ) as mqtt_factory,
            patch("src.portal.bridge.time.monotonic", return_value=100) as now,
        ):
            for _ in range(2):
                bridge._ensure_serial()
                bridge._ensure_mqtt()
            self.assertEqual(serial_factory.call_count, 1)
            self.assertEqual(mqtt_factory.call_count, 1)
            now.return_value = 102
            bridge._ensure_serial()
            bridge._ensure_mqtt()
            self.assertEqual(serial_factory.call_count, 2)
            self.assertEqual(mqtt_factory.call_count, 1)
            now.return_value = 105
            bridge._ensure_mqtt()
            self.assertEqual(mqtt_factory.call_count, 2)
            self.assertEqual(bridge.status()["serial"], "disconnected")
            self.assertEqual(bridge.status()["mqtt"], "disconnected")
        bridge.close()

    def test_serial_wire_protocol(self):
        from unittest.mock import MagicMock

        bridge = EventBridge(Config(database_url="", serial_port="test", mqtt_host=""))
        bridge.serial = MagicMock()
        bridge.serial.write.return_value = 2
        bridge.serial.readline.side_effect = [b"ACK 0\r\n", b"ACK 1\r\n"]
        bridge._serial_event({"direction": "entrada"})
        bridge._serial_event({"direction": "salida"})
        self.assertEqual(bridge.status()["serial"], "connected")
        self.assertEqual(
            [call.args[0] for call in bridge.serial.write.call_args_list],
            [b"0\n", b"1\n"],
        )
        bridge.close()

    def test_serial_missing_ack_is_not_reported_as_delivered(self):
        from unittest.mock import MagicMock

        bridge = EventBridge(Config(database_url="", serial_port="test", mqtt_host=""))
        device = MagicMock()
        device.write.return_value = 2
        device.readline.return_value = b""
        bridge.serial = device
        bridge._serial_event({"direction": "entrada"})
        self.assertIsNone(bridge.serial)
        self.assertEqual(bridge.status()["serial"], "disconnected")
        device.close.assert_called_once()
        bridge.close()

    def test_mediapipe_centroids_match_mirrored_video_and_preserve_track(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        config = Config(database_url="", mqtt_host="")
        detector = MagicMock()
        box = SimpleNamespace(xmin=0.7, ymin=0.2, width=0.2, height=0.4)
        face = SimpleNamespace(location_data=SimpleNamespace(relative_bounding_box=box))
        detector.process.return_value = SimpleNamespace(detections=[face])
        with patch(
            "mediapipe.python.solutions.face_detection.FaceDetection",
            return_value=detector,
        ):
            tracker = MediaPipeTracker(config)
        try:
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            first = tracker.detect(image)[0]
            box.xmin = 0.1
            second = tracker.detect(image)[0]
            self.assertEqual(first[0], second[0])
            self.assertAlmostEqual((first[1] + first[3]) / 2, 20, delta=1)
            self.assertAlmostEqual((second[1] + second[3]) / 2, 80, delta=1)
            detector.process.return_value = SimpleNamespace(detections=[face, face])
            self.assertEqual(tracker.detect(image), [])
            detector.process.return_value = SimpleNamespace(detections=[face])
            self.assertNotEqual(tracker.detect(image)[0][0], first[0])
        finally:
            tracker.close()
        detector.close.assert_called_once()

    def test_tracker_status_checks_dependencies(self):
        with patch("src.portal.vision.importlib.util.find_spec", return_value=None):
            status = tracker_status(Config(database_url=""))
        self.assertEqual(status["state"], "missing_dependency")
        self.assertEqual(status["missing_dependencies"], ["mediapipe", "cv2"])
        self.assertEqual(status["backend"], "mediapipe_face_detection")


class LocalModelTests(unittest.TestCase):
    def test_existing_dlib_models_load_and_blank_has_no_faces(self):
        config = Config(database_url="", mqtt_host="")
        faces = FaceEngine(config)
        if faces.status()["missing_models"]:
            self.skipTest("Optional local OpenCV model weights absent")
        self.assertEqual(faces.embeddings(np.zeros((320, 320, 3), dtype=np.uint8)), [])
        self.assertEqual(faces.status()["state"], "ready")


if __name__ == "__main__":
    unittest.main()
