import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS cams_users (
 id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
 password_hash TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS cams_sessions (
 token_hash TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES cams_users(id) ON DELETE CASCADE,
 expires_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS cams_sessions_owner ON cams_sessions(owner_id);
CREATE TABLE IF NOT EXISTS cams_persons (
 id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES cams_users(id) ON DELETE CASCADE,
 name TEXT NOT NULL, external_id TEXT NOT NULL, image BLOB NOT NULL,
 encoding TEXT NOT NULL, encoding_engine TEXT NOT NULL DEFAULT 'unknown', created_at REAL NOT NULL, UNIQUE(owner_id,external_id));
CREATE TABLE IF NOT EXISTS cams_events (
 id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES cams_users(id) ON DELETE CASCADE,
 session_id TEXT NOT NULL, track_id INTEGER NOT NULL,
 direction TEXT NOT NULL CHECK(direction IN ('entrada','salida')),
 person_id TEXT REFERENCES cams_persons(id), person_name TEXT,
 created_at REAL NOT NULL, crossing_key TEXT NOT NULL, UNIQUE(owner_id,crossing_key));
CREATE INDEX IF NOT EXISTS cams_events_owner_time ON cams_events(owner_id,created_at);
CREATE INDEX IF NOT EXISTS cams_events_person ON cams_events(person_id);
CREATE INDEX IF NOT EXISTS cams_sessions_expiry ON cams_sessions(expires_at);
"""


class Database:
    def __init__(self, config):
        self.config = config
        self.postgres = bool(config.database_url)

    @contextmanager
    def connect(self, write=False):
        if self.postgres:
            import psycopg
            from psycopg.rows import dict_row

            conn = psycopg.connect(
                self.config.database_url,
                prepare_threshold=None,
                row_factory=dict_row,
                connect_timeout=5,
            )
        else:
            conn = sqlite3.connect(self.config.sqlite_path, timeout=5)
            conn.row_factory = sqlite3.Row
        try:
            if self.postgres:
                # Transaction-local settings survive transaction pooling without
                # requiring support for libpq's startup `options` parameter.
                conn.execute("""SELECT set_config('search_path','public',true),
                    set_config('statement_timeout','10000',true),
                    set_config('lock_timeout','5000',true)""")
            else:
                conn.execute("PRAGMA foreign_keys=ON")
                if write:
                    conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, conn, sql, params=()):
        return conn.execute(sql.replace("?", "%s") if self.postgres else sql, params)

    def initialize(self):
        if not self.postgres:
            Path(self.config.sqlite_path).resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        with self.connect() as conn:
            if self.postgres:
                # Serialize startup migrations across workers using a transaction lock.
                conn.execute("SELECT pg_advisory_xact_lock(714209381)")
                conn.execute(
                    Path(__file__)
                    .with_name("migrations")
                    .joinpath("postgres.sql")
                    .read_text(encoding="utf-8"),
                    prepare=False,
                )
            else:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(SQLITE_SCHEMA)
                columns = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(cams_persons)")
                }
                if "encoding_engine" not in columns:
                    conn.execute(
                        "ALTER TABLE cams_persons ADD COLUMN encoding_engine TEXT NOT NULL DEFAULT 'unknown'"
                    )

    def is_integrity_error(self, exc):
        if isinstance(exc, sqlite3.IntegrityError):
            return True
        if self.postgres:
            import psycopg

            return isinstance(exc, psycopg.IntegrityError)
        return False

    def owner_lock(self, conn, owner):
        if self.postgres:
            self.execute(
                conn, "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))", (owner,)
            )

    def persons(self, owner, conn=None, private=False):
        columns = "id,name,external_id,created_at" + (
            ",encoding,encoding_engine" if private else ""
        )
        if conn is None:
            with self.connect() as connection:
                return self.persons(owner, connection, private)
        rows = self.execute(
            conn,
            f"SELECT {columns} FROM cams_persons WHERE owner_id=? ORDER BY created_at,id",
            (owner,),
        ).fetchall()
        return [dict(row) for row in rows]

    def save_events(self, owner, events):
        if not events:
            return []
        saved = []
        with self.connect(write=True) as conn:
            for event in events:
                row = dict(event, id=str(uuid.uuid4()), created_at=time.time())
                if (
                    row["person_id"]
                    and self.execute(
                        conn,
                        "SELECT id FROM cams_persons WHERE id=? AND owner_id=?",
                        (row["person_id"], owner),
                    ).fetchone()
                    is None
                ):
                    raise ValueError("Event identity does not belong to owner")
                result = self.execute(
                    conn,
                    """INSERT INTO cams_events
                    (id,owner_id,session_id,track_id,direction,person_id,person_name,created_at,crossing_key)
                    VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(owner_id,crossing_key) DO NOTHING RETURNING id""",
                    (
                        row["id"],
                        owner,
                        row["session_id"],
                        row["track_id"],
                        row["direction"],
                        row["person_id"],
                        row["person_name"],
                        row["created_at"],
                        row["crossing_key"],
                    ),
                )
                if result.fetchone():
                    row.pop("crossing_key")
                    saved.append(row)
        return saved

    def encodings(self, owner):
        return [
            dict(person, encoding=decode_encoding(person["encoding"]))
            for person in self.persons(owner, private=True)
            if person["encoding_engine"] == "face_recognition_dlib"
        ]


def decode_encoding(value):
    """psycopg JSONB is already decoded; SQLite stores JSON text."""
    return json.loads(value) if isinstance(value, str) else value
