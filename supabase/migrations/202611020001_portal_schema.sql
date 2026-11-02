-- Run as the dedicated backend role, never as an anon/authenticated client.
-- Schema intentionally uses cams_ names; browser roles get no table access.
CREATE TABLE IF NOT EXISTS cams_users (
    id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
    password_hash TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS cams_sessions (
    token_hash TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES cams_users(id) ON DELETE CASCADE,
    expires_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS cams_sessions_owner ON cams_sessions(owner_id);
CREATE TABLE IF NOT EXISTS cams_persons (
    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES cams_users(id) ON DELETE CASCADE,
    name TEXT NOT NULL, external_id TEXT NOT NULL, image BYTEA NOT NULL,
    encoding JSONB NOT NULL, encoding_engine TEXT NOT NULL DEFAULT 'unknown', created_at DOUBLE PRECISION NOT NULL,
    UNIQUE(owner_id, external_id)
);
ALTER TABLE cams_persons ADD COLUMN IF NOT EXISTS encoding_engine TEXT NOT NULL DEFAULT 'unknown';
CREATE TABLE IF NOT EXISTS cams_events (
    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES cams_users(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL, track_id INTEGER NOT NULL, direction TEXT NOT NULL CHECK(direction IN ('entrada','salida')),
    person_id TEXT REFERENCES cams_persons(id), person_name TEXT,
    created_at DOUBLE PRECISION NOT NULL, crossing_key TEXT NOT NULL,
    UNIQUE(owner_id, crossing_key)
);
CREATE INDEX IF NOT EXISTS cams_events_owner_time ON cams_events(owner_id, created_at);
CREATE INDEX IF NOT EXISTS cams_events_person ON cams_events(person_id);
CREATE INDEX IF NOT EXISTS cams_sessions_expiry ON cams_sessions(expires_at);
ALTER TABLE cams_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE cams_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE cams_persons ENABLE ROW LEVEL SECURITY;
ALTER TABLE cams_events ENABLE ROW LEVEL SECURITY;
-- No client policies. Only the owner / a dedicated BYPASSRLS backend role works.
REVOKE ALL ON cams_users, cams_sessions, cams_persons, cams_events FROM PUBLIC;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        REVOKE ALL ON cams_users, cams_sessions, cams_persons, cams_events FROM anon;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        REVOKE ALL ON cams_users, cams_sessions, cams_persons, cams_events FROM authenticated;
    END IF;
END $$;
