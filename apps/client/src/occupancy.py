"""Persistent occupancy for a single access point, keyed by recognized identity."""

import sqlite3
from pathlib import Path

from src.direction_tracker import Direction


class Occupancy:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS occupants (user TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, timestamp TEXT DEFAULT
                (strftime('%Y-%m-%dT%H:%M:%fZ','now')), user TEXT NOT NULL,
                direction TEXT NOT NULL, inside INTEGER NOT NULL,
                signal TEXT NOT NULL DEFAULT 'not_requested');
        """)

    @property
    def inside(self):
        return self.db.execute("SELECT count(*) FROM occupants").fetchone()[0]

    def totals(self):
        counts = dict(
            self.db.execute("SELECT direction, count(*) FROM events GROUP BY direction")
        )
        return counts.get("entrada", 0), counts.get("salida", 0), self.inside

    def record(self, user: str, direction: Direction):
        if not user or user in ("Unknown", "Desconocido"):
            return None
        with self.db:
            if direction is Direction.ENTRADA:
                changed = self.db.execute(
                    "INSERT OR IGNORE INTO occupants VALUES (?)", (user,)
                ).rowcount
            else:
                changed = self.db.execute(
                    "DELETE FROM occupants WHERE user=?", (user,)
                ).rowcount
            if not changed:
                return None
            return self.db.execute(
                "INSERT INTO events(user,direction,inside) VALUES (?,?,?)",
                (user, direction.value, self.inside),
            ).lastrowid

    def signal_result(self, event_id: int, status: str):
        with self.db:
            self.db.execute("UPDATE events SET signal=? WHERE id=?", (status, event_id))

    def close(self):
        self.db.close()
