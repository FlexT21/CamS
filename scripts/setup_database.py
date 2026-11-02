"""Create or verify CamS tables using this checkout's private environment."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/server"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-supabase",
        action="store_true",
        help="Refuse SQLite: require a usable PostgreSQL connection.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check existing tables without creating them.",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "apps/server/.env")
    from src.portal.config import Config
    from src.portal.database import Database

    try:
        config = Config()
        if args.require_supabase and not config.database_url:
            print(
                "Supabase pendiente: configura SUPABASE_DB_URL en el .env de esta carpeta.",
                file=sys.stderr,
            )
            print(
                "La URL pública y la clave anon no sustituyen la conexión PostgreSQL.",
                file=sys.stderr,
            )
            return 2
        db = Database(config)
        if not args.check_only:
            db.initialize()
        with db.connect() as conn:
            if db.postgres:
                rows = db.execute(
                    conn,
                    "SELECT tablename AS name FROM pg_tables WHERE schemaname='public' AND tablename IN ('cams_users','cams_sessions','cams_persons','cams_events')",
                ).fetchall()
            else:
                rows = db.execute(
                    conn,
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('cams_users','cams_sessions','cams_persons','cams_events')",
                ).fetchall()
        tables = sorted(row["name"] for row in rows)
        expected = ["cams_events", "cams_persons", "cams_sessions", "cams_users"]
        if tables != expected:
            print(
                "Esquema incompleto. Ejecuta sin --check-only para instalar las tablas.",
                file=sys.stderr,
            )
            return 1
        print(
            "Base verificada:",
            "PostgreSQL / Supabase" if db.postgres else "SQLite local",
        )
        print("Tablas:", ", ".join(tables))
        return 0
    except Exception as error:
        # Never echo a DSN, SQL parameters, passwords or personal records.
        print(
            f"No se pudo verificar la base ({type(error).__name__}). Revisa credenciales, conectividad y permisos.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
