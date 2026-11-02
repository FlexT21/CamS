"""Rebuild portal encodings from stored JPEGs using the existing dlib engine.

This changes only descriptor values, retaining names, IDs, JPEGs and event links.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/server"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    from src.portal.config import Config
    from src.portal.database import Database
    from src.portal.vision import FaceEngine
    import cv2
    import numpy as np

    db = Database(Config())
    db.initialize()
    faces = FaceEngine(db.config)
    with db.connect() as conn:
        rows = db.execute(
            conn, "SELECT id,image,encoding,encoding_engine FROM cams_persons"
        ).fetchall()
    converted = failed = unchanged = 0
    for row in rows:
        if row["encoding_engine"] == "face_recognition_dlib":
            unchanged += 1
            continue
        frame = cv2.imdecode(
            np.frombuffer(bytes(row["image"]), np.uint8), cv2.IMREAD_COLOR
        )
        values = faces.embeddings(frame) if frame is not None else []
        if len(values) != 1:
            failed += 1
            continue
        if args.apply:
            with db.connect(write=True) as conn:
                db.execute(
                    conn,
                    "UPDATE cams_persons SET encoding=?, encoding_engine='face_recognition_dlib' WHERE id=?",
                    (json.dumps(values[0]), row["id"]),
                )
        converted += 1
    print(
        "Registros compatibles:",
        unchanged,
        "Convertidos" if args.apply else "Convertibles",
        converted,
        "Necesitan nueva captura:",
        failed,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
