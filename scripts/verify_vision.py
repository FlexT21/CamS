"""Verify actual MediaPipe and existing dlib recognition without production writes."""
import argparse
import base64
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'apps/server'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixtures-directory', type=Path)
    args = parser.parse_args()
    import cv2
    import numpy as np
    from src.portal.config import Config
    from src.portal.service import Portal
    from src.portal.vision import FaceEngine, MediaPipeTracker, match
    config = Config(database_url='', mqtt_host='', serial_port='')
    engine = FaceEngine(config)
    blank = np.zeros((320,320,3), dtype=np.uint8)
    assert engine.embeddings(blank) == []
    detector = MediaPipeTracker(config)
    try:
        assert detector.detect(blank) == []
        if args.fixtures_directory:
            path = args.fixtures_directory/'ana'/'frente.png'
            image = cv2.imread(str(path))
            if image is None:
                raise SystemExit('Falta la imagen sintética de prueba')
            with tempfile.TemporaryDirectory() as temporary:
                portal = Portal(replace(config, sqlite_path=str(Path(temporary)/'test.sqlite3')))
                portal.db.initialize()
                account = portal.register('verification@example.test', 'isolated-verification-only', 'Verification')
                ok,jpeg = cv2.imencode('.jpg',image)
                assert ok
                person = portal.enroll(account['user']['id'],'Synthetic fixture','fixture',base64.b64encode(jpeg).decode())
                embedding = engine.embeddings(image)
                assert len(embedding) == 1
                assert match(embedding[0],portal.db.encodings(account['user']['id']),config.face_threshold)['id'] == person['id']
                assert len(detector.detect(image)) == 1
                assert engine.landmarks(image), 'La malla facial no detectó el rostro'
        print('REAL_VISION_OK: MediaPipe, FaceMesh y reconocedor face-recognition/dlib vigente')
    finally:
        detector.close()
        engine.close()


if __name__ == '__main__':
    main()
