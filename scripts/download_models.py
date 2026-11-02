"""Check the model files bundled with the existing face-recognition engine."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/server"))
from src.portal.config import Config
from src.portal.vision import FaceEngine, tracker_status

def main():
    engine = FaceEngine(Config())
    status = engine.status()
    detector = tracker_status(Config())
    if status["missing_models"] or status["missing_dependencies"] or detector["missing_dependencies"]:
        raise SystemExit("Install requirements-vision.txt: existing MediaPipe/dlib models or dependencies are missing")
    print("MediaPipe and packaged face-recognition/dlib models are available. No replacement models downloaded.")

if __name__ == "__main__":
    main()
