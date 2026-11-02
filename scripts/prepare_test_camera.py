"""Build a deterministic synthetic webcam fixture for browser QA."""
from pathlib import Path
import cv2

root = Path(__file__).resolve().parents[1]
source = root / 'apps/server/users/ana/frente.png'
image = cv2.imread(str(source))
if image is None:
    raise SystemExit('Falta la imagen sintética de prueba de Ana')
image = cv2.resize(image, (640, 640))
frame = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).tobytes()
target = root / '.runtime/camera-ana.y4m'
target.parent.mkdir(exist_ok=True)
with target.open('wb') as output:
    output.write(b'YUV4MPEG2 W640 H640 F10:1 Ip A1:1 C420jpeg\n')
    for _ in range(30):
        output.write(b'FRAME\n')
        output.write(frame)
print('Fixture de cámara sintética creada')
