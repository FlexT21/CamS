# Adaptador web del reconocimiento vigente

`vision.py` adapta MediaPipe FaceDetection/FaceMesh y las funciones existentes `src.utils.face_encodings` y `src.services.users.recognize_user`. Reutiliza el umbral `THRESHOLD_DISTANCE`, distancia euclídea, una coincidencia y reintentos cada segundo. No utiliza YOLO ni SFace.

`monitor.py` conserva la identidad y el lado estable de un rostro por trayectoria, deja pendientes los cruces sin confirmar identidad y registra el conteo de la página web. La cámara se muestra espejada; la identificación recibe los píxeles originales como antes.

Las personas del portal se almacenan con `encoding_engine=face_recognition_dlib`; cada alta nueva conserva tres fotografías y tres embeddings, y el reconocimiento compara contra los tres. `scripts/migrate_face_encodings.py --apply` reconstruye los descriptores desde sus JPEG conservando IDs, nombres y eventos. El índice de registro (`card_slot`) vincula la primera persona a la primera tarjeta, la segunda a la segunda, y así sucesivamente.

La autenticación, aislamiento por cuenta, persistencia y esquema Supabase pertenecen al portal. Ver [README-PORTAL.md](../../../../README-PORTAL.md) y [SYSTEM.md](../../../../SYSTEM.md).
