# CamS web con los motores vigentes de Vision Artificial

El portal React usa MediaPipe y face-recognition/dlib mediante los adaptadores
del servidor. No arranca YOLO, SFace ni ByteTrack.

## Inicio

```powershell
cd D:\Vision_Artificial\Vision_Artificial
.\scripts\start.ps1
```

Abre http://127.0.0.1:5173. API: http://127.0.0.1:8765/api/portal. El navegador abre la cámara. `scripts/stop.ps1` detiene los procesos administrados.

## Reconocimiento y conteo

MediaPipe FaceDetection detecta un rostro a la vez; el video se muestra espejado como antes. La malla MediaPipe FaceMesh aparece cerca de la línea. El JPEG original, sin anotaciones y sin espejar los píxeles, se procesa con `src.utils.face_encodings`. La identificación llama al mismo `src.services.users.recognize_user` que usa el servidor anterior, con distancia euclídea y `THRESHOLD_DISTANCE` (0.52 en esta instalación).

Se reintenta cada segundo mientras no haya identidad; una coincidencia confirma el rostro, como en el cliente previo. La identidad se conserva durante la trayectoria. Si cruza antes de identificarse, el cruce espera mientras siga visible. Rostros sin confirmar no aumentan el contador ni generan una señal de acceso. Si aparecen varios rostros, no se elige arbitrariamente el primero.

En la imagen mostrada: izquierda → derecha es entrada (+1), derecha → izquierda es salida (−1). Se usa el centro del rostro, no una caja de cuerpo YOLO. La línea está al 25 % del ancho y tiene una banda neutra de ±max(10 píxeles, 1.5 % del ancho). Los eventos guardados aparecen en los contadores e historial web; la presencia es una estimación basada en entradas menos salidas, sin valores negativos.

Registro: captura el rostro en el módulo de personas; se guarda el JPEG y el embedding dlib con la marca `encoding_engine=face_recognition_dlib`. Las imágenes importadas del portal anterior se recodifican localmente con el motor vigente para evitar mezclar descriptores de otro algoritmo. El alta web guarda los rostros en la base de datos de la cuenta y no comparte
automáticamente rostros entre cuentas.

## Base de datos y Supabase

SQLite local en `.runtime/portal.sqlite3`, o PostgreSQL configurado mediante `SUPABASE_DB_URL` en `.env`. Supabase no está conectado todavía: hace falta una cadena PostgreSQL privada válida. No basta con la URL pública ni con una clave anon.

```powershell
.\.venv-portal\Scripts\python.exe scripts/setup_database.py --require-supabase
```

Este comando crea/verifica `cams_users`, `cams_sessions`, `cams_persons`, `cams_events`; sin credenciales falla explícitamente. El esquema está en `supabase/migrations/202611020001_portal_schema.sql`. Usa RLS y revoca el acceso de los roles públicos; las operaciones pasan por FastAPI y filtran el propietario. La autenticación propia de CamS se conserva. Consulta [SYSTEM.md](SYSTEM.md).

## Control de acceso

El portal controla el receptor `COM3` a 9600 baudios. Carga
`micro/CamSAccessReceiver/CamSAccessReceiver.ino` en el Arduino. El portal
envía `ACCEPTED` al confirmar una identidad de entrada; después la tarjeta RFID
autorizada habilita el relé.

## Verificación

```powershell
cd apps/server
..\..\.venv-portal\Scripts\python.exe -m unittest discover -s src/portal/tests -v
cd ../web
npm.cmd test
npm.cmd run build
```

Las versiones del entorno instalado se fijan en `requirements-portal.constraints.txt`. Las dependencias de visión están en `requirements-vision.txt`, con MediaPipe y dlib; no hay instalación de Torch/YOLO. `scripts/download_models.py` ahora comprueba los modelos empaquetados del reconocedor existente, sin descargar modelos de reemplazo.
