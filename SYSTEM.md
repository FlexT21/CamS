# CamS integrado en Vision Artificial

Instalación del 2 de noviembre de 2026. Raíz de trabajo: `D:\Vision_Artificial\Vision_Artificial`.

## Componentes y motores

| Componente | Implementación |
| --- | --- |
| Interfaz web | React 19, TypeScript, Vite 6, diseño CamS |
| API web | FastAPI, `apps/server/src/portal_main.py` |
| Detección y seguimiento | MediaPipe FaceDetection, un rostro a la vez, como el cliente existente |
| Malla facial | MediaPipe FaceMesh, visible en la zona de la línea |
| Identificación | `face-recognition/dlib`, reutiliza `src.utils.face_encodings` y `src.services.users.recognize_user`; distancia euclídea con `THRESHOLD_DISTANCE=0.52` |
| Cadencia | Un intento por segundo mientras no esté reconocido; confirma con una coincidencia y conserva la identidad |
| Persistencia | SQLite local o PostgreSQL/Supabase mediante psycopg |
| Arduino | CamSAccessReceiver, COM3 a 9600 baudios, RFID + relé |
| Mensajería | Mosquitto local, topic `cams/events` |

El cliente de escritorio original en `apps/client` y los servicios anteriores del servidor se conservan. React agrega el conteo e historial web alrededor de los motores vigentes. **No se sustituyen por YOLO ni SFace.** La referencia copiada de esos motores quedó archivada en `.runtime/cams-reference`, fuera del arranque.

## Arranque y parada

Desde la raíz:

```powershell
.\iniciar_cams.ps1
# Equivalente con dependencias y modelos ya instalados:
.\scripts\start.ps1 -SkipVision -SkipModels
# Parada del portal de esta carpeta:
.\scripts\stop.ps1
```

Interfaz: http://127.0.0.1:5173. API: http://127.0.0.1:8765/api/portal.

Para una instalación nueva, `./iniciar_cams.ps1 -InstalarVision` instala las dependencias y comprueba los modelos vigentes. Python usa `.venv-portal` de esta carpeta. Las versiones del entorno integrado se conservan en `apps/server/requirements-portal.constraints.txt`; las dependencias MediaPipe/dlib están fijadas en `requirements-vision.txt`; no se instala Torch/YOLO. React conserva `package-lock.json`.

Para usar el cliente anterior, ejecuta `./iniciar_escritorio.ps1` con el portal detenido: comparten el puerto 8765 y la cámara. La documentación histórica está en SYSTEM-LEGACY.md. Los directorios de origen y destino no deben ejecutar instancias simultáneas sobre 8765/5173/COM3.

## Supabase

La aplicación ya contiene el driver PostgreSQL, el esquema y su inicializador. La URL pública de un proyecto no permite crear tablas ni autenticar una conexión PostgreSQL. Obtén la cadena privada en Supabase → Connect → Transaction pooler y guárdala en el `.env` de esta raíz:

```dotenv
SUPABASE_DB_URL="postgresql://USUARIO:CONTRASEÑA_CODIFICADA@HOST_POOLER:6543/postgres?sslmode=require"
```

No compartas la contraseña en el chat ni la coloques en variables VITE_. El servidor usa `prepare_threshold=None` para el pooler y mantiene los datos fuera del navegador.

Para crear las tablas y verificar la conexión real:

```powershell
.\.venv-portal\Scripts\python.exe scripts/setup_database.py --require-supabase
```

Para verificar sin cambiar el esquema, agrega `--check-only`. El modo `--require-supabase` devuelve error si falta la conexión; nunca informa Supabase operativo usando SQLite. La aplicación también falla ante una conexión remota configurada pero inaccesible, sin cambiar silenciosamente a la base local.

Las tablas son `cams_users`, `cams_sessions`, `cams_persons` y `cams_events`. La migración `supabase/migrations/202611020001_portal_schema.sql` habilita RLS y revoca acceso a los roles públicos; se ejecuta desde el backend con permisos adecuados. La autenticación es propia de CamS, con scrypt y tokens opacos; no utiliza Supabase Auth.

En esta integración todavía no había una cadena PostgreSQL utilizable. La base SQLite se copió consistentemente desde CamS para conservar el estado existente. Los registros, fotografías y sesiones siguen locales; no se han transmitido a Supabase. Cambiar el DSN no migra automáticamente esos datos a la nube. Reinicia el portal después de configurar la conexión.

Referencia: [conexión PostgreSQL de Supabase](https://supabase.com/docs/guides/database/connecting-to-postgres).

## Cruces y Arduino

En la imagen visible espejada, igual que en el cliente anterior, izquierda → derecha = entrada (+1); derecha → izquierda = salida (−1). Se conserva el lado e identidad durante retornos. La banda neutra es ±max(10 píxeles, 1.5 % del ancho), y la línea está al 25 % del ancho. Solo cuentan cruces con reconocimiento confirmado; los pendientes esperan mientras vive el track.

El cliente envía `ACCEPTED` más LF cuando la cámara reconoce a una persona.
`CamSAccessReceiver.ino` espera entonces una tarjeta RFID autorizada durante 10
segundos antes de activar el relé del pin 7, el LED verde del pin 8 y el buzzer
del pin 4. Una tarjeta no autorizada no abre la cerradura.

El sketch está en `micro/CamSAccessReceiver/CamSAccessReceiver.ino`.

## Datos, respaldo y límites

La base local y logs están en `.runtime`; los modelos dlib se incluyen en el paquete `face-recognition-models`, fuera de Git. El manifiesto `.runtime/integration-manifest.json` registra los archivos incorporados y el respaldo de archivos reemplazados. No se copiaron entornos virtuales ni la carpeta node_modules del origen: ambos se instalaron dentro del destino.

El reconocimiento no incluye prueba de vida y el conteo depende de trayectorias visibles. Las cuentas y datos están separados por propietario. La precisión debe validarse con la cámara y personas reales del lugar.

## Conservación de la dinámica anterior

El video se espeja para mostrarlo, pero la imagen enviada al reconocedor mantiene sus píxeles originales. Los centroides y puntos de la malla se transforman a las coordenadas de la imagen visible. La identidad se obtiene con el mismo servicio previo; no se normalizan los descriptores ni se cambia a similitud coseno. El código original de detección y reconocimiento no se sobrescribió.

Los registros importados tienen metadatos de motor (`encoding_engine`). Si venían de otro algoritmo, se vuelven a calcular desde el JPEG mediante el motor vigente; no se decide compatibilidad solamente por el tamaño del vector. Los casos sin un rostro válido requieren recaptura y no se comparan como si fueran descriptores dlib.
