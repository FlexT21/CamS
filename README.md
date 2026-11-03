# Portal CamS integrado

La aplicación completa de CamS ahora está instalada en esta carpeta: React en `apps/web`, servidor en `apps/server/src/portal`, modelos locales y entorno `.venv-portal`.

Ejecuta `./iniciar_cams.ps1` y abre http://127.0.0.1:5173. Consulta [SYSTEM.md](SYSTEM.md) para motores de reconocimiento, Supabase, base local, Arduino y límites de verificación. El cliente de escritorio anterior se conserva; se inicia con `./iniciar_escritorio.ps1` y requiere detener el portal.

## Documentación del cliente anterior

# CamS

CamS es un sistema de reconocimiento facial en tiempo real. Un cliente Python captura imágenes de una cámara, detecta rostros y envía los frames al servidor mediante WebSocket. El servidor compara el rostro recibido con las imágenes registradas para cada usuario y publica un evento MQTT cuando reconoce a alguien.

## Características

- Captura de vídeo desde una cámara local o una fuente compatible con OpenCV.
- Detección de rostro en el cliente con MediaPipe.
- Reconocimiento facial en el servidor con `face-recognition`.
- Agrupación de las codificaciones de cada usuario mediante K-Means.
- Comunicación cliente-servidor por WebSocket.
- Publicación de reconocimientos en MQTT.
- Ejecución local o mediante Docker Compose.

## Arquitectura

```text
┌──────────────┐       WebSocket        ┌────────────────────┐
│  apps/client │ ─────────────────────> │   apps/server      │
│ Cámara +     │   metadata + JPEG      │ FastAPI +          │
│ MediaPipe    │ <───────────────────── │ face-recognition   │
└──────────────┘       resultado        └─────────┬──────────┘
                                                   │ MQTT
                                                   ▼
                                          ┌──────────────────┐
                                          │    Mosquitto     │
                                          │ user/recognized  │
                                          └──────────────────┘
```

Al iniciar el servidor, se cargan las imágenes de `apps/server/users/` y sus codificaciones se mantienen en memoria. Si se agregan o modifican imágenes, hay que reiniciar el servidor para recargar los usuarios.

## Requisitos

### Opción Docker Compose

- Docker Desktop con Docker Compose.
- Una cámara accesible desde el equipo donde se ejecuta el cliente.

### Opción local

- Python 3.12.
- `uv` recomendado para instalar las dependencias reproducibles.
- Una cámara compatible con OpenCV.
- Dependencias nativas necesarias para `face-recognition` y OpenCV. En Windows puede ser necesario instalar previamente las herramientas de compilación de C++ y CMake.

## Configuración

Copia `.env.example` como `.env` en la raíz del proyecto:

```powershell
Copy-Item .env.example .env
```

Variables disponibles:

| Variable | Predeterminado | Descripción |
| --- | --- | --- |
| `SERVER_PREFIX` | `/api` | Prefijo de la API HTTP y WebSocket. |
| `SERVER_PORT` | `8765` | Puerto publicado por FastAPI. |
| `SERVER_CORS_ORIGINS` | `["*"]` | Orígenes permitidos por CORS en formato JSON. |
| `VALID_IMAGE_EXTENSIONS` | `[".png",".jpg",".jpeg",".gif"]` | Extensiones aceptadas para las imágenes de usuarios. |
| `THRESHOLD_DISTANCE` | `0.60` | Distancia euclídea máxima para considerar una coincidencia. Un valor menor es más estricto. Ajusta este valor con muestras reales de usuarios registrados y no registrados. |
| `K_MEANS_CLUSTERS` | `3` | Número de centroides por usuario cuando hay suficientes imágenes. |
| `MQTT_BROKER_ADDRESS` | `localhost` | Host del broker MQTT en ejecución local. |
| `MQTT_BROKER_PORT` | `1883` | Puerto MQTT sin TLS. |
| `MQTT_RETRIES_ATTEMPS` | `5` | Número de intentos de conexión MQTT. |
| `MQTT_RETRY_DELAY_SECONDS` | `2` | Espera entre intentos MQTT. |

En Docker Compose, el servidor usa automáticamente `mosquitto` como dirección del broker dentro de la red de Compose.

## Registrar usuarios

Puedes registrar una persona directamente desde la cámara del cliente. Inicia
el servidor y ejecuta el modo de registro indicando el nombre de usuario:

```powershell
uv run python -m src.main 0 --server ws://localhost:8765/api/ws/ --register ana
```

La ventana muestra la cámara; pulsa la barra espaciadora para tomar cada foto y
`Esc` para cancelar. Por defecto se solicitan tres fotos con exactamente un
rostro detectable. Usa `--photos` para cambiar la cantidad. Las imágenes se
guardan en `apps/server/users/<usuario>/` y el servidor actualiza su caché sin
reiniciarse.

Cada usuario debe tener una carpeta con su nombre dentro de `apps/server/users/`. Guarda en ella varias fotografías nítidas del rostro:

```text
apps/server/users/
├── ana/
│   ├── frente.jpg
│   └── perfil.png
└── juan/
    ├── frente.jpg
    └── sonrisa.jpg
```

El nombre de la carpeta es el nombre que aparecerá en el resultado del reconocimiento. Las imágenes deben contener un rostro detectable. Las fotografías sin rostro se ignoran. De manera local el servidor carga esta información una sola vez al iniciar, por lo que debes reiniciarlo después de agregar, eliminar o modificar fotografías.

Se recomienda usar al menos `K_MEANS_CLUSTERS` imágenes válidas por usuario. Con el valor predeterminado, son tres imágenes por usuario. Esta carpeta está ignorada por Git para evitar subir datos biométricos; conserva las fotografías fuera del repositorio y monta o copia los datos en cada entorno.

## Ejecutar con Docker Compose

Desde la raíz del proyecto:

```powershell
docker compose up --build
```

Para ejecutar solo los servicios de infraestructura y servidor:

```powershell
docker compose up --build server mosquitto
```

El servidor quedará disponible en:

- Healthcheck: `http://localhost:8765/api/healthcheck/`
- WebSocket: `ws://localhost:8765/api/ws/`
- Documentación OpenAPI: `http://localhost:8765/api/openapi.json`
- MQTT: `localhost:1883`
- MQTT sobre WebSocket: `localhost:9001`

El directorio `apps/server/users` se monta como volumen en el contenedor, por lo que las imágenes locales se mantienen disponibles después de recrear el servicio.

Para detener los contenedores:

```powershell
docker compose down
```

Los datos persistentes de Mosquitto se encuentran en `mosquitto/data/` y sus logs en `mosquitto/log/`.

## Ejecutar localmente

### Servidor

Desde `apps/server`:

```powershell
uv sync
Copy-Item ..\..\.env .env
uv run python -m src.main
```

El servidor escucha en `0.0.0.0:8765` por defecto. En Linux o macOS, usa `cp ../../.env .env` en lugar de `Copy-Item`.

El broker MQTT debe estar disponible en `localhost:1883`. Puedes levantarlo con Docker sin levantar el servidor en contenedor:

```powershell
docker compose up mosquitto
```

### Cliente

Con el servidor y Mosquitto en ejecución, abre otra terminal en `apps/client`:

```powershell
uv sync
uv run python -m src.main 0 --server ws://localhost:8765/api/ws/
```

El argumento posicional es el identificador de cámara de OpenCV. Para usar una ruta de vídeo o una fuente alternativa, pásala como texto:

```powershell
uv run python -m src.main video.mp4 --server ws://localhost:8765/api/ws/
```

Opciones disponibles:

| Opción | Predeterminado | Descripción |
| --- | --- | --- |
| `cam` | `0` | Índice de cámara o ruta de vídeo. |
| `--server`, `-s` | `ws://localhost:8765/api/ws/` | URL WebSocket del servidor. |
| `--interval` | `1.0` | Segundos entre intentos de reconocimiento. |
| `--arduino-port` | `COM3` | Puerto USB serie del Arduino. Usa `CamSAccessReceiver.ino`. |
| `--line-position` | `0.25` | Posición de la línea respecto al ancho de la imagen mostrada. |
| `--database` | `apps/client/data/occupancy.sqlite3` | Base SQLite de ocupación y eventos. |

Pulsa `Esc` en la ventana de vídeo para cerrar el cliente.

Durante la ejecución, el cliente muestra una línea azul punteada en el lado izquierdo (25 % del ancho de la imagen espejada). Un cruce de **derecha a izquierda es entrada (+1)**; uno de **izquierda a derecha es salida (−1)**. Solo se registra el cruce después de confirmar una identidad. El registro SQLite conserva las personas dentro y los eventos al reiniciar. Una identidad ya dentro no vuelve a sumar por otro cruce de entrada; una identidad fuera no genera una salida negativa.

Para usar el control de acceso, carga `micro/CamSAccessReceiver/CamSAccessReceiver.ino`
en el Arduino y ejecuta el cliente con `--arduino-port COM3`. La cámara envía
`ACCEPTED`; el Arduino abre una ventana de 10 segundos para confirmar una tarjeta
RFID autorizada antes de activar el relé, el LED verde y el buzzer. La señal de
cámara por sí sola no abre la cerradura.

Consulta [SYSTEM.md](SYSTEM.md) para instalación, protocolo, registro persistente, pruebas y límites de operación.

## Protocolo WebSocket

Cada solicitud se envía como dos mensajes consecutivos:

1. Un mensaje JSON con metadatos.
2. Los bytes de una imagen JPEG.

Ejemplo de metadatos:

```json
{
  "type": "face_image",
  "frame_id": 0,
  "device_id": "client_1"
}
```

Respuesta de reconocimiento:

```json
{
  "type": "recognition_result",
  "frame_id": 0,
  "status": "ok",
  "user": "ana",
  "success": true,
  "distance": 0.42
}
```

Si no se detecta exactamente un rostro válido, `status` será `no_face`, `user` será `Unknown` y `success` será `false`.

## API

### `GET /api/healthcheck/`

Comprueba que el servidor está ejecutándose:

```json
{
  "status": "ok",
  "message": "Server is running",
  "extra": {
    "Broker connection": "not_checked"
  }
}
```

### `WS /api/ws/`

Recibe metadatos y una imagen binaria, y devuelve el resultado del reconocimiento. La conexión permanece abierta para procesar varios frames.

## MQTT

Cuando una cara supera el umbral configurado, el servidor publica:

- Topic: `user/recognized`
- Mensaje: `User <nombre> recognized with distance <distancia>`

Para observar los eventos desde una instalación local de Mosquitto:

```powershell
mosquitto_sub -h localhost -p 1883 -t user/recognized -v
```

El broker está configurado actualmente con acceso anónimo y sin TLS. Esa configuración es adecuada para desarrollo local, pero debe endurecerse antes de exponer el sistema a una red no confiable.

## Estructura del proyecto

```text
.
├── apps/
│   ├── client/
│   │   ├── src/              # Captura, tracking, MediaPipe y cliente WebSocket
│   │   ├── pyproject.toml
│   │   └── requirements.txt
│   └── server/
│       ├── src/
│       │   ├── api/          # Healthcheck y WebSocket
│       │   ├── core/         # Configuración y carga de usuarios
│       │   ├── messaging/    # Publicación MQTT
│       │   ├── services/     # Reconocimiento
│       │   └── utils/        # Imágenes, codificaciones y rutas
│       ├── users/            # Fotografías locales de usuarios
│       ├── Dockerfile
│       ├── pyproject.toml
│       └── requirements.txt
├── mosquitto/
│   ├── config/mosquitto.conf
│   ├── data/
│   └── log/
├── docker-compose.yaml
├── .env.example
└── README.md
```
