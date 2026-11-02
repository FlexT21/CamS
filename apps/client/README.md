# Cliente CamS

Cliente Python que captura vídeo con OpenCV, detecta rostros con MediaPipe y se conecta al servidor mediante WebSocket para reconocer a la persona durante un cruce.

## Requisitos

- Python `3.12.10` o superior.
- Una cámara accesible por OpenCV.
- El servidor CamS ejecutándose y accesible por WebSocket.

## Instalación

Desde esta carpeta:

```powershell
uv sync
```

## Ejecución

Con el servidor ejecutándose en `localhost:8765`:

```powershell
uv run python -m src.main 0 --server ws://localhost:8765/api/ws/
```

El argumento posicional es el índice de la cámara. También puede ser una ruta de vídeo:

```powershell
uv run python -m src.main video.mp4 --server ws://localhost:8765/api/ws/
```

Pulsa `Esc` en la ventana de vídeo para cerrar el cliente.

La ventana muestra la línea azul a la izquierda. En la imagen espejada: derecha → izquierda = entrada; izquierda → derecha = salida. Solo cuentan identidades reconocidas. El total y los eventos se conservan en `data/occupancy.sqlite3`.

Con Arduino Uno conectado, carga `../../arduino/occupancy_led/occupancy_led.ino` y ejecuta:

```powershell
uv run python -m src.main 0 --arduino-port COM3
```

Cada entrada nueva envía ASCII `0` a 9600 baudios para encender el pin 13. Reemplaza COM3 por el puerto de tu placa. El LED permanece encendido; ASCII `1` lo apaga manualmente. Sin `--arduino-port` funciona el conteo y se informa que la señal está desactivada.

Consulta [SYSTEM.md](../../SYSTEM.md) y [VERIFICACION.md](../../VERIFICACION.md).

## Opciones

| Opción | Predeterminado | Descripción |
| --- | --- | --- |
| `cam` | `0` | Índice de cámara o ruta de vídeo. |
| `--server`, `-s` | `ws://localhost:8765/api/ws/` | URL del WebSocket del servidor. |
| `--interval` | `1.0` | Intervalo configurado entre intentos de reconocimiento. |

| `--arduino-port` | desactivado | Puerto serie, por ejemplo COM3. |
| `--line-position` | `0.25` | Fracción horizontal de la línea en la imagen mostrada. |
| `--database` | `data/occupancy.sqlite3` | Registro persistente SQLite. |

El cliente también admite `--register NOMBRE --photos 3` y `--admin` para registro y administración de escritorio.
