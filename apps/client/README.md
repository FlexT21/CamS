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

### Señal de acceso por Arduino

El cliente puede enviar la decisión de reconocimiento a un Arduino por serial.
Carga `../../micro/CamSAccessReceiver/CamSAccessReceiver.ino` en el Arduino y conecta
el dispositivo al equipo. El sketch usa `9600` baudios y conserva el acceso RFID
con las tarjetas configuradas en el código. Primero, `ACCEPTED` verifica a la
persona por cámara y abre una ventana de 10 segundos para presentar una tarjeta
autorizada; solo después de ambas verificaciones activa el relé del pin 7, el LED
verde del pin 8 y el buzzer del pin 4 durante cinco segundos. `DENIED`, una tarjeta
no autorizada o una tarjeta presentada sin verificación de cámara mantienen el
relé apagado, encienden el LED rojo del pin 6 y emiten dos pitidos.

Inicia el cliente indicando el puerto serial (por ejemplo, `COM3` en Windows):

```powershell
uv run python -m src.main 0 --server ws://localhost:8765/api/ws/ --arduino-port COM3
```

La señal enviada es una línea ASCII terminada en salto de línea: `ACCEPTED`
para un usuario reconocido. Los rechazos de cámara no envían una señal al
Arduino, para no cancelar una verificación válida de otra persona. La tarjeta
solo se evalúa después de recibir `ACCEPTED`. El puerto predeterminado es `COM3`;
puedes cambiarlo con `--arduino-port` cuando uses otro puerto.

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
| `--register [USERNAME]` | - | Abre la interfaz de registro; si no se indica el nombre, lo solicita. |
| `--photos` | `3` | Número de fotos válidas requeridas durante el registro. |
