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

La ventana muestra una línea azul punteada cerca del borde derecho, el centroide detectado, un vector de dirección y los contadores de `Ingresaron`, `Salieron` y `Dentro`. En la imagen mostrada, un cruce de derecha a izquierda se registra como `salida`; uno de izquierda a derecha se registra como `entrada`.

## Opciones

| Opción | Predeterminado | Descripción |
| --- | --- | --- |
| `cam` | `0` | Índice de cámara o ruta de vídeo. |
| `--server`, `-s` | `ws://localhost:8765/api/ws/` | URL del WebSocket del servidor. |
| `--interval` | `1.0` | Intervalo configurado entre intentos de reconocimiento. |
