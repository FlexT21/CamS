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

Para registrar una persona desde la misma cámara, inicia el cliente con el nombre
de usuario. Pulsa la barra espaciadora para capturar cada foto y `Esc` para
cancelar:

```powershell
uv run python -m src.main 0 --server ws://localhost:8765/api/ws/ --register ana
```

Por defecto se solicitan tres fotos válidas. Puedes cambiar la cantidad con
`--photos`. El servidor guarda las imágenes en `apps/server/users/<usuario>/` y
actualiza el reconocimiento sin reiniciarse.

## Interfaz administrativa

También puedes abrir un panel de administración para usar el reconocimiento y el
registro desde una sola ventana:

```powershell
uv run python -m src.main --admin --server ws://localhost:8765/api/ws/
```

El panel solicita el usuario `admin` y la contraseña inicial `admin123`. Ambos
valores se almacenan únicamente como hashes PBKDF2 dentro del código; no se
guarda la contraseña en texto plano. Después del inicio de sesión, la cámara se
selecciona desde un desplegable con los dispositivos detectados; usa
`Actualizar` para volver a escanearlos. También permite indicar la URL del
servidor, el nombre de usuario y la cantidad de fotos de registro. Cada
operación abre la ventana de cámara existente; pulsa `Esc` para finalizarla y
volver al panel.

La ventana administrativa tiene un área ampliada para visualizar mejor la
cámara. El reconocimiento facial solo se ejecuta antes de la línea de cruce;
cuando una persona ya la cruzó, o se desplaza hacia la salida, no se envía su
imagen al servidor. La interfaz de vídeo muestra la línea de cruce, el
centroide y las flechas con los cálculos de dirección.
En particular, el área derecha de la línea se usa únicamente para detectar y
registrar salidas: ningún rostro ubicado allí se envía al reconocimiento.

La ventana muestra una línea azul punteada cerca del borde derecho, el centroide detectado, un vector de dirección y los contadores de `Ingresaron`, `Salieron` y `Dentro`. En la imagen mostrada, un cruce de derecha a izquierda se registra como `salida`; uno de izquierda a derecha se registra como `entrada`.

## Opciones

| Opción | Predeterminado | Descripción |
| --- | --- | --- |
| `cam` | `0` | Índice de cámara o ruta de vídeo. |
| `--server`, `-s` | `ws://localhost:8765/api/ws/` | URL del WebSocket del servidor. |
| `--interval` | `1.0` | Intervalo configurado entre intentos de reconocimiento. |
| `--register [USERNAME]` | - | Abre la interfaz de registro; si no se indica el nombre, lo solicita. |
| `--photos` | `3` | Número de fotos válidas requeridas durante el registro. |
