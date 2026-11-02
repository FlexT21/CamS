# Sistema CamS: reconocimiento, ocupación y Arduino Uno

Especificación revisada el 2 de noviembre de 2026. El pin solicitado es **13**.

## Comportamiento

1. La cámara se muestra espejada. La línea está a la izquierda, al 25 % del ancho; `--line-position 0.25` permite cambiarla.
2. Se solicita reconocimiento en cuanto aparece un rostro, antes del cruce. `--interval` regula los reintentos. El JPEG se envía limpio, sin líneas ni malla.
3. El centro del rostro debe pasar de un lado estable de la línea al otro. La banda de tolerancia evita disparos por pequeños movimientos sobre la línea.
4. Derecha → izquierda: **entrada**. Izquierda → derecha: **salida**. Aparecer, desaparecer o permanecer sobre un lado no cuenta.
5. Si el cruce ocurre mientras llega la respuesta, queda pendiente en el track y se aplica al confirmar la identidad. Si se pierde el track antes de identificarlo, se descarta. Un rostro no reconocido nunca activa el LED.
6. La entrada agrega la identidad a las personas dentro. Una entrada repetida para la misma identidad se ignora hasta que se registre su salida. La salida elimina la identidad; si no estaba dentro, se ignora. El conteo nunca es negativo.
7. Cada entrada aceptada se guarda y envía ASCII `0` por USB serie al Arduino, si el puerto fue configurado. Una salida no envía comandos al LED.

## Preparación y ejecución en Windows

Necesitas Python 3.12, dependencias del servidor, fotografías registradas y un broker MQTT para las notificaciones. El reconocimiento puede responder aunque MQTT falle; sus publicaciones son independientes del comando USB.

Desde `apps/server`:

```powershell
uv sync
Copy-Item ..\..\.env .env
uv run python -m src.main
```

Desde `apps/client`:

```powershell
uv sync
uv run python -m serial.tools.list_ports
uv run python -m src.main 0 --server ws://localhost:8765/api/ws/ --arduino-port COM3
```

Reemplaza COM3 por el puerto real. Alternativa con los entornos `.venv312` ya instalados: desde la raíz, `./iniciar_cams.ps1 -ArduinoPort COM3`. El script necesita Mosquitto instalado. Para actualizar un entorno existente: `./apps/client/.venv312/Scripts/python.exe -m pip install pyserial==3.5`.

Sin `--arduino-port`, el programa conserva el conteo y avisa que Arduino está desactivado. Si un puerto configurado no abre, falla con un error en lugar de fingir que envió la señal. Cierra el monitor serie antes de ejecutar el cliente.

## Arduino Uno, pin 13

Abre `arduino/occupancy_led/occupancy_led.ino` en Arduino IDE, elige Arduino Uno y el puerto correspondiente, y carga el programa. El LED integrado permite comprobar el pin 13 sin cablear un LED adicional.

| Entrada serie | Acción física | Confirmación |
| --- | --- | --- |
| Carácter `0` (ASCII 0x30) | `digitalWrite(13, HIGH)` | `LED_ON` con salto de línea |
| Carácter `1` (ASCII 0x31) | `digitalWrite(13, LOW)` | `LED_OFF` con salto de línea |
| Otros caracteres | Sin cambios | Ninguna |

Velocidad: 9600 baudios, 8N1. El cero es un comando de aplicación; para encender el LED se escribe HIGH en el pin. No se envía un byte NUL ni se aplica LOW al pin. Al abrir el puerto se esperan dos segundos por el reinicio del Uno. El LED queda encendido después de la entrada; no se apaga al salir ni al cerrar el cliente. Se apaga con `1` o al reiniciar la placa.

El cliente espera confirmación durante un segundo. Si falla, conserva la entrada y registra `failed`; muestra el error y no reenvía automáticamente un comando cuya entrega podría ser incierta. `acknowledged` confirma la respuesta del firmware, no mide eléctricamente el LED.

Referencias del protocolo: [pySerial API](https://pyserial.readthedocs.io/en/stable/pyserial_api.html) y [Arduino digitalWrite](https://docs.arduino.cc/language-reference/en/functions/digital-io/digitalwrite/).

## Registro persistente

Por defecto: `apps/client/data/occupancy.sqlite3`. Se crea automáticamente y no se incluye en Git ni en la entrega ZIP.

- `occupants`: identidades actualmente dentro.
- `events`: identificador, fecha UTC, usuario, dirección, ocupación resultante y estado de señal.
- `signal`: `not_requested` para salida, `disabled` sin Arduino, `acknowledged` con confirmación, `failed` si hay error. Una terminación entre guardar y enviar puede dejar `not_requested` en una entrada: entrega no confirmada.

`Ingresaron` y `Salieron` muestran eventos acumulados de esa base; `Dentro` cuenta identidades presentes. Los cambios de ocupación y eventos se guardan en una sola transacción. Para consultar con Python:

```powershell
python -c "import sqlite3; c=sqlite3.connect('apps/client/data/occupancy.sqlite3'); print(c.execute('SELECT * FROM events ORDER BY id').fetchall())"
```

La base empieza vacía. Para comenzar una sesión vacía, usa otra ruta con `--database data/nueva_sesion.sqlite3`, solo cuando el recinto esté vacío. El sistema no reconstruye personas que ya estaban dentro antes de empezar ni cruces fuera de cámara.

## Límites operativos y comprobación física

El seguimiento soporta **una persona a la vez**. Si detecta varios rostros, descarta el track y pausa el conteo. No es un detector de cuerpos ni un sistema de seguimiento multiobjeto. Un relevo rápido de personas sin pérdida visible del rostro puede confundir la identidad; el acceso debe ser individual. El conteo representa personas reconocidas, por lo que desconocidos o rostros ocultos pueden dejarlo distinto de la ocupación real. No incluye detección de suplantación.

No ejecutes varios clientes sobre el mismo acceso/base: la señal USB y la asociación de tracks están diseñadas para un único cliente. Identidades de carpetas distintas para una misma persona se cuentan por separado. Los cambios de nombres de usuarios no migran el registro existente.

Validación en el equipo de destino: comenzar con recinto y base vacíos, registrar una persona, cruzar desde la derecha hacia la izquierda y confirmar +1, evento con identidad y LED encendido. Permanecer dentro no debe sumar. Regresar hacia la derecha debe dejar 0. Un rostro desconocido no debe sumar ni enviar `0`. Cerrar y reabrir el cliente debe conservar el registro. Separar físicamente a cada persona durante el paso.
