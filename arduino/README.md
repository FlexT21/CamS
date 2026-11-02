# Desarrollo local y Arduino

Desde PowerShell, en la raiz de CamS:

```powershell
.\scripts\start.ps1
# Compatible con comandos anteriores (ambas opciones ya estan activadas):
.\scripts\start.ps1 -Vision -DownloadModels
# Reutilizar YOLO/torch y modelos ya instalados:
.\scripts\start.ps1 -SkipVision -SkipModels
.\scripts\stop.ps1
```

Requisitos: Windows, Python Launcher `py`, modulo `uv` disponible mediante
`py -3 -m uv`, Node.js y `npm.cmd`. Ejecute los comandos desde la raiz del
repositorio completo: el arranque utiliza `apps/server/requirements-portal.txt`,
`apps/server/src/portal_main.py` y `apps/web/package.json`.
El portal carga el `.env` de la raiz y respeta las variables del proceso.
El script configura el backend en `127.0.0.1:8765` y la interfaz en
`http://127.0.0.1:5173`.

El arranque crea `D:\Vision_Artificial\Vision_Artificial\.venv-portal` (relativo a la raiz
del checkout) con `py -3 -m uv venv --python 3.12`
(uv puede descargar ese Python aislado) e instala los requisitos del portal
mediante `uv pip install --python ...`. Por defecto instala tambien las
dependencias de vision en ese mismo entorno: primero `torch` y `torchvision`
desde `https://download.pytorch.org/whl/cpu`, despues
`apps/server/requirements-vision.txt` (YOLO). Comprueba que PyTorch sea CPU
tras instalar. No requiere CUDA ni descarga sus dependencias GPU.
Las dependencias de reconocimiento facial estan en `requirements-portal.txt`.
`-SkipVision` omite la instalacion de YOLO/torch, conservando cualquier
instalacion previa. `-Vision` y su alias `-WithVision` siguen siendo compatibles.
Por defecto ejecuta `scripts/download_models.py` antes de iniciar servicios;
este programa descarga a `.runtime/models` los modelos
faltantes y conserva los ya disponibles. `-SkipModels` omite este paso.
`-DownloadModels` sigue siendo compatible. Los switches `-SkipVision` y
`-SkipModels` tienen prioridad sobre sus respectivas opciones de activacion.
Ejecuta `npm.cmd install` en `apps/web`. No instala Docker, Mosquitto ni
paquetes grandes del sistema.
Solo copia `.env.example` a `.env` cuando `.env` no existe. Configure alli sus
valores reales; no incluya credenciales en los scripts ni en Git.

El backend se ejecuta en el host con `python -m src.portal_main`, desde
`apps/server`, y la interfaz en `http://127.0.0.1:5173`; el backend usa
`http://127.0.0.1:8765`. Los procesos se inician ocultos. Sus salidas e
identidades quedan en `.runtime/`, ignorado por Git mediante el patron `.*`
del repositorio.

MQTT local usa `127.0.0.1:1883`. Se prefiere el servicio `mosquitto` de
`docker-compose.yaml` si Docker y su motor funcionan. Solo se arranca ese
servicio. Si no esta disponible, se busca `mosquitto.exe` en PATH o en
`Program Files/mosquitto`, usando su listener local por defecto. No se reutiliza
la configuracion del contenedor en Windows porque contiene rutas Linux.
Un listener existente se conserva sin adquirir su propiedad. La comprobacion
de disponibilidad es TCP; consulte el estado de conexion MQTT del backend
en el dashboard y en `.runtime/backend.out.log` y `.runtime/backend.err.log`.
Sin broker, se muestra una advertencia explicita y se permite iniciar el
dashboard, cuyos eventos MQTT no funcionaran hasta disponer del broker.

La parada valida PID, fecha de creacion, ejecutable y comando antes de detener
cada proceso y sus descendientes. Solo detiene el contenedor que este arranque
creo y cuya identidad y etiquetas Compose coinciden. No ejecuta `compose down`
ni elimina datos. Si Docker no responde, conserva el registro para reintentar.
Si un PID activo no coincide con su identidad guardada, la parada falla y
conserva `.runtime/processes.json`; revise el conflicto antes de reintentar.
Ejecute la parada antes de volver a arrancar; no borre manualmente
`.runtime/processes.json` mientras haya procesos activos.

## Indicador serial

Abra `arduino/door_access/door_access.ino` en Arduino IDE, seleccione placa y
puerto y cargue el sketch. Solo se usa `pin 13`; no conecta ni activa
reles ni una cerradura real. El nivel electrico HIGH/LOW se aplica exactamente
como se indica, aunque algunas placas tienen un LED fisicamente activo en LOW.

Protocolo: **9600 baudios**, comandos ASCII terminados en LF (CRLF permitido):

| Evento | Enviar | Salida LED | Respuesta |
| --- | --- | --- | --- |
| Entrada | `0\n` | HIGH | `ACK 0\r\n` |
| Salida | `1\n` | LOW | `ACK 1\r\n` |

Lineas vacias, valores invalidos y comandos multiples en una linea se ignoran
sin ACK. Un comando incompleto espera el salto de linea. Al reiniciar, la
salida queda LOW; abrir el puerto puede reiniciar determinadas placas.

Configure `SERIAL_PORT=COM3` en el `.env` de la raiz, sustituyendo COM3 por el
puerto real del Administrador de dispositivos, y `SERIAL_BAUD=9600` para
coincidir con el sketch. Guarde estos valores en `.env` antes de iniciar el
portal; el arranque conserva ese archivo sin modificarlo. El backend abre
el puerto configurado y envia los comandos de entrada/salida. El sketch
responde con el ACK correspondiente a cada comando valido.
El sketch por si solo no consume MQTT. Cierre el monitor serie de Arduino IDE
antes de arrancar el backend: un solo proceso puede abrir el puerto.
El proceso que usa COM debe ejecutarse en el host Windows; el contenedor del
broker no obtiene acceso al puerto COM. La deteccion de personas y la
comunicacion MQTT/serial se ejecutan en el backend local.

Corrección del protocolo solicitada: entrada = ASCII `0` seguido de LF, pin 13 HIGH, respuesta `ACK 0`; salida = ASCII `1` seguido de LF, pin 13 LOW, respuesta `ACK 1`. Cargar el sketch en Arduino IDE antes de usar el portal.

## Diagnóstico de conexión

El backend exige ahora la confirmación `ACK 0` al encender el pin 13 y `ACK 1` al apagarlo. Si el puerto abre pero no llega la respuesta del firmware en un segundo, la integración se marca desconectada y registra un error; ya no basta con que Windows acepte la escritura USB. El puerto de esta instalación se configura en `.env` mediante `SERIAL_PORT` y requiere reiniciar el backend. La confirmación comprueba ejecución del firmware, no mide eléctricamente el pin.

## Placa de esta instalación: Arduino Nano

Puerto detectado: COM3 (CH340), comunicación del programa: 9600 baudios, LED en D13. En Arduino IDE, selecciona Arduino Nano y prueba el procesador ATmega328P (Old Bootloader) para esta placa: su cargador respondió a 57600 baudios. La velocidad de carga es distinta de los 9600 baudios que utiliza el programa.

El sketch también compila para `arduino:avr:nano:cpu=atmega328old`. La compilación por sí sola no acredita que se haya cargado correctamente: hay que verificar la memoria y obtener `ACK 0` / `ACK 1` desde la placa.
