# CamS

CamS es un sistema web de reconocimiento facial, conteo de personas y control
de acceso mediante cámara y RFID.

## Inicio

Desde PowerShell, en la raíz del proyecto:

```powershell
.\scripts\start.ps1
```

Abre:

```text
http://127.0.0.1:5173
```

El backend web escucha en `http://127.0.0.1:8765`. Para detener los procesos:

```powershell
.\scripts\stop.ps1
```

Si una sesión anterior terminó de forma abrupta y el PID registrado fue
reutilizado por Windows, descarta solo el estado obsoleto (sin detener ese
proceso):

```powershell
.\scripts\stop.ps1 -DiscardStaleState
```

La operación no requiere ningún cliente de escritorio ni terminal: las cuentas,
personas, monitor, eventos y estado se gestionan desde la interfaz web. Los
comandos mostrados son únicamente para iniciar, detener o validar el servicio.

## Flujo de acceso

1. Crea una cuenta e inicia sesión en la página.
2. Registra la persona desde el módulo **Personas**.
3. Abre **Monitor** y permite el acceso a la cámara.
4. La página reconoce a la persona antes del cruce.
5. El portal envía `ACCEPTED\n` al Arduino por `COM3` a 9600 baudios.
6. La persona acerca una tarjeta RFID autorizada dentro de 10 segundos.
7. `CamSAccessReceiver.ino` activa el relé, el LED verde y el buzzer.

Carga el firmware desde:

[`micro/CamSAccessReceiver/CamSAccessReceiver.ino`](micro/CamSAccessReceiver/CamSAccessReceiver.ino)

La página y el Arduino deben usar el mismo equipo y el Arduino debe estar
conectado en `COM3`.

## Reconocimiento y almacenamiento

El portal usa MediaPipe para detección y seguimiento, y
`face-recognition/dlib` para generar y comparar embeddings mediante distancia
euclídea. Los rostros registrados se guardan en `.runtime/portal.sqlite3`,
separados por cuenta. El umbral predeterminado es `0.52`.

La cámara debe funcionar en `localhost` o bajo HTTPS. Solo se cuenta un cruce
cuando la identidad fue confirmada; una persona desconocida no genera un evento
de acceso ni una señal al Arduino.

## Dependencias

El arranque administra el entorno `.venv-portal`, instala las dependencias de
visión y ejecuta React, FastAPI y Mosquitto. Para una instalación sin visión:

```powershell
.\scripts\start.ps1 -SkipVision -SkipModels
```

Esa opción permite abrir el panel, pero no permite reconocimiento facial.

## Pruebas

```powershell
Set-Location apps/server
..\..\.venv-portal\Scripts\python.exe -m unittest discover -s src/portal/tests -v
Set-Location ../web
npm.cmd test
npm.cmd run build
```
