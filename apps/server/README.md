# Servidor CamS

Backend de la interfaz web. FastAPI gestiona las cuentas, las personas
registradas desde el módulo **Personas**, el reconocimiento de los fotogramas
del módulo **Monitor**, los cruces, los eventos y la comunicación con MQTT y
Arduino.

El servidor se inicia como parte de `scripts/start.ps1`; la operación diaria,
el registro de personas y la consulta de eventos se realizan en la interfaz
web. Instalación: [README principal](../../README.md). Funcionamiento:
[SYSTEM.md](../../SYSTEM.md). Verificación: [VERIFICACION.md](../../VERIFICACION.md).
