# Informe de verificación de CamS

Fecha: 2 de noviembre de 2026. Revisión de los módulos Python de cliente y servidor, los tres README existentes, configuración, Docker y script de inicio. Se creó SYSTEM.md para documentar la operación completa.

## Hallazgos y correcciones

| Hallazgo anterior | Resultado de la corrección |
| --- | --- |
| Derecha → izquierda se consideraba salida | Ahora es entrada; el recorrido contrario es salida |
| Línea al 82 % del ancho, a la derecha | Línea al 25 %, a la izquierda, configurable |
| Cruces contabilizados sin identidad | Se exige reconocimiento confirmado; se conservan cruces pendientes mientras vive el track |
| Bucle OpenCV sin cesión al event loop | La tarea de reconocimiento ahora puede ejecutarse durante la captura |
| `--interval` no se utilizaba | Regula los reintentos por track |
| Tracking reiniciado después de cada cruce | Se conserva la identidad y el lado; SQLite impide entradas o salidas duplicadas |
| Totales en memoria y resta recortada a cero | Registro transaccional por identidad, ocupación persistente y eventos UTC |
| No había implementación Arduino | Cliente USB serie, confirmación y firmware del Uno para el pin 13 |
| Reintentos individuales podían separar JSON y JPEG | Reintento de la transacción completa; cierre de conexión al cancelar y validación de frame_id |
| Servidor recibía BGR donde se esperaba RGB | Conversión a RGB antes de codificar los rostros |
| JPEG con anotaciones y malla invisible por orden de dibujo | Reconocimiento sobre imagen limpia y malla visible en la imagen final |
| Sin usuarios, la distancia infinita podía invalidar el JSON | Se devuelve `distance: null` |
| Varios rostros podían asociarse a la primera identidad | Cliente pausa el seguimiento y servidor exige exactamente un rostro |
| MQTT podía impedir responder al reconocimiento | Respuesta antes de publicar; publicación con QoS 1 y confirmación, fuera del hilo asíncrono |
| Healthcheck afirmaba que MQTT funcionaba sin medirlo | Informa `not_checked` |
| Rutas dependientes del directorio de ejecución | Carpeta del servidor resuelta desde el archivo fuente |
| Docker declaraba Python 3.10 aunque el proyecto requiere 3.12 | Base Python 3.12 y corrección de continuación de línea; usuarios mediante volumen, sin fotos en la imagen |

## Verificación ejecutada

**10 pruebas automatizadas aprobadas**, con Python 3.12 en los entornos locales `.venv312`:

- Cliente: siete pruebas, incluyendo el bucle de cámara simulado de entrada y salida, reconocimiento que responde después del cruce, una sola señal por entrada, banda de tolerancia, pérdida de track, persistencia y duplicados.
- Arduino: escritura exacta de `b'ACCEPTED\n'` y cierre de puerto mediante
  puerto simulado; el firmware `CamSAccessReceiver` confirma después la tarjeta
  RFID.
- Red: intercambio real en localhost con un servidor WebSocket de prueba y cancelación de una solicitud pendiente.
- Servidor: tres pruebas sobre base de reconocimiento vacía, coincidencia y rechazo por umbral, conversión RGB y ruta FastAPI WebSocket con JPEG inválido/vacío y varios rostros.
- Ruff: todos los archivos fuente y pruebas pasan el análisis; formato aplicado.
- Compilación sintáctica Python (`compileall`): sin errores.
- Parser PowerShell del script de inicio: sin errores.
- `git diff --check`: sin errores de espacios.
- Locks de cliente y servidor resueltos con uv; pySerial agregado al cliente.

Comandos reproducibles, dentro de cada aplicación:

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s tests -v
.\.venv312\Scripts\python.exe -m ruff check src tests
```

Con una instalación nueva mediante uv, usa `uv run python` y `uv run ruff` en lugar de las rutas del entorno existente.

## Lo que requiere validación física

No se detectaron puertos serie al ejecutar `serial.tools.list_ports`. No se cargó el programa en un Arduino ni se observó un LED real. Arduino CLI y Docker no estaban disponibles, por lo que no se compiló el firmware ni se construyó la imagen de Docker. La captura y el recorrido se comprobaron con frames y detecciones simuladas, sin una prueba presencial de cámara.

La carga local produjo **cero codificaciones de usuarios registrados**. Se verificó el algoritmo con vectores controlados y el protocolo con JPEG, pero no se midió exactitud facial con fotografías de personas registradas. Debes registrar fotografías válidas y reiniciar el servidor antes de usarlo con personas reales.

Las librerías instaladas emiten avisos de obsolescencia de Protobuf y `pkg_resources`; las pruebas pasan con Python 3.12. No se afirma compatibilidad con Python 3.14 ni exactitud biométrica del 100 %.

El sistema exige paso individual y registra solo personas reconocidas. Consulta SYSTEM.md para la prueba presencial, el estado inicial vacío y los límites de identidad y ocupación.

## Entrega

`entrega/CamS_corregido_Arduino_Uno_pin13.zip`: código fuente completo, dependencias y locks, pruebas, documentación y sketch Arduino. Excluye entornos Python, `.env` locales, imágenes biométricas, bases de ocupación y logs. Las fotografías originales permanecen en el equipo; para otra instalación deben agregarse a `apps/server/users/<nombre>/`.
