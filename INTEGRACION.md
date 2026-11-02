# Verificación de la integración CamS en Vision Artificial

Fecha: 2 de noviembre de 2026.

## Resultado instalado

Raíz activa: `D:\Vision_Artificial\Vision_Artificial`. React en `apps/web`, adaptador web en `apps/server/src/portal`, entorno independiente `.venv-portal`. Arranque `iniciar_cams.ps1`; el arranque anterior se conserva en `iniciar_escritorio.ps1`.

Los archivos de detección y reconocimiento del cliente y servidor originales se conservaron. El portal usa MediaPipe FaceDetection/FaceMesh y llama a `src.utils.face_encodings` y `src.services.users.recognize_user`. Se mantienen distancia euclídea, `THRESHOLD_DISTANCE`, confirmación con una coincidencia, reintentos cada segundo y video espejado. El conteo y el historial se presentan en React. No se utiliza YOLO ni SFace.

## Validación ejecutada

- 42 pruebas del portal aprobadas, incluidas orientación del centroide en video espejado, permanencia de identidad, entrada/salida, espera de reconocimiento antes de contar, idempotencia, autenticación, aislamiento por cuenta y entrega serie con confirmación.
- 6 pruebas de React aprobadas y compilación TypeScript/Vite exitosa.
- 7 pruebas del cliente anterior y 3 del servidor anterior aprobadas.
- Inferencia real en un proceso de prueba: carga dlib, detección MediaPipe, malla FaceMesh, registro y coincidencia facial con una imagen sintética local; base temporal aislada, sin escrituras de prueba en producción.
- Cuatro registros importados del portal se recodificaron desde sus JPEG usando dlib. Se conservaron IDs, fotografías y eventos. El campo `encoding_engine` impide comparar descriptores de otro algoritmo como si fueran del motor vigente.
- Ruff aprobado en módulos del portal e inicializadores; Python compilado sin errores y parser PowerShell del arranque sin errores.
- Página HTTP 200 y proxy API HTTP 401 sin autenticación, como corresponde.
- El estado de procesos confirma que backend y React se ejecutan desde la carpeta destino.
- Base SQLite verificada con las cuatro tablas `cams_users`, `cams_sessions`, `cams_persons`, `cams_events`.

## Supabase: pendiente de credenciales

El driver psycopg y la migración PostgreSQL están instalados. No se creó ni modificó una base remota: no se dispone de un `SUPABASE_DB_URL` utilizable. La integración opcional de cuenta no fue instalada. Se dejó la alternativa mediante `.env` privado.

`python scripts/setup_database.py --require-supabase` se comprobó y rechaza explícitamente la falta de conexión en lugar de declarar Supabase activo con SQLite. Al configurar un DSN válido, el mismo comando crea/verifica las tablas del portal. La migración incorpora `encoding_engine`, RLS y revocación de acceso de roles públicos.

Los datos copiados permanecen locales. Cambiar la conexión no transfiere automáticamente cuentas, imágenes ni eventos a Supabase.

## Hardware y límites

La prueba real de cámara presencial y el encendido físico del pin 13 no se verificaron en esta integración. El Nano mantiene el fallo de carga documentado en la conversación. Al arrancar se registran fallos de conexión serie si la placa no está disponible; el resto del portal continúa.

El conteo se basa en un rostro reconocido a la vez y no incluye prueba de vida. Los eventos históricos no se borraron ni reclasificaron. Las bibliotecas antiguas emiten advertencias de obsolescencia; se conserva Python 3.12.
