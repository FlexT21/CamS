# CamS Web

Frontend React y TypeScript en español. Requiere Node.js 22 o posterior.

## Desarrollo

```sh
npm install
npm run dev
```

Abrir http://127.0.0.1:5173. Vite redirige `/api/portal` a `http://127.0.0.1:8765` conservando la ruta completa. El backend debe estar disponible en ese puerto. No se incluyen datos de demostración ni credenciales predeterminadas.

```sh
npm test
npm run build
```

`npm run preview` sirve el build para inspección visual; el proxy de desarrollo solo se configura en `npm run dev`. Para una instalación con API en producción usar nginx o configurar un proxy equivalente.

## Contrato

Todas las rutas tienen el prefijo `/api/portal`. Login y registro son públicos; el resto utiliza `Authorization: Bearer <token>`.

| Método | Ruta | Datos |
| --- | --- | --- |
| POST | /auth/register | `{email,password,name}` → `{access_token,token_type,expires_at,expires_in,user}`; contraseña de 10 a 128 caracteres |
| POST | /auth/login | `{email,password}` → `{access_token,token_type,expires_at,expires_in,user}` |
| GET | /me | `{name,email,...}` |
| GET | /persons | Lista de personas o `{persons:[]}` |
| POST | /persons | `{name,external_id,image}`; `full_name` del formulario se envía como `name` |
| POST | /monitor/frame | `{image,session_id}` → `{tracks,events,message,width,height}` |
| GET | /events | `{events:[{id,session_id,track_id,direction,person_id,person_name,created_at}]}`; fecha en segundos Unix |
| GET | /stats | `{entries,exits,inside,persons}` con valores numéricos |
| GET | /status | `{database:{backend,mode,state},face:{state},tracker:{state},mqtt,serial,...}` |

Las imágenes se capturan como JPEG data URL, con ancho máximo de 1280 px. La captura y el reconocimiento requieren permiso de cámara y un contexto seguro (localhost o HTTPS). La imagen se mantiene en memoria hasta enviar el registro. El token de sesión se guarda en localStorage cuando está disponible y se elimina al cerrar sesión o recibir HTTP 401. La cuenta no implica contratación ni facturación.

El sidebar y el panel central tienen controles independientes. La cámara ocupa la mitad de la fila en escritorio por defecto. «Ampliar panel» oculta la columna auxiliar y amplía la cámara dentro del dashboard, sin entrar en pantalla completa; «Reducir panel» recupera ambas columnas. En móvil las columnas se apilan. El registro muestra la captura, el envío pendiente, la confirmación real y los errores de duplicado encima de la cámara. El botón se bloquea durante el envío y una guarda adicional evita solicitudes duplicadas.

El reconocimiento usa una sola petición a la vez y espera 300 ms después de cada respuesta antes de enviar el siguiente fotograma. Ante HTTP 503 reintenta con espera progresiva de 1 a 15 segundos; una respuesta correcta restablece la cadencia normal. Cada inicio genera un `session_id`; una petición que supera 20 segundos detiene el reconocimiento e informa del error. Detener la cámara, cambiar de módulo o cerrar sesión cancela las peticiones y libera los tracks de `getUserMedia`. Las cajas usan las dimensiones recibidas del backend; la línea visual de referencia está al 25 % del ancho. Los indicadores, personas, eventos y estado se consultan cada 15 segundos y al actualizar manualmente. Los errores se muestran sin sustituirlos por resultados ficticios.

## Docker

```sh
docker build -t cams-web .
docker run --rm --network cams-default -p 5173:8080 cams-web
```

Nginx sirve la SPA en el puerto 8080 y redirige `/api/portal/` a
`portal:8765`, el servicio backend de Compose. Ejecutar ambos servicios en la
misma red Docker; en el ejemplo sustituir `cams-default` por la red real del
despliegue. También se puede utilizar el servicio web del archivo Compose del
proyecto. Para desarrollo local directo utilizar `npm run dev` con el backend
en `127.0.0.1:8765`. No se modifican los archivos de despliegue del
repositorio. En un dominio remoto se requiere HTTPS para la cámara.

Las fuentes de Google son opcionales: si no hay conexión se utiliza la fuente sans-serif local. No se envían fotografías a servicios externos desde el frontend.
