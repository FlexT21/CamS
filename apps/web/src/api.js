export const API_PREFIX = "/api/portal";
const messages = {
  'Face is already enrolled': 'Este rostro ya está registrado en tu cuenta.',
  'External ID is already enrolled': 'Este identificador ya está registrado en tu cuenta.',
  'Enrollment requires exactly one face': 'La fotografía debe contener exactamente un rostro visible.',
  'Invalid email or password': 'El correo o la contraseña son incorrectos.',
  'Account already exists': 'Ya existe una cuenta con ese correo.',
  'Invalid or expired session': 'La sesión venció. Inicia sesión nuevamente.',
  'Face inference failed': 'No se pudo analizar el rostro. Intenta con una foto más nítida.',
  'MediaPipe tracking failed': 'No se pudo procesar el seguimiento de personas.',
  'Portal service temporarily unavailable': 'El servicio no está disponible en este momento.',
  'Monitor session limit reached; wait for session TTL': 'Hay otras sesiones de cámara activas. Cierra las otras pestañas y espera hasta 5 minutos antes de volver a intentarlo.',
};

let fallbackMonitorSession;
export function monitorSessionId() {
  try {
    const stored = sessionStorage.getItem('cams.monitorSession');
    if (stored && /^[A-Za-z0-9_-]{1,64}$/.test(stored)) return stored;
    const id = crypto.randomUUID();
    sessionStorage.setItem('cams.monitorSession', id);
    return id;
  } catch {
    return fallbackMonitorSession ??= crypto.randomUUID();
  }
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** @param {string} path
 * @param {{token?: string, method?: string, body?: unknown, signal?: AbortSignal}} options */
export async function request(
  path,
  { token, method = "GET", body, signal } = {},
) {
  const response = await fetch(`${API_PREFIX}${path}`, {
    method,
    signal,
    headers: {
      Accept: "application/json",
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
  const raw = await response.text();
  let data;
  try {
    data = raw ? JSON.parse(raw) : null;
  } catch {
    data = null;
  }
  if (!response.ok) {
    const detail = data?.detail ?? data?.message ?? data?.error;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((item) => item.msg ?? JSON.stringify(item)).join(". ")
          : `El servidor respondió con un error (${response.status}).`;
    throw new ApiError(messages[message] ?? message, response.status);
  }
  if (data === null)
    throw new ApiError(
      "El servidor no devolvió una respuesta JSON válida.",
      response.status,
    );
  return data;
}

export function asList(value, key) {
  if (Array.isArray(value)) return value;
  if (Array.isArray(value?.[key])) return value[key];
  throw new Error(`La respuesta de ${key} no tiene el formato esperado.`);
}

export function recognitionDelay(failures = 0) {
  return failures > 0 ? Math.min(15000, 1000 * 2 ** (failures - 1)) : 300;
}

export function errorMessage(error) {
  return error instanceof TypeError
    ? "No pudimos conectar con el servidor. Revisa tu conexión y vuelve a intentar."
    : error instanceof Error
      ? error.message
      : "Ocurrió un error inesperado.";
}
