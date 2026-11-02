import { useEffect, useRef, useState } from "react";
import {
  Camera as CameraIcon,
  ScanFace,
  Square,
  Video,
  VideoOff,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { errorMessage, monitorSessionId, recognitionDelay } from "./api";

export type Track = {
  id: string | number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  center_x: number;
  user: unknown;
  distance_to_line: number;
  face?: {
    x: number;
    y: number;
    width: number;
    height: number;
    points: { x: number; y: number }[];
    mesh_points?: { x: number; y: number }[];
    mesh_connections?: [number, number][];
  } | null;
};
export type Frame = {
  tracks: Track[];
  events: unknown[];
  message: string;
  width: number;
  height: number;
  line_x: number;
};
export type Api = (
  path: string,
  options?: { method?: string; body?: unknown; signal?: AbortSignal },
) => Promise<any>;
export type CameraNotice = { text: string; error?: boolean };

function label(user: unknown) {
  if (typeof user === "string") return user;
  if (user && typeof user === "object") {
    const item = user as Record<string, unknown>;
    return String(item.name ?? item.full_name ?? "Persona identificada");
  }
  return "Sin identificar";
}

function FaceGuide({ face, recognized }: { face: NonNullable<Track["face"]>; recognized: boolean }) {
  const [rightEye, leftEye, nose, rightMouth, leftMouth] = face.points;
  if (!rightEye || !leftEye || !nose || !rightMouth || !leftMouth) return null;
  const color = recognized ? "#65e3ac" : "#ffffff";
  const top = Math.max(0, face.y - face.height * 0.15);
  const bottom = face.y + face.height * 1.08;
  const left = Math.max(0, face.x - face.width * 0.1);
  const right = face.x + face.width * 1.1;
  return (
    <g className="face-landmarks" aria-label="Puntos faciales detectados">
      {face.mesh_points && face.mesh_connections && (
        <path fill="none" stroke={color} strokeWidth="0.6" opacity="0.55"
          d={face.mesh_connections.map(([a, b]) => {
            const p = face.mesh_points![a], q = face.mesh_points![b];
            return p && q ? `M ${p.x} ${p.y} L ${q.x} ${q.y}` : "";
          }).join(" ")} />
      )}
      <path
        d={`M ${face.x + face.width / 2} ${top} C ${right} ${top}, ${right} ${bottom}, ${face.x + face.width / 2} ${bottom} C ${left} ${bottom}, ${left} ${top}, ${face.x + face.width / 2} ${top}`}
        fill="rgba(17,29,49,.10)"
        stroke={color}
        strokeWidth="2"
      />
      <path d={`M ${rightEye.x} ${rightEye.y} L ${nose.x} ${nose.y} L ${leftEye.x} ${leftEye.y} M ${rightMouth.x} ${rightMouth.y} L ${nose.x} ${nose.y} L ${leftMouth.x} ${leftMouth.y} M ${rightEye.x} ${rightEye.y} L ${leftEye.x} ${leftEye.y} M ${rightMouth.x} ${rightMouth.y} L ${leftMouth.x} ${leftMouth.y}`} fill="none" stroke={color} strokeWidth="1.5" opacity=".9" />
      {face.points.map((point, index) => <circle key={index} cx={point.x} cy={point.y} r="3.5" fill={index === 2 ? "#f6c644" : color} />)}
    </g>
  );
}

export default function Camera({
  mode,
  api,
  onCapture,
  onEvents,
  onExpired,
  expanded,
  onToggleSize,
  externalNotice,
}: {
  mode: "enroll" | "monitor";
  api: Api;
  onCapture?: (image: string) => void;
  onEvents: () => void;
  onExpired: () => void;
  expanded: boolean;
  onToggleSize: () => void;
  externalNotice?: CameraNotice | null;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const generation = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const pending = useRef<AbortController | null>(null);
  const [ready, setReady] = useState(false);
  const [opening, setOpening] = useState(false);
  const [running, setRunning] = useState(false);
  const [notice, setNotice] = useState("Activa la cámara para comenzar.");
  const [error, setError] = useState("");
  const [frame, setFrame] = useState<Frame | null>(null);
  const callbacks = useRef({ onEvents, onExpired });
  callbacks.current = { onEvents, onExpired };

  function stopCycle() {
    generation.current++;
    clearTimeout(timer.current);
    pending.current?.abort();
    pending.current = null;
  }

  function stop() {
    stopCycle();
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    setReady(false);
    setOpening(false);
    setRunning(false);
    setFrame(null);
    setNotice("Cámara detenida. Puedes activarla cuando lo necesites.");
  }

  useEffect(
    () => () => {
      stopCycle();
      stream.current?.getTracks().forEach((track) => track.stop());
    },
    [],
  );

  async function start() {
    if (stream.current || opening) return;
    const current = ++generation.current;
    setOpening(true);
    setError("");
    try {
      if (!navigator.mediaDevices?.getUserMedia)
        throw new Error(
          "La cámara necesita HTTPS o localhost y un navegador compatible.",
        );
      const media = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });
      if (current !== generation.current || !video.current) {
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      stream.current = media;
      media.getVideoTracks().forEach((track) => {
        track.onended = () => {
          stop();
          setError("La cámara se desconectó o su permiso fue revocado.");
        };
      });
      video.current.srcObject = media;
      await video.current.play();
      if (current !== generation.current) return;
      setReady(true);
      setNotice(
        mode === "enroll"
          ? "Ubica el rostro al centro y captura una foto nítida."
          : "Cámara lista. Inicia el reconocimiento para analizar el acceso.",
      );
    } catch (err) {
      if (current !== generation.current) return;
      stream.current?.getTracks().forEach((track) => track.stop());
      stream.current = null;
      setError(
        err instanceof DOMException && err.name === "NotAllowedError"
          ? "Permiso de cámara denegado. Habilítalo en la configuración del navegador."
          : errorMessage(err),
      );
    } finally {
      if (current === generation.current) setOpening(false);
    }
  }

  function snapshot() {
    const source = video.current;
    if (!source || !source.videoWidth || source.readyState < 2)
      throw new Error(
        "La cámara aún no tiene una imagen disponible. Intenta de nuevo.",
      );
    const canvas = document.createElement("canvas");
    const scale = Math.min(1, 1280 / source.videoWidth);
    canvas.width = Math.round(source.videoWidth * scale);
    canvas.height = Math.round(source.videoHeight * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("No se pudo preparar la captura de cámara.");
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  }

  function capture() {
    try {
      onCapture?.(snapshot());
      setError("");
      setNotice(
        "Fotografía capturada. Completa los datos para guardar el registro.",
      );
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  function recognize() {
    if (!ready || running) return;
    stopCycle();
    const current = generation.current;
    const sessionId = monitorSessionId();
    let failures = 0;
    setRunning(true);
    setError("");
    setNotice("Analizando el acceso…");
    const tick = async () => {
      if (current !== generation.current) return;
      const controller = new AbortController();
      pending.current = controller;
      const timeout = setTimeout(() => controller.abort(), 20000);
      try {
        const result: Frame = await api("/monitor/frame", {
          method: "POST",
          body: { image: snapshot(), session_id: sessionId },
          signal: controller.signal,
        });
        if (current !== generation.current) return;
        if (
          !Array.isArray(result.tracks) ||
          !Array.isArray(result.events) ||
          !(result.width > 0) ||
          !(result.height > 0) ||
          !(result.line_x > 0)
        )
          throw new Error(
            "El servidor devolvió un análisis de cámara inválido.",
          );
        failures = 0;
        setError("");
        setFrame(result);
        const recognized = [...new Set(result.tracks.filter(track => !!track.user).map(track => label(track.user)))];
        const crossing = result.events[0] as {direction?: string; person_name?: string} | undefined;
        setNotice(crossing
          ? `${crossing.person_name || "Persona sin identificar"}: ${crossing.direction === "entrada" ? "entrada" : "salida"} registrada.`
          : recognized.length
            ? `¡Fuiste reconocido! ${recognized.join(", ")}.`
            : result.tracks.length
              ? "Persona detectada. Buscando un rostro registrado…"
              : "Supervisando el acceso. Esperando personas frente a la cámara.");
        if (result.events.length) callbacks.current.onEvents();
      } catch (err) {
        if (current !== generation.current) return;
        if ((err as { status?: number }).status === 503) {
          failures++;
          setFrame(null);
          setError(
            `${errorMessage(err)} Reintentando en ${recognitionDelay(failures) / 1000} s.`,
          );
          timer.current = setTimeout(tick, recognitionDelay(failures));
          return;
        }
        setError(
          controller.signal.aborted
            ? "El análisis tardó demasiado. Puedes volver a iniciarlo."
            : errorMessage(err),
        );
        setRunning(false);
        setFrame(null);
        if ((err as { status?: number }).status === 401)
          callbacks.current.onExpired();
        return;
      } finally {
        clearTimeout(timeout);
        if (pending.current === controller) pending.current = null;
      }
      if (current === generation.current)
        timer.current = setTimeout(tick, recognitionDelay());
    };
    void tick();
  }

  return (
    <section className="camera-card">
      <div className="card-heading">
        <div>
          <span className="eyebrow">CÁMARA EN VIVO</span>
          <h2>
            {mode === "enroll"
              ? "Una cara, una identidad"
              : "Tu punto de acceso"}
          </h2>
        </div>
        <div className="camera-heading-actions">
          <span className={`live-pill ${ready ? "online" : ""}`}>
            <i />
            {running ? "Analizando" : ready ? "En vivo" : "Desconectada"}
          </span>
          <button
            className="panel-toggle"
            onClick={onToggleSize}
            aria-expanded={expanded}
            aria-controls="camera-viewfinder"
          >
            {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
            {expanded ? "Reducir panel" : "Ampliar panel"}
          </button>
        </div>
      </div>
      <div
        className={`camera-notice ${error || externalNotice?.error ? "has-error" : ""}`}
        role={error || externalNotice?.error ? "alert" : "status"}
        aria-live="polite"
      >
        <ScanFace size={17} />
        <span>{error || externalNotice?.text || notice}</span>
      </div>
      <div className="viewfinder" id="camera-viewfinder">
        <video
          ref={video}
          muted
          playsInline
          aria-label="Vista de la cámara de acceso"
        />
        {!ready && (
          <div className="camera-empty">
            <div className="camera-symbol">
              <Video size={34} strokeWidth={1.5} />
            </div>
            <h3>Todo empieza con una mirada</h3>
            <p>
              Activa tu cámara para{" "}
              {mode === "enroll"
                ? "registrar una nueva persona"
                : "supervisar entradas y salidas"}
              .
            </p>
            <span>Tu cámara solo se utiliza mientras está activa.</span>
          </div>
        )}
        {ready && mode === "enroll" && (
          <div className="face-guide">
            <span />
            <span />
            <span />
            <span />
          </div>
        )}
        {ready && mode === "monitor" && (
          <svg
            className="tracking-overlay"
            viewBox={`0 0 ${frame?.width || video.current?.videoWidth || 1280} ${frame?.height || video.current?.videoHeight || 720}`}
            preserveAspectRatio="xMidYMid meet"
            aria-label="Detecciones, centroides y línea de acceso"
          >
            <line
              x1={frame?.line_x || (frame?.width || video.current?.videoWidth || 1280) * 0.25}
              x2={frame?.line_x || (frame?.width || video.current?.videoWidth || 1280) * 0.25}
              y1="0"
              y2={frame?.height || video.current?.videoHeight || 720}
              stroke="#f6c644"
              strokeWidth="3"
              strokeDasharray="12 10"
            />
            {frame?.tracks.map((track) => (
              <g key={track.id}>
                <rect
                  x={track.x1}
                  y={track.y1}
                  width={Math.max(0, track.x2 - track.x1)}
                  height={Math.max(0, track.y2 - track.y1)}
                  fill="none"
                  stroke={track.user ? "#65e3ac" : "#f6c644"}
                  strokeWidth="3"
                />
                <circle
                  cx={track.center_x}
                  cy={(track.y1 + track.y2) / 2}
                  r="5"
                  fill="#f6c644"
                />
                {track.face && <FaceGuide face={track.face} recognized={!!track.user} />}
                <rect
                  x={track.x1}
                  y={Math.max(0, track.y1 - 29)}
                  width={Math.min(
                    frame.width - track.x1,
                    Math.max(150, label(track.user).length * 10 + 25),
                  )}
                  height="27"
                  rx="4"
                  fill="#111d31"
                />
                <text
                  x={track.x1 + 8}
                  y={Math.max(19, track.y1 - 10)}
                  fill="white"
                  fontSize="17"
                >
                  {label(track.user)}
                </text>
              </g>
            ))}
          </svg>
        )}
        <div className="frame-corner tl" />
        <div className="frame-corner br" />
        <span className="camera-watermark">
          CamS <span>VISION</span>
        </span>
      </div>
      <div className="camera-actions">
        {!ready ? (
          <button
            className="button primary"
            disabled={opening}
            onClick={() => void start()}
          >
            <Video size={17} />
            {opening ? "Conectando…" : "Activar cámara"}
          </button>
        ) : (
          <>
            {mode === "enroll" ? (
              <button className="button primary" onClick={capture}>
                <CameraIcon size={18} />
                Capturar fotografía
              </button>
            ) : running ? (
              <button
                className="button primary"
                onClick={() => {
                  stopCycle();
                  setRunning(false);
                  setFrame(null);
                  setNotice("Reconocimiento pausado. La cámara sigue activa.");
                }}
              >
                <Square size={15} />
                Pausar reconocimiento
              </button>
            ) : (
              <button className="button primary" onClick={recognize}>
                <ScanFace size={18} />
                Iniciar reconocimiento
              </button>
            )}
            <button className="button subtle" onClick={stop}>
              <VideoOff size={17} />
              Detener cámara
            </button>
          </>
        )}
      </div>
      <div className="camera-footnote">
        <span className="tiny-dot" />
        {mode === "enroll"
          ? "Buena iluminación, sin accesorios que cubran el rostro."
          : "Izquierda → derecha: ENTRADA (+1) · Derecha → izquierda: SALIDA (−1)"}
      </div>
    </section>
  );
}
