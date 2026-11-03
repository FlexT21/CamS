import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";
import {
  Activity,
  ArrowDownLeft,
  ArrowRight,
  ArrowUpRight,
  Camera as CameraIcon,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Clock3,
  DoorOpen,
  Eye,
  EyeOff,
  Fingerprint,
  LayoutDashboard,
  LoaderCircle,
  LockKeyhole,
  LogOut,
  Menu,
  Plus,
  RefreshCw,
  ScanFace,
  ShieldCheck,
  Sparkles,
  UserPlus,
  Users,
  X,
} from "lucide-react";
import Camera, { type Api, type CameraNotice } from "./Camera";
import { ApiError, asList, errorMessage, request } from "./api";

type User = { name: string; email: string };
type Person = {
  id?: string | number;
  name?: string;
  full_name?: string;
  external_id?: string;
  created_at?: string | number;
  image?: string;
};
type Event = {
  id?: string | number;
  name?: string;
  user?: string | { name?: string; full_name?: string };
  person_name?: string;
  type?: string;
  direction?: string;
  event_type?: string;
  timestamp?: string | number;
  created_at?: string | number;
  message?: string;
};
type Stats = {
  entries: number;
  exits: number;
  inside: number;
  persons: number;
};
type ServiceStatus = {
  database?: { backend?: string; mode?: string; state?: string };
  face?: { state?: string };
  tracker?: { state?: string };
  mqtt?: string | { state?: string };
  serial?: string | { state?: string };
};
type Page = "monitor" | "persons";

function readToken() {
  try {
    return localStorage.getItem("cams.token") || "";
  } catch {
    return "";
  }
}
function saveToken(token: string) {
  try {
    if (token) localStorage.setItem("cams.token", token);
    else localStorage.removeItem("cams.token");
  } catch {
    /* Session still works in memory when storage is unavailable. */
  }
}
function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <div className="brand">
      <span className="brand-icon">
        <ScanFace size={25} strokeWidth={2.2} />
      </span>
      {!compact && (
        <span>
          Cam<span className="brand-s">S</span>
          <small>VISIÓN QUE CONECTA</small>
        </span>
      )}
    </div>
  );
}
function formatDate(value?: string | number) {
  if (value === undefined || value === null || value === "")
    return "Hora no disponible";
  const numeric =
    typeof value === "number"
      ? value
      : /^\d+(\.\d+)?$/.test(value)
        ? Number(value)
        : null;
  const date = new Date(numeric !== null ? numeric * 1000 : value);
  return Number.isNaN(date.getTime())
    ? "Hora no disponible"
    : new Intl.DateTimeFormat("es-CO", {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      }).format(date);
}
function eventName(event: Event) {
  return (
    event.person_name ||
    event.name ||
    (typeof event.user === "string"
      ? event.user
      : event.user?.name || event.user?.full_name) ||
    "Persona sin identificar"
  );
}
function eventType(event: Event) {
  const value = event.direction || event.type || event.event_type || "";
  return ["entry", "in", "entrada"].includes(value.toLowerCase())
    ? "Entrada"
    : ["exit", "out", "salida"].includes(value.toLowerCase())
      ? "Salida"
      : value || "Evento";
}

export default function App() {
  const [token, setToken] = useState(readToken);
  const [user, setUser] = useState<User | null>(null);
  const [checking, setChecking] = useState(!!token);
  const [sessionError, setSessionError] = useState("");
  const [screen, setScreen] = useState<"landing" | "login" | "signup">(
    "landing",
  );
  const [retry, setRetry] = useState(0);

  const logout = useCallback(() => {
    if (token) void request('/auth/logout', { token, method: 'POST', signal: AbortSignal.timeout(5000) }).catch(() => {});
    saveToken("");
    setToken("");
    setUser(null);
    setChecking(false);
    setScreen("login");
  }, [token]);
  const expire = useCallback(() => {
    logout();
    setSessionError("Tu sesión expiró. Inicia sesión de nuevo para continuar.");
  }, [logout]);
  const api: Api = useCallback(
    async (path, options = {}) => {
      try {
        return await request(path, {
          ...options,
          token,
          signal: options.signal ?? AbortSignal.timeout(20000),
        });
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) expire();
        throw err;
      }
    },
    [token, expire],
  );

  useEffect(() => {
    if (!token) return;
    const controller = new AbortController();
    let disposed = false;
    const timeout = setTimeout(() => controller.abort(), 20000);
    setChecking(true);
    void request("/me", { token, signal: controller.signal })
      .then((data) => {
        if (controller.signal.aborted) return;
        const current = data.user ?? data;
        if (
          typeof current.name !== "string" ||
          typeof current.email !== "string"
        )
          throw new Error(
            "La respuesta de sesión no contiene los datos del usuario.",
          );
        setUser(current);
        setSessionError("");
      })
      .catch((err) => {
        if (disposed) return;
        if (err instanceof ApiError && err.status === 401) expire();
        else
          setSessionError(
            controller.signal.aborted
              ? "No se pudo verificar la sesión a tiempo."
              : errorMessage(err),
          );
      })
      .finally(() => {
        clearTimeout(timeout);
        if (!disposed) setChecking(false);
      });
    return () => {
      disposed = true;
      clearTimeout(timeout);
      controller.abort();
    };
  }, [token, retry, expire]);

  function authenticated(data: { token: string; user: User }) {
    saveToken(data.token);
    setToken(data.token);
    setUser(data.user);
    setSessionError("");
  }

  if (checking)
    return (
      <div className="loading-page">
        <Logo />
        <LoaderCircle className="spin" size={28} />
        <p>Verificando tu sesión…</p>
      </div>
    );
  if (token && user)
    return (
      <Dashboard user={user} api={api} onLogout={logout} onExpired={expire} />
    );
  if (token && sessionError)
    return (
      <div className="loading-page">
        <Logo />
        <p className="error-text" role="alert">
          {sessionError}
        </p>
        <button
          className="button primary"
          onClick={() => setRetry((value) => value + 1)}
        >
          Volver a intentar
        </button>
        <button className="button subtle" onClick={logout}>
          Volver al inicio de sesión
        </button>
      </div>
    );
  if (screen === "landing") return <Landing onNavigate={setScreen} />;
  return (
    <Auth
      screen={screen}
      onNavigate={setScreen}
      onAuthenticated={authenticated}
      sessionError={sessionError}
    />
  );
}

function Landing({
  onNavigate,
}: {
  onNavigate: (screen: "landing" | "login" | "signup") => void;
}) {
  return (
    <div className="landing">
      <nav className="landing-nav">
        <a href="#" aria-label="CamS inicio">
          <Logo />
        </a>
        <div className="landing-links">
          <a href="#capacidades">Cómo funciona</a>
          <button className="button subtle" onClick={() => onNavigate("login")}>
            Iniciar sesión
          </button>
          <button
            className="button primary"
            onClick={() => onNavigate("signup")}
          >
            Crear cuenta <ArrowUpRight size={17} />
          </button>
        </div>
      </nav>
      <main className="hero">
        <div className="hero-copy">
          <span className="pill-label">
            <span className="tiny-dot" /> INTELIGENCIA EN CADA ACCESO
          </span>
          <h1>
            Tu espacio.
            <br />
            Tu gente.
            <br />
            <span>Todo bajo control.</span>
          </h1>
          <p>
            Transforma una cámara en una nueva forma de cuidar tu espacio.
            Registra personas y supervisa el acceso con visión artificial, desde
            un solo lugar.
          </p>
          <div className="hero-actions">
            <button
              className="button primary large"
              onClick={() => onNavigate("signup")}
            >
              Comenzar ahora <ArrowRight size={19} />
            </button>
            <button
              className="button outline large"
              onClick={() => onNavigate("login")}
            >
              Ya tengo una cuenta
            </button>
          </div>
          <div className="hero-trust">
            <ShieldCheck size={18} />
            Acceso autenticado <span />
            Tu cámara, tu control
          </div>
        </div>
        <div
          className="hero-visual"
          aria-label="Ilustración del sistema de reconocimiento de CamS"
        >
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="hero-visual-label">
            <span className="tiny-dot" /> VISIÓN ARTIFICIAL
          </div>
          <div className="identity-art">
            <div className="art-corners" />
            <ScanFace size={134} strokeWidth={0.8} />
            <div className="scan-line" />
            <span>UNA IDENTIDAD. UN ACCESO.</span>
          </div>
          <div className="floating-card floating-top">
            <span className="feature-icon">
              <Fingerprint size={24} />
            </span>
            <div>
              Reconoce lo que importa<small>Identidad en cada registro</small>
            </div>
            <ShieldCheck size={20} />
          </div>
          <div className="floating-card floating-bottom">
            <span className="feature-icon yellow">
              <DoorOpen size={23} />
            </span>
            <div>
              Conecta con tu espacio
              <small>Entradas y salidas, en un lugar</small>
            </div>
          </div>
          <div className="visual-caption">
            CamS <span>UNA VISIÓN MÁS CLARA</span>
          </div>
        </div>
      </main>
      <section className="features" id="capacidades">
        <div className="feature-intro">
          <span className="eyebrow">MENOS FRICCIÓN. MÁS VISIBILIDAD.</span>
          <h2>El acceso, simplificado.</h2>
        </div>
        <article>
          <UserPlus size={24} />
          <h3>Registra</h3>
          <p>
            Crea una identidad con un nombre, un identificador y una fotografía.
          </p>
        </article>
        <article>
          <ScanFace size={24} />
          <h3>Reconoce</h3>
          <p>
            Analiza la imagen de tu cámara y visualiza las detecciones en vivo.
          </p>
        </article>
        <article>
          <Activity size={24} />
          <h3>Supervisa</h3>
          <p>Consulta los eventos de acceso y los indicadores de tu espacio.</p>
        </article>
      </section>
      <footer className="landing-footer">
        <span>© {new Date().getFullYear()} CamS</span>
        <span>Una nueva perspectiva de seguridad.</span>
      </footer>
    </div>
  );
}

function Auth({
  screen,
  onNavigate,
  onAuthenticated,
  sessionError,
}: {
  screen: "login" | "signup";
  onNavigate: (screen: "landing" | "login" | "signup") => void;
  onAuthenticated: (data: { token: string; user: User }) => void;
  sessionError: string;
}) {
  const signup = screen === "signup";
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [visible, setVisible] = useState(false);
  const active = useRef<AbortController | null>(null);
  useEffect(() => {
    setError("");
    return () => active.current?.abort();
  }, [screen]);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy) return;
    const values = new FormData(event.currentTarget);
    const email = String(values.get("email")).trim();
    const password = String(values.get("password"));
    const name = String(values.get("name") ?? "").trim();
    if (signup && !name) {
      setError("Escribe tu nombre.");
      return;
    }
    setBusy(true);
    setError("");
    const controller = new AbortController();
    active.current = controller;
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const result = await request(signup ? "/auth/register" : "/auth/login", {
        method: "POST",
        body: { email, password, ...(signup ? { name } : {}) },
        signal: controller.signal,
      });
      const sessionToken = result.access_token ?? result.token;
      if (
        !sessionToken ||
        typeof sessionToken !== "string" ||
        typeof result.user?.name !== "string" ||
        typeof result.user?.email !== "string"
      )
        throw new Error(
          "La respuesta del servidor no contiene una sesión válida.",
        );
      onAuthenticated({ token: sessionToken, user: result.user });
    } catch (err) {
      setError(
        controller.signal.aborted
          ? "La solicitud tardó demasiado. Inténtalo de nuevo."
          : errorMessage(err),
      );
    } finally {
      clearTimeout(timeout);
      setBusy(false);
    }
  }
  return (
    <div className="auth-page">
      <aside className="auth-story">
        <button className="brand-button" onClick={() => onNavigate("landing")}>
          <Logo />
        </button>
        <div>
          <span className="pill-label">BIENVENIDO A UNA NUEVA VISIÓN</span>
          <h1>
            El lugar donde
            <br />
            tu seguridad
            <br />
            <em>toma forma.</em>
          </h1>
          <p>
            Personas, accesos y tranquilidad.
            <br />
            Todo conectado en una sola plataforma.
          </p>
          <div className="auth-art">
            <ScanFace size={120} strokeWidth={0.8} />
            <span className="auth-art-line" />
          </div>
        </div>
        <small>CamS · Visión que conecta</small>
      </aside>
      <main className="auth-main">
        <button className="back-link" onClick={() => onNavigate("landing")}>
          <ChevronLeft size={16} /> Volver al inicio
        </button>
        <div className="auth-form-wrap">
          <span className="section-icon">
            <LockKeyhole size={23} />
          </span>
          <span className="eyebrow">TU ESPACIO CAMS</span>
          <h2>
            {signup ? "Empieza a ver diferente." : "Qué bueno verte de nuevo."}
          </h2>
          <p>
            {signup
              ? "Crea tu cuenta para acceder al panel de control."
              : "Ingresa tus datos y continúa donde lo dejaste."}
          </p>
          <form onSubmit={submit} key={screen}>
            {signup && (
              <label>
                Nombre completo
                <input
                  name="name"
                  autoComplete="name"
                  placeholder="Tu nombre"
                  required
                  maxLength={120}
                  disabled={busy}
                />
              </label>
            )}
            <label>
              Correo electrónico
              <input
                type="email"
                name="email"
                autoComplete="email"
                placeholder="nombre@empresa.com"
                required
                maxLength={254}
                disabled={busy}
              />
            </label>
            <label>
              Contraseña
              <div className="password-field">
                <input
                  type={visible ? "text" : "password"}
                  name="password"
                  autoComplete={signup ? "new-password" : "current-password"}
                  placeholder={
                    signup ? "Mínimo 10 caracteres" : "Tu contraseña"
                  }
                  minLength={signup ? 10 : 1}
                  maxLength={128}
                  required
                  disabled={busy}
                />
                <button
                  type="button"
                  onClick={() => setVisible((value) => !value)}
                  aria-label={
                    visible ? "Ocultar contraseña" : "Mostrar contraseña"
                  }
                >
                  {visible ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
              {signup && (
                <small className="password-hint">
                  Usa entre 10 y 128 caracteres.
                </small>
              )}
            </label>
            {(error || sessionError) && (
              <div className="error-box" role="alert">
                {error || sessionError}
              </div>
            )}
            <button type="submit" className="button primary full" disabled={busy}>
              {busy ? <LoaderCircle size={18} className="spin" /> : null}
              {busy
                ? "Conectando…"
                : signup
                  ? "Crear mi cuenta"
                  : "Iniciar sesión"}
              {!busy && <ArrowRight size={18} />}
            </button>
          </form>
          <p className="auth-switch">
            {signup ? "¿Ya tienes una cuenta?" : "¿Primera vez en CamS?"}{" "}
            <button
              disabled={busy}
              onClick={() => onNavigate(signup ? "login" : "signup")}
            >
              {signup ? "Inicia sesión" : "Crea una cuenta"}
            </button>
          </p>
          {signup && (
            <small className="signup-note">
              Este formulario crea una cuenta de acceso a CamS.
            </small>
          )}
        </div>
        <span className="auth-bottom">
          <ShieldCheck size={15} /> Tu sesión está protegida con acceso
          autenticado.
        </span>
      </main>
    </div>
  );
}

function Dashboard({
  user,
  api,
  onLogout,
  onExpired,
}: {
  user: User;
  api: Api;
  onLogout: () => void;
  onExpired: () => void;
}) {
  const [page, setPage] = useState<Page>("persons");
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [persons, setPersons] = useState<Person[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [status, setStatus] = useState<string>("Consultando");
  const [services, setServices] = useState<ServiceStatus | null>(null);
  const [personOffset, setPersonOffset] = useState(0);
  const [personTotal, setPersonTotal] = useState(0);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const [updated, setUpdated] = useState<Date | null>(null);
  const [filter, setFilter] = useState("");
  const [enrollmentImage, setEnrollmentImage] = useState("");
  const [panelExpanded, setPanelExpanded] = useState(false);
  const [enrollmentNotice, setEnrollmentNotice] = useState<CameraNotice | null>(
    null,
  );
  useEffect(() => {
    setPanelExpanded(false);
    setEnrollmentNotice(null);
  }, [page]);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      const resources = ["persons", "events", "stats", "status"] as const;
      const signal = AbortSignal.any([
        controller.signal,
        AbortSignal.timeout(20000),
      ]);
      const results = await Promise.allSettled(
        resources.map((resource) =>
          api(
            resource === "persons"
              ? `/persons?limit=50&offset=${personOffset}`
              : `/${resource}`,
            { signal },
          ),
        ),
      );
      if (controller.signal.aborted) return;
      const failures: Record<string, string> = {};
      results.forEach((result, index) => {
        const resource = resources[index];
        try {
          if (result.status === "rejected") throw result.reason;
          if (resource === "persons") {
            setPersons(asList(result.value, "persons"));
            setPersonTotal(
              result.value.total ?? asList(result.value, "persons").length,
            );
          }
          if (resource === "events") setEvents(asList(result.value, "events"));
          if (resource === "stats") {
            if (
              !["entries", "exits", "inside", "persons"].every(
                (key) =>
                  typeof result.value?.[key] === "number" &&
                  Number.isFinite(result.value[key]),
              )
            )
              throw new Error("Los indicadores recibidos no son válidos.");
            setStats(result.value);
          }
          if (resource === "status") {
            const state: ServiceStatus = result.value;
            if (
              !state.database?.state ||
              !state.face?.state ||
              !state.tracker?.state
            )
              throw new Error(
                "No se pudo interpretar el estado de los servicios.",
              );
            setServices(state);
            setStatus(
              state.database.state === "connected"
                ? "Servicio conectado"
                : "Revisar base de datos",
            );
          }
        } catch (err) {
          failures[resource] = errorMessage(err);
          if (resource === "status") setStatus("Sin conexión");
        }
      });
      setErrors(failures);
      setLoading(false);
      if (results.some((result) => result.status === "fulfilled"))
        setUpdated(new Date());
      timer = setTimeout(() => void load(), 15000);
    };
    setLoading(true);
    void load();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [api, revision, personOffset]);

  const shownPersons = persons.filter((person) =>
    `${person.name ?? person.full_name ?? ""} ${person.external_id ?? ""}`
      .toLocaleLowerCase()
      .includes(filter.toLocaleLowerCase()),
  );
  return (
    <div className={`dashboard ${collapsed ? "sidebar-collapsed" : ""}`}>
      {mobileOpen && (
        <button
          className="sidebar-scrim"
          aria-label="Cerrar menú"
          onClick={() => setMobileOpen(false)}
        />
      )}
      <aside className={`sidebar ${mobileOpen ? "mobile-open" : ""}`}>
        <div className="sidebar-brand">
          <Logo compact={collapsed} />
          <button
            className="collapse-button"
            aria-label={
              collapsed ? "Expandir barra lateral" : "Contraer barra lateral"
            }
            onClick={() => setCollapsed((value) => !value)}
          >
            {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
          <button
            className="mobile-close icon-button"
            aria-label="Cerrar menú"
            onClick={() => setMobileOpen(false)}
          >
            <X size={20} />
          </button>
        </div>
        <div className="workspace-label">
          <span className="workspace-avatar">
            <LayoutDashboard size={18} />
          </span>
          <div>
            Mi espacio<small>Panel de administración</small>
          </div>
        </div>
        <span className="nav-label">PRINCIPAL</span>
        <nav className="side-nav">
          <button
            title="Registro de personas"
            className={page === "persons" ? "active" : ""}
            onClick={() => {
              setPage("persons");
              setMobileOpen(false);
            }}
          >
            <Users size={20} />
            <span>Módulo 1 · Registro de personas</span>
            {page === "persons" && <i />}
          </button>
          <button
            title="Control de acceso"
            className={page === "monitor" ? "active" : ""}
            onClick={() => {
              setPage("monitor");
              setMobileOpen(false);
            }}
          >
            <ScanFace size={20} />
            <span>Módulo 2 · Control de acceso</span>
            {page === "monitor" && <i />}
          </button>
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-tip">
            <span className="tip-icon">
              <Sparkles size={21} />
            </span>
            <h3>Una visión más clara.</h3>
            <p>Tu espacio y las personas que lo hacen posible.</p>
            <span>
              CON TECNOLOGÍA CAMS <ArrowUpRight size={14} />
            </span>
          </div>
          <div className="sidebar-status">
            <span className={`tiny-dot ${errors.status ? "warning" : ""}`} />
            <span>{status}</span>
          </div>
          <div className="user-block">
            <span className="avatar">
              {user.name.slice(0, 1).toUpperCase()}
            </span>
            <div>
              <strong>{user.name}</strong>
              <small title={user.email}>{user.email}</small>
            </div>
            <button
              title="Cerrar sesión"
              aria-label="Cerrar sesión"
              onClick={onLogout}
            >
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </aside>
      <div className="dashboard-main">
        <header className="topbar">
          <div>
            <button
              className="mobile-menu icon-button"
              aria-label="Abrir menú"
              onClick={() => setMobileOpen(true)}
            >
              <Menu size={22} />
            </button>
            <span className="breadcrumb">
              Mi espacio <ChevronRight size={14} />{" "}
              <strong>
                {page === "monitor"
                  ? "Control de acceso"
                  : "Registro de personas"}
              </strong>
            </span>
          </div>
          <div className="topbar-right">
            <span className="system-tag">
              <ShieldCheck size={15} />
              Acceso autenticado
            </span>
            <span className="avatar small">
              {user.name.slice(0, 1).toUpperCase()}
            </span>
          </div>
        </header>
        <main className="main-content">
          <div className="page-heading">
            <div>
              <span className="eyebrow">TU ESPACIO, EN PERSPECTIVA</span>
              <h1>
                {page === "monitor"
                  ? "Control de acceso"
                  : "Registro de personas"}
                <span className="heading-dot">.</span>
              </h1>
              <p>
                {page === "monitor"
                  ? "Una mirada en tiempo real a las personas que entran y salen."
                  : "Cada persona tiene una identidad. Conecta la suya con tu espacio."}
              </p>
            </div>
            <button
              className="button white"
              disabled={loading}
              onClick={refresh}
            >
              <RefreshCw size={16} className={loading ? "spin" : ""} />
              Actualizar
            </button>
          </div>
          {Object.entries(errors).length > 0 && (
            <div className="error-box dashboard-errors" role="alert">
              {Object.entries(errors).map(([key, value]) => (
                <p key={key}>
                  <strong>
                    {
                      {
                        persons: "Personas",
                        events: "Eventos",
                        stats: "Indicadores",
                        status: "Servicio",
                      }[key]
                    }
                    :
                  </strong>{" "}
                  {value}
                </p>
              ))}
              <span>
                Los datos anteriores pueden estar desactualizados. Usa
                Actualizar para reintentar.
              </span>
            </div>
          )}
          <div className="stats-grid">
            {[
              {
                title: "Entradas",
                value: stats?.entries,
                icon: ArrowDownLeft,
                color: "green",
                caption: "Ingresos registrados",
              },
              {
                title: "Salidas",
                value: stats?.exits,
                icon: ArrowUpRight,
                color: "blue",
                caption: "Salidas registradas",
              },
              {
                title: "Personas dentro",
                value: stats?.inside,
                icon: Users,
                color: "yellow",
                caption: "Ocupación actual",
              },
              {
                title: "Personas registradas",
                value: stats?.persons,
                icon: Fingerprint,
                color: "purple",
                caption: "Identidades en tu espacio",
              },
            ].map((item) => (
              <article className={`stat-card ${item.color}`} key={item.title}>
                <div>
                  <span>{item.title}</span>
                  <span className="stat-icon">
                    <item.icon size={20} />
                  </span>
                </div>
                <strong>{item.value ?? "—"}</strong>
                <small>{item.caption}</small>
              </article>
            ))}
          </div>
          <div
            className={`section-grid ${panelExpanded ? "panel-expanded" : ""}`}
          >
            <Camera
              key={page}
              mode={page === "persons" ? "enroll" : "monitor"}
              api={api}
              expanded={panelExpanded}
              onToggleSize={() => setPanelExpanded((value) => !value)}
              externalNotice={page === "persons" ? enrollmentNotice : null}
              onCapture={(image) => {
                setEnrollmentImage(image);
                setEnrollmentNotice({
                  text: "Fotografía capturada. Completa los datos para guardar el registro.",
                });
              }}
              onEvents={refresh}
              onExpired={onExpired}
            />
            {page === "persons" ? (
              <Enrollment
                api={api}
                image={enrollmentImage}
                onClear={() => {
                  setEnrollmentImage("");
                  setEnrollmentNotice(null);
                }}
                onSaved={refresh}
                onNotice={setEnrollmentNotice}
              />
            ) : (
              <section className="activity-card">
                <div className="card-heading">
                  <div>
                    <span className="eyebrow">LO QUE ESTÁ PASANDO</span>
                    <h2>Actividad reciente</h2>
                  </div>
                  <span className="count-pill">{events.length}</span>
                </div>
                <div className="activity-list">
                  {loading && !events.length ? (
                    <Empty
                      icon="clock"
                      title="Consultando actividad"
                      text="Conectando con tu espacio…"
                    />
                  ) : !events.length ? (
                    <Empty
                      icon="clock"
                      title={
                        errors.events
                          ? "Actividad no disponible"
                          : "Tu próximo acceso empieza aquí"
                      }
                      text={
                        errors.events
                          ? "No pudimos consultar los eventos del servidor."
                          : "Los eventos aparecerán cuando el sistema registre una entrada o una salida."
                      }
                    />
                  ) : (
                    events.slice(0, 6).map((event, index) => (
                      <div className="activity-item" key={event.id ?? index}>
                        <span
                          className={`event-icon ${eventType(event) === "Entrada" ? "entry" : "exit"}`}
                        >
                          {eventType(event) === "Entrada" ? (
                            <ArrowDownLeft size={19} />
                          ) : (
                            <ArrowUpRight size={19} />
                          )}
                        </span>
                        <div>
                          <strong>{eventName(event)}</strong>
                          <small>
                            {formatDate(event.timestamp || event.created_at)}
                          </small>
                          {event.message && <small>{event.message}</small>}
                        </div>
                        <span
                          className={`event-badge ${eventType(event) === "Entrada" ? "entry" : ""}`}
                        >
                          {eventType(event)}
                        </span>
                      </div>
                    ))
                  )}
                </div>
                <div className="activity-footer">
                  <Clock3 size={14} />
                  Actualización automática cada 15 segundos
                </div>
              </section>
            )}
          </div>
          {page === "persons" ? (
            <section className="table-card">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">TU COMUNIDAD</span>
                  <h2>
                    Personas registradas{" "}
                    <span className="count-pill">{personTotal}</span>
                  </h2>
                </div>
                <input
                  type="search"
                  className="search-input"
                  aria-label="Buscar en esta página por nombre o identificador"
                  placeholder="Buscar en esta página…"
                  value={filter}
                  onChange={(event) => setFilter(event.target.value)}
                />
              </div>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Persona</th>
                      <th>Identificador</th>
                      <th>Fecha de registro</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shownPersons.map((person, index) => (
                      <tr key={person.id ?? index}>
                        <td>
                          <span className="person-cell">
                            {person.image?.startsWith("data:image/") ? (
                              <img
                                className="person-photo"
                                src={person.image}
                                alt=""
                              />
                            ) : (
                              <span className="avatar light">
                                {(person.name || person.full_name || "?")
                                  .slice(0, 1)
                                  .toUpperCase()}
                              </span>
                            )}
                            <strong>
                              {person.name ||
                                person.full_name ||
                                "Nombre no disponible"}
                            </strong>
                          </span>
                        </td>
                        <td>
                          <span className="id-tag">
                            {person.external_id || "—"}
                          </span>
                        </td>
                        <td>{formatDate(person.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {!shownPersons.length && (
                <Empty
                  icon="users"
                  title={
                    filter
                      ? "Sin coincidencias en esta página"
                      : errors.persons
                        ? "No se pudo cargar el registro"
                        : "Aún no hay personas registradas"
                  }
                  text={
                    filter
                      ? "Prueba con otro nombre o identificador."
                      : "Captura una fotografía y completa el formulario para crear un registro."
                  }
                />
              )}
              {personTotal > 50 && (
                <div className="pagination">
                  <span>
                    Página {Math.floor(personOffset / 50) + 1} de{" "}
                    {Math.ceil(personTotal / 50)} · {personTotal} personas
                  </span>
                  <div>
                    <button
                      className="button white"
                      disabled={!personOffset || loading}
                      onClick={() => {
                        setFilter("");
                        setPersonOffset((value) => Math.max(0, value - 50));
                      }}
                    >
                      <ChevronLeft size={14} />
                      Anterior
                    </button>
                    <button
                      className="button white"
                      disabled={personOffset + 50 >= personTotal || loading}
                      onClick={() => {
                        setFilter("");
                        setPersonOffset((value) => value + 50);
                      }}
                    >
                      Siguiente
                      <ChevronRight size={14} />
                    </button>
                  </div>
                </div>
              )}
            </section>
          ) : (
            <div className="insight-banner">
              <span className="insight-icon">
                <ShieldCheck size={24} />
              </span>
              <div>
                <h3>Más contexto. Mejores decisiones.</h3>
                <p>
                  Registra a las personas de tu espacio para identificar sus
                  accesos en la cámara.
                </p>
              </div>
              <button
                className="text-button"
                onClick={() => setPage("persons")}
              >
                Registrar una persona <ArrowRight size={17} />
              </button>
            </div>
          )}
          <ServiceIndicators services={services} stale={!!errors.status} />
          <footer className="dashboard-footer">
            <span>
              CamS <span>·</span> Visión que conecta
            </span>
            <span>
              {updated
                ? `Última consulta: ${updated.toLocaleTimeString("es-CO")}`
                : "Esperando conexión con el servidor"}
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}

function Empty({
  icon,
  title,
  text,
}: {
  icon: "clock" | "users";
  title: string;
  text: string;
}) {
  return (
    <div className="empty-state">
      <span>
        {icon === "clock" ? (
          <Activity size={26} strokeWidth={1.5} />
        ) : (
          <Users size={26} strokeWidth={1.5} />
        )}
      </span>
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

function ServiceIndicators({
  services,
  stale,
}: {
  services: ServiceStatus | null;
  stale: boolean;
}) {
  const labels: Record<string, string> = {
    connected: "Conectada",
    ready: "Listo",
    available: "Disponible",
    not_loaded: "En espera",
    missing_model: "Falta el modelo",
    missing_models: "Faltan modelos",
    missing_dependency: "Falta una dependencia",
    disconnected: "Desconectado",
    unverified: "Puerto abierto · Arduino sin confirmar",
    disabled: "Desactivado",
    error: "Error",
  };
  const stateOf = (value: ServiceStatus["mqtt"]) =>
    typeof value === "string" ? value : value?.state;
  const items = [
    {
      name: "Base de datos",
      state: services?.database?.state,
      extra: services?.database
        ? `${services.database.backend ?? ""} · ${{ local: "local", remote: "remota" }[services.database.mode || ""] || services.database.mode || ""}`
        : "",
    },
    { name: "Reconocimiento facial", state: services?.face?.state },
    { name: "Detección", state: services?.tracker?.state },
    { name: "MQTT", state: stateOf(services?.mqtt) },
    { name: "Serial", state: stateOf(services?.serial) },
  ];
  return (
    <section
      className="service-indicators"
      aria-label="Estado de los servicios"
    >
      <span className="service-label">
        <Activity size={13} />
        Servicios {stale && "· Sin actualizar"}
      </span>
      <div>
        {items.map((item) => (
          <span
            key={item.name}
            className="service-item"
            title={item.extra || undefined}
          >
            <i
              className={
                stale ||
                !item.state ||
                ["disabled", "not_loaded"].includes(item.state)
                  ? "neutral"
                  : ["connected", "ready", "available"].includes(item.state)
                    ? "healthy"
                    : "attention"
              }
            />
            <span>
              {item.name}
              <small>
                {stale
                  ? "Sin verificar"
                  : item.state
                    ? labels[item.state] || item.state
                    : "Sin información"}
              </small>
            </span>
          </span>
        ))}
      </div>
    </section>
  );
}

function Enrollment({
  api,
  image,
  onClear,
  onSaved,
  onNotice,
}: {
  api: Api;
  image: string;
  onClear: () => void;
  onSaved: () => void;
  onNotice: (notice: CameraNotice) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const controller = useRef<AbortController | null>(null);
  const saving = useRef(false);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    if (image) setSuccess("");
  }, [image]);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (saving.current) {
      onNotice({
        text: "Ya hay un registro en curso. Espera la respuesta antes de volver a enviarlo.",
      });
      return;
    }
    setError("");
    setSuccess("");
    if (!image) {
      setError("Captura una fotografía antes de guardar el registro.");
      onNotice({
        text: "Captura una fotografía antes de guardar el registro.",
        error: true,
      });
      return;
    }
    const form = event.currentTarget;
    const values = new FormData(form);
    const name = String(values.get("full_name") || "").trim();
    const external_id = String(values.get("external_id") || "").trim();
    if (!name || !external_id) {
      setError("Completa el nombre y el identificador.");
      onNotice({ text: "Completa el nombre y el identificador.", error: true });
      return;
    }
    saving.current = true;
    setBusy(true);
    onNotice({
      text: "Guardando registro. Espera la confirmación del servidor…",
    });
    controller.current = new AbortController();
    const timeout = setTimeout(() => controller.current?.abort(), 20000);
    try {
      await api("/persons", {
        method: "POST",
        body: { name, external_id, image },
        signal: controller.current.signal,
      });
      form.reset();
      onClear();
      setSuccess(`${name} se registró correctamente.`);
      onNotice({ text: `${name} se registró correctamente.` });
      onSaved();
    } catch (err) {
      const message = controller.current?.signal.aborted
        ? "El registro tardó demasiado. Verifica la lista antes de volver a enviarlo."
        : (err as { status?: number }).status === 409
          ? `No se pudo crear el registro: ${errorMessage(err)} Revisa el identificador antes de volver a enviarlo.`
          : errorMessage(err);
      setError(message);
      onNotice({ text: message, error: true });
    } finally {
      clearTimeout(timeout);
      saving.current = false;
      setBusy(false);
    }
  }
  return (
    <section className="enrollment-card">
      <div className="card-heading">
        <div>
          <span className="eyebrow">UNA NUEVA IDENTIDAD</span>
          <h2>Datos de la persona</h2>
        </div>
        <span className="section-icon mini">
          <UserPlus size={20} />
        </span>
      </div>
      <form onSubmit={submit}>
        <div className={`photo-preview ${image ? "captured" : ""}`}>
          {image ? (
            <>
              <img src={image} alt="Fotografía capturada para el registro" />
              <span>
                <Check size={14} />
                Fotografía lista
              </span>
              <button
                type="button"
                aria-label="Descartar fotografía"
                onClick={onClear}
                disabled={busy}
              >
                <X size={15} />
              </button>
            </>
          ) : (
            <>
              <span className="photo-icon">
                <CameraIcon size={24} />
              </span>
              <div>
                <strong>Tu fotografía aparecerá aquí</strong>
                <small>Activa la cámara y captura el rostro.</small>
              </div>
            </>
          )}
        </div>
        <label>
          Nombre completo <span>*</span>
          <input
            name="full_name"
            autoComplete="off"
            placeholder="Ej. María Fernanda López"
            required
            maxLength={120}
            disabled={busy}
          />
        </label>
        <label>
          Identificador externo <span>*</span>
          <input
            name="external_id"
            autoComplete="off"
            placeholder="Ej. documento o código de empleado"
            required
            maxLength={120}
            disabled={busy}
          />
          <small>Usa un identificador único para cada persona.</small>
        </label>
        {error && (
          <div className="error-box" role="alert">
            {error}
          </div>
        )}
        {success && (
          <div className="success-box" role="status">
            <CheckCircle2 size={17} />
            {success}
          </div>
        )}
        <button
          type="submit"
          className="button primary full"
          disabled={busy || !image}
        >
          {busy ? (
            <LoaderCircle className="spin" size={17} />
          ) : (
            <Plus size={18} />
          )}
          {busy ? "Guardando registro…" : "Registrar persona"}
        </button>
        <p className="form-note">
          <LockKeyhole size={13} />
          La fotografía se envía al guardar el registro.
        </p>
      </form>
    </section>
  );
}
