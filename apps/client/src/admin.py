import asyncio
import hashlib
import hmac
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

import cv2

from src.main import main


_CREDENTIAL_SALT = bytes.fromhex("11a52af5f407b08b30fff0ecff016b11")
_CREDENTIAL_ITERATIONS = 600_000
_ADMIN_USERNAME_HASH = (
    "2e30d0d457d8686232a817ca0ad944d68c7783fb1d014f3a081ca3321930a5ba"
)
_ADMIN_PASSWORD_HASH = (
    "842d98809003d3c826f341269b37877bd468ca842ba461a15d723419a9f8dff7"
)


def _credential_hash(value: str) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        value.encode("utf-8"),
        _CREDENTIAL_SALT,
        _CREDENTIAL_ITERATIONS,
    )


class AdminInterface:
    """Desktop control panel for the client features."""

    def __init__(self, default_server: str, default_camera: str = "0") -> None:
        self.root = tk.Tk()
        self.root.title("CamS - Inicio de sesión")
        self.root.geometry("760x540")
        self.root.minsize(680, 480)

        self.server = tk.StringVar(value=default_server)
        self.camera = tk.StringVar()
        self.camera_devices: dict[str, int] = {}
        self.default_camera = default_camera
        self.photos = tk.StringVar(value="3")
        self.username = tk.StringVar()
        self.status = tk.StringVar(value="Listo para iniciar una operación.")
        self.operation_active = False
        self.login_username = tk.StringVar()
        self.login_password = tk.StringVar()

        self._build_login()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_login(self) -> None:
        container = ttk.Frame(self.root, padding=20)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(1, weight=1)

        ttk.Label(
            container, text="CamS - Inicio de sesión", font=("Segoe UI", 18, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=(20, 25))
        ttk.Label(container, text="Usuario:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        username_entry = ttk.Entry(container, textvariable=self.login_username)
        username_entry.grid(row=1, column=1, sticky=tk.EW, pady=5)
        ttk.Label(container, text="Contraseña:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        password_entry = ttk.Entry(
            container, textvariable=self.login_password, show="*"
        )
        password_entry.grid(row=2, column=1, sticky=tk.EW, pady=5)
        ttk.Button(
            container, text="Iniciar sesión", command=self._authenticate
        ).grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(20, 5))
        ttk.Label(
            container,
            text="Acceso protegido con credenciales almacenadas como hashes.",
            foreground="#666666",
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(12, 0))
        username_entry.focus_set()
        password_entry.bind("<Return>", lambda _event: self._authenticate())

    def _authenticate(self) -> None:
        username_valid = hmac.compare_digest(
            _credential_hash(self.login_username.get()),
            bytes.fromhex(_ADMIN_USERNAME_HASH),
        )
        password_valid = hmac.compare_digest(
            _credential_hash(self.login_password.get()),
            bytes.fromhex(_ADMIN_PASSWORD_HASH),
        )
        if not username_valid or not password_valid:
            self.login_password.set("")
            messagebox.showerror(
                "Acceso denegado",
                "El usuario o la contraseña no son válidos.",
                parent=self.root,
            )
            return

        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.title("CamS - Administración")
        self._build_dashboard()
        self._refresh_cameras()

    def _build_dashboard(self) -> None:
        container = ttk.Frame(self.root, padding=20)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(1, weight=1)

        ttk.Label(
            container, text="Administración CamS", font=("Segoe UI", 18, "bold")
        ).grid(row=0, column=0, columnspan=3, pady=(0, 18))
        ttk.Label(container, text="Cámara:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.camera_selector = ttk.Combobox(
            container, textvariable=self.camera, state="readonly"
        )
        self.camera_selector.grid(row=1, column=1, sticky=tk.EW, pady=5)
        ttk.Button(
            container, text="Actualizar", command=self._refresh_cameras
        ).grid(row=1, column=2, padx=(8, 0), pady=5)
        ttk.Label(container, text="Servidor WebSocket:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(container, textvariable=self.server).grid(
            row=2, column=1, columnspan=2, sticky=tk.EW, pady=5
        )

        registration = ttk.LabelFrame(
            container, text="Registro de usuario", padding=10
        )
        registration.grid(
            row=3, column=0, columnspan=3, sticky=tk.EW, pady=(18, 10)
        )
        registration.columnconfigure(1, weight=1)
        ttk.Label(registration, text="Nombre:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(registration, textvariable=self.username).grid(
            row=0, column=1, sticky=tk.EW, padx=8
        )
        ttk.Label(registration, text="Fotos:").grid(row=0, column=2, sticky=tk.W)
        ttk.Spinbox(
            registration, from_=1, to=20, textvariable=self.photos, width=5
        ).grid(row=0, column=3, sticky=tk.W, padx=(8, 0))
        self.register_button = ttk.Button(
            registration, text="Registrar usuario", command=self._register
        )
        self.register_button.grid(
            row=1, column=0, columnspan=4, sticky=tk.EW, pady=(12, 0)
        )

        self.recognize_button = ttk.Button(
            container, text="Iniciar reconocimiento", command=self._recognize
        )
        self.recognize_button.grid(
            row=4, column=0, columnspan=3, sticky=tk.EW, pady=5
        )
        ttk.Separator(container).grid(
            row=5, column=0, columnspan=3, sticky=tk.EW, pady=12
        )
        ttk.Label(container, textvariable=self.status, wraplength=450).grid(
            row=6, column=0, columnspan=3, sticky=tk.W
        )
        ttk.Label(
            container,
            text="La ventana de cámara se cierra con Esc. Sesión administrativa activa.",
            foreground="#666666",
        ).grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=(12, 0))

    def _refresh_cameras(self) -> None:
        cameras: dict[str, int] = {}
        for camera_index in range(10):
            capture = cv2.VideoCapture(camera_index)
            available = capture.isOpened()
            capture.release()
            if available:
                cameras[f"Cámara {camera_index}"] = camera_index

        self.camera_devices = cameras
        values = list(cameras)
        self.camera_selector["values"] = values
        if values:
            preferred = f"Cámara {self.default_camera}"
            self.camera.set(preferred if preferred in cameras else values[0])
            self.status.set(f"{len(values)} cámara(s) disponible(s).")
        else:
            self.camera.set("")
            self.status.set("No se detectaron cámaras disponibles.")

    def _read_configuration(self) -> tuple[int, str, int] | None:
        camera = self.camera_devices.get(self.camera.get())
        server_url = self.server.get().strip()
        if camera is None or not server_url:
            messagebox.showerror(
                "Configuración incompleta",
                "Selecciona una cámara y confirma la URL del servidor.",
                parent=self.root,
            )
            return None

        try:
            photos = int(self.photos.get())
        except ValueError:
            messagebox.showerror(
                "Cantidad inválida",
                "La cantidad de fotos debe ser un número entero.",
                parent=self.root,
            )
            return None
        if photos < 1:
            messagebox.showerror(
                "Cantidad inválida",
                "La cantidad de fotos debe ser mayor que cero.",
                parent=self.root,
            )
            return None
        return camera, server_url, photos

    def _start_operation(
        self, operation: Callable[[], None], description: str
    ) -> None:
        if self.operation_active:
            messagebox.showinfo(
                "Operación en curso",
                "Cierra la ventana de cámara antes de iniciar otra operación.",
                parent=self.root,
            )
            return

        self.operation_active = True
        self.recognize_button.state(["disabled"])
        self.register_button.state(["disabled"])
        self.status.set(f"{description}. Cierra la ventana de cámara con Esc.")

        def worker() -> None:
            try:
                operation()
            except Exception as error:
                self.root.after(0, self._operation_failed, str(error))
            else:
                self.root.after(0, self._operation_finished)

        threading.Thread(target=worker, daemon=True).start()

    def _recognize(self) -> None:
        configuration = self._read_configuration()
        if configuration is None:
            return
        camera, server_url, _ = configuration
        self._start_operation(
            lambda: asyncio.run(
                main(
                    camera,
                    server_url=server_url,
                    recognition_interval=1.0,
                )
            ),
            "Reconocimiento iniciado",
        )

    def _register(self) -> None:
        username = self.username.get().strip()
        if not username:
            messagebox.showerror(
                "Nombre requerido",
                "Indica el nombre de la persona que se registrará.",
                parent=self.root,
            )
            return
        configuration = self._read_configuration()
        if configuration is None:
            return
        camera, server_url, photos = configuration
        self._start_operation(
            lambda: asyncio.run(
                main(
                    camera,
                    server_url=server_url,
                    recognition_interval=1.0,
                    registration_username=username,
                    registration_photos=photos,
                )
            ),
            f"Registro de '{username}' iniciado",
        )

    def _operation_finished(self) -> None:
        self.operation_active = False
        self.recognize_button.state(["!disabled"])
        self.register_button.state(["!disabled"])
        self.status.set("Operación finalizada. Puedes iniciar otra.")

    def _operation_failed(self, error: str) -> None:
        self._operation_finished()
        self.status.set("La operación terminó con un error.")
        messagebox.showerror("Error en la operación", error, parent=self.root)

    def _close(self) -> None:
        if self.operation_active:
            messagebox.showwarning(
                "Operación en curso",
                "Cierra primero la ventana de cámara con Esc.",
                parent=self.root,
            )
            return
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_admin_interface(server_url: str) -> None:
    AdminInterface(default_server=server_url).run()
