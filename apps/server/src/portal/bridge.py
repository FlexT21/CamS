"""Best-effort output of committed server-generated access events."""

import json
import logging
import queue
import threading
import time

logger = logging.getLogger(__name__)


class EventBridge:
    def __init__(self, config):
        self.config = config
        self.queue = queue.Queue(maxsize=256)
        self.lock = threading.Lock()
        self.thread = None
        self.stopping = threading.Event()
        self.serial = None
        self.mqtt = None
        self.serial_state = "disconnected" if config.serial_port else "disabled"
        self.mqtt_state = "disconnected" if config.mqtt_host else "disabled"
        self.dropped = 0
        self.serial_ready_at = 0.0
        self.serial_retry_at = 0.0
        self.mqtt_retry_at = 0.0

    def start(self):
        """Parent lifespan opt-in. Idempotent; initialize/publish never call this."""
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return not self.stopping.is_set()
            if not (self.config.serial_port or self.config.mqtt_host):
                return False
            self.stopping.clear()
            self.serial_retry_at = self.mqtt_retry_at = 0.0
            self.thread = threading.Thread(
                target=self._run, name="cams-event-bridge", daemon=True
            )
            try:
                self.thread.start()
            except Exception:
                self.thread = None
                logger.warning("Portal hardware worker could not start")
                return False
            return True

    def publish_saved(self, events):
        """Called only with rows returned by Database.save_events after commit."""
        if not (self.config.serial_port or self.config.mqtt_host):
            return
        for event in events:
            if event.get("direction") not in ("entrada", "salida") or not event.get(
                "id"
            ):
                continue
            with self.lock:
                if self.stopping.is_set():
                    self.dropped += 1
                    continue
                try:
                    self.queue.put_nowait(dict(event))
                except queue.Full:
                    self.dropped += 1
                    logger.warning(
                        "Portal hardware queue full; committed event not delivered"
                    )

    def publish_access(self, person_name):
        """Notify the Arduino that the camera accepted the next entry."""
        if not self.config.serial_port:
            return
        with self.lock:
            if self.stopping.is_set():
                self.dropped += 1
                return
            try:
                self.queue.put_nowait(
                    {
                        "direction": "entrada",
                        "id": f"camera-access:{person_name}:{time.monotonic_ns()}",
                    }
                )
            except queue.Full:
                self.dropped += 1
                logger.warning("Portal hardware queue full; camera access not delivered")

    def _close_serial(self):
        device, self.serial = self.serial, None
        self.serial_state = "disconnected" if self.config.serial_port else "disabled"
        self.serial_ready_at = 0.0
        if device is not None:
            try:
                device.close()
            except Exception:
                logger.warning("Portal serial close failed")

    def _ensure_serial(self):
        if not self.config.serial_port:
            return False
        if self.serial is not None and self.serial.is_open:
            return True
        if time.monotonic() < self.serial_retry_at or self.stopping.is_set():
            return False
        self.serial_retry_at = time.monotonic() + 2
        self._close_serial()
        try:
            import serial

            self.serial = serial.Serial(
                self.config.serial_port,
                self.config.serial_baud,
                timeout=1,
                write_timeout=1,
            )
            # Opening an Arduino port resets the board. Do not send during boot.
            self.serial_ready_at = time.monotonic() + 2
            self.serial_state = "unverified"
            return True
        except Exception:
            self._close_serial()
            logger.warning("Portal serial connection failed")
            return False

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if client is self.mqtt:
            self.mqtt_state = (
                "connected"
                if reason_code == 0 and not self.stopping.is_set()
                else "disconnected"
            )

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        if client is self.mqtt:
            self.mqtt_state = "disconnected"

    def _close_mqtt(self):
        client, self.mqtt = self.mqtt, None
        self.mqtt_state = "disconnected" if self.config.mqtt_host else "disabled"
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                logger.warning("Portal MQTT disconnect failed")
            try:
                client.loop_stop()
            except Exception:
                logger.warning("Portal MQTT loop shutdown failed")

    def _ensure_mqtt(self):
        if not self.config.mqtt_host or self.mqtt is not None:
            return
        if time.monotonic() < self.mqtt_retry_at or self.stopping.is_set():
            return
        self.mqtt_retry_at = time.monotonic() + 5
        try:
            import paho.mqtt.client as mqtt

            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.mqtt = client
            client.connect_timeout = 2
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.reconnect_delay_set(min_delay=2, max_delay=5)
            if self.config.mqtt_username:
                client.username_pw_set(
                    self.config.mqtt_username, self.config.mqtt_password
                )
            if self.config.mqtt_tls:
                client.tls_set()
            # The paho loop owns CONNECT/CONNACK and reconnects, without any publish.
            client.connect_async(
                self.config.mqtt_host, self.config.mqtt_port, keepalive=30
            )
            client.loop_start()
        except Exception:
            self._close_mqtt()
            logger.warning("Portal MQTT connection failed")

    def _serial_event(self, event):
        if not self.config.serial_port or event.get("direction") not in (
            "entrada",
            "salida",
        ):
            return
        if not self._ensure_serial():
            return
        try:
            remaining = self.serial_ready_at - time.monotonic()
            if remaining > 0 and self.stopping.wait(remaining):
                return
            if event["direction"] != "entrada":
                self.serial_state = "connected"
                return
            payload = b"ACCEPTED\n"
            self.serial.reset_input_buffer()
            if self.serial.write(payload) != len(payload):
                raise OSError("Incomplete serial write")
            self.serial_state = "connected"
        except Exception as error:
            self._close_serial()
            self.serial_retry_at = time.monotonic() + 2
            logger.warning("Portal serial delivery failed: %s", error)

    def _mqtt_event(self, event):
        if not self.config.mqtt_host or event.get("direction") not in (
            "entrada",
            "salida",
        ):
            return
        self._ensure_mqtt()
        client = self.mqtt
        if client is None or not client.is_connected():
            return
        try:
            info = client.publish("cams/events", json.dumps(event), qos=1, retain=False)
            info.wait_for_publish(timeout=2)
        except Exception:
            if not client.is_connected():
                self.mqtt_state = "disconnected"
            logger.warning("Portal MQTT delivery failed")

    def _run(self):
        try:
            while not self.stopping.is_set():
                self._ensure_mqtt()
                self._ensure_serial()
                try:
                    event = self.queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    self._serial_event(event)
                    if not self.stopping.is_set():
                        self._mqtt_event(event)
                finally:
                    self.queue.task_done()
        finally:
            self._close_serial()
            self._close_mqtt()

    def status(self):
        serial_state = self.serial_state
        mqtt_state = self.mqtt_state
        serial, mqtt = self.serial, self.mqtt
        if serial is not None and not serial.is_open:
            serial_state = "disconnected"
        if mqtt is not None and not mqtt.is_connected():
            mqtt_state = "disconnected"
        return {
            "serial": serial_state,
            "mqtt": mqtt_state,
            "pending_outputs": self.queue.qsize(),
            "dropped_outputs": self.dropped,
        }

    def close(self):
        with self.lock:
            self.stopping.set()
            thread = self.thread
        if thread:
            thread.join(timeout=5)
            if thread.is_alive():
                return  # Worker owns cleanup; do not close a device during a write.
        with self.lock:
            if self.thread is not thread:
                return
            self._close_serial()
            self._close_mqtt()
            while True:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                    self.dropped += 1
                except queue.Empty:
                    break
