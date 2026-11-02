import serial


class ArduinoSignal:
    """Send access decisions to an Arduino over a line-based serial protocol."""

    def __init__(self, port: str, baudrate: int = 9600) -> None:
        self._connection = serial.Serial(port=port, baudrate=baudrate, timeout=1)

    def send(self, accepted: bool) -> None:
        signal = "ACCEPTED\n" if accepted else "DENIED\n"
        self._connection.write(signal.encode("ascii"))
        self._connection.flush()

    def close(self) -> None:
        if self._connection.is_open:
            self._connection.close()

    def __enter__(self) -> "ArduinoSignal":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()
