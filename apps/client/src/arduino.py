"""USB serial protocol: ASCII 0 turns pin 13 on, ASCII 1 turns it off."""

import time


class Arduino:
    def __init__(self, port: str, baudrate: int = 9600):
        import serial

        self.serial = serial.Serial(port, baudrate, timeout=1, write_timeout=1)
        # The Uno resets when its USB serial port opens.
        time.sleep(2)
        self.serial.reset_input_buffer()

    def entry(self):
        if self.serial.write(b"0") != 1:
            raise OSError("No se pudo enviar el comando 0 completo")
        reply = self.serial.readline().strip()
        if reply != b"LED_ON":
            raise OSError(f"Arduino no confirmó LED_ON: {reply!r}")

    def close(self):
        self.serial.close()
