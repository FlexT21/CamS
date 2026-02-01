from typing import Callable, TypeVar

import paho.mqtt.client as mqtt

T = TypeVar("T")


class MQTTPublisher:
    def __init__(
        self,
        *,
        broker_address: str,
        broker_port: int,
        retries: int = 3,
        delay: float = 2.0,
    ) -> None:
        self.client = mqtt.Client()
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.retries = retries
        self.delay = delay

    def _connect(self) -> None:
        self.client.connect(self.broker_address, self.broker_port)

    def _close(self) -> None:
        self.client.disconnect()

    def __execution_with_retry(self, operation: Callable[[], T]) -> T:
        for attempt in range(self.retries):
            try:
                self._connect()
                result = operation()
                self._close()
                return result
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.retries - 1:
                    import time

                    time.sleep(self.delay)

        raise ConnectionError(
            f"Failed to connect to MQTT broker after {self.retries} attempts."
        )

    def publish(self, topic: str, message: str) -> None:
        def operation() -> None:
            self.client.publish(topic, message)

        self.__execution_with_retry(operation)
