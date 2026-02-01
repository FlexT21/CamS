from typing import Annotated, Generator

from fastapi import Depends

from src.core.config import settings
from src.messaging.publisher import MQTTPublisher


def get_mqtt_publisher() -> Generator[MQTTPublisher, None, None]:
    publisher = MQTTPublisher(
        broker_address=settings.MQTT_BROKER_ADDRESS,
        broker_port=settings.MQTT_BROKER_PORT,
        retries=settings.MQTT_RETRIES_ATTEMPS,
        delay=settings.MQTT_RETRY_DELAY_SECONDS,
    )

    try:
        yield publisher
    finally:
        pass


MQTTPublisherDep = Annotated[MQTTPublisher, Depends(get_mqtt_publisher)]
