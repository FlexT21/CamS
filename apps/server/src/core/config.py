from typing import List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Server settings
    SERVER_PREFIX: str = "/api"
    SERVER_PORT: int = 8765
    SERVER_CORS_ORIGINS: List[str]

    # Image settings
    VALID_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]

    # Recognition settings
    THRESHOLD_DISTANCE: float = 0.55
    K_MEANS_CLUSTERS: int = 3

    # MQTT settings
    MQTT_BROKER_ADDRESS: str = "localhost"
    MQTT_BROKER_PORT: int = 1883


settings = Settings()  # type: ignore


if __name__ == "__main__":
    print(settings.model_dump_json(indent=4))
