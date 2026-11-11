"""Service settings from environment variables (prefix ``SENTIMETRO_``) or ``.env``."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTIMETRO_", env_file=".env", extra="ignore")

    model_dir: Path = Path("models/production")
    max_batch_size: int = Field(default=64, ge=1, le=1024)
    max_text_chars: int = Field(default=5000, ge=10)
    # Optional shared secret for /predict (header `X-API-Key`). Empty = open API.
    api_key: SecretStr | None = None
    cors_origins: list[str] = Field(default_factory=list)
    log_level: str = "INFO"
    log_json: bool = True
    enable_demo: bool = True
