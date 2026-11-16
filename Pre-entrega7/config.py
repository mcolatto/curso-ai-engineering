"""Configuración del modelo y límites de ejecución del orquestador."""

import os
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel, ConfigDict, Field, RedisDsn, SecretStr, TypeAdapter, ValidationError,
    field_validator, model_validator,
)


BASE_DIR = Path(__file__).resolve().parent
REDIS_URL_DEFAULT = "redis://redis:6379/0"
VALIDADOR_REDIS_URL = TypeAdapter(RedisDsn)


class Configuracion(BaseModel):
    """Validar parámetros locales sin crear clientes ni consultar servicios."""

    model_config = ConfigDict(hide_input_in_errors=True)

    openai_api_key: SecretStr = Field(repr=False)
    redis_url: SecretStr = Field(default=SecretStr(REDIS_URL_DEFAULT), repr=False)
    model: str = "gpt-6.1-sol"
    timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    max_tokens: int = Field(default=4096, gt=0)
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    recursion_limit: int = Field(default=10, gt=0)
    job_timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = Field(default=None, repr=False)
    langsmith_project: str = Field(default="preentrega7", min_length=1)
    langsmith_agent_id: str | None = Field(
        default=None, pattern=r"^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$",
    )
    langsmith_agent_environment: Literal["local", "development", "staging", "production"] = "local"

    @model_validator(mode="after")
    def validar_langsmith(self) -> "Configuracion":
        if self.langsmith_tracing and (
            self.langsmith_api_key is None or not self.langsmith_api_key.get_secret_value().strip()
        ):
            raise ValueError("LANGSMITH_API_KEY es obligatoria si LANGSMITH_TRACING está activado")
        return self

    @field_validator("openai_api_key")
    @classmethod
    def validar_clave(cls, value: SecretStr) -> SecretStr:
        """Exigir una clave con contenido sin comprobar su autenticidad en red."""
        key = value.get_secret_value().strip()
        if not key:
            raise ValueError("OPENAI_API_KEY es obligatoria y no puede estar vacía")
        return SecretStr(key)

    @field_validator("redis_url")
    @classmethod
    def validar_redis_url(cls, value: SecretStr) -> SecretStr:
        """Validar la URL sin exponer posibles credenciales."""
        url = value.get_secret_value().strip()
        try:
            VALIDADOR_REDIS_URL.validate_python(url)
        except ValidationError:
            raise ValueError("REDIS_URL debe ser una URL redis:// o rediss:// válida") from None
        return SecretStr(url)

    @field_validator("model")
    @classmethod
    def validar_modelo(cls, value: str) -> str:
        """Quitar espacios exteriores y rechazar identificadores vacíos."""
        model = value.strip()
        if not model:
            raise ValueError("OPENAI_MODEL no puede estar vacío")
        return model


def cargar_configuracion() -> Configuracion:
    """Leer variables del entorno; main.py cargará .env cuando se ejecute."""
    reasoning = os.getenv("OPENAI_REASONING_EFFORT", "").strip()
    return Configuracion(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        redis_url=os.getenv("REDIS_URL", REDIS_URL_DEFAULT),
        model=os.getenv("OPENAI_MODEL", "gpt-6.1-sol"),
        timeout_seconds=os.getenv("OPENAI_TIMEOUT_SECONDS", "30"),
        max_tokens=os.getenv("OPENAI_MAX_TOKENS", "4096"),
        reasoning_effort=reasoning or None,
        recursion_limit=os.getenv("RECURSION_LIMIT", "10"),
        job_timeout_seconds=os.getenv("JOB_TIMEOUT_SECONDS", "300"),
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false"),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY", "").strip() or None,
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "preentrega7").strip(),
        langsmith_agent_id=os.getenv("LANGSMITH_AGENT_ID", "").strip() or None,
        langsmith_agent_environment=os.getenv("LANGSMITH_AGENT_ENVIRONMENT", "local").strip(),
    )
