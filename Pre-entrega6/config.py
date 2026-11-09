"""Configuración del modelo y límites de ejecución del orquestador."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints.sqlite"
TRACE_PATH = BASE_DIR / "traza_ejecucion.json"


class Configuracion(BaseModel):
    """Validar parámetros locales sin crear clientes ni consultar servicios."""

    model_config = ConfigDict(hide_input_in_errors=True)

    openai_api_key: SecretStr = Field(repr=False)
    model: str = "gpt-6.1-sol"
    timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    max_tokens: int = Field(default=4096, gt=0)
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    recursion_limit: int = Field(default=10, gt=0)

    @field_validator("openai_api_key")
    @classmethod
    def validar_clave(cls, value: SecretStr) -> SecretStr:
        """Exigir una clave con contenido sin comprobar su autenticidad en red."""
        key = value.get_secret_value().strip()
        if not key:
            raise ValueError("OPENAI_API_KEY es obligatoria y no puede estar vacía")
        return SecretStr(key)

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
        model=os.getenv("OPENAI_MODEL", "gpt-6.1-sol"),
        timeout_seconds=os.getenv("OPENAI_TIMEOUT_SECONDS", "30"),
        max_tokens=os.getenv("OPENAI_MAX_TOKENS", "4096"),
        reasoning_effort=reasoning or None,
        recursion_limit=os.getenv("RECURSION_LIMIT", "10"),
    )
