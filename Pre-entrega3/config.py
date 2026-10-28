"""Configuración compartida por la ingesta y las consultas RAG."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "preentrega3_documentos"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
TOKEN_ENCODING = "cl100k_base"


class Configuracion(BaseModel):
    """Validar credenciales y parámetros comunes sin exponer secretos."""

    model_config = ConfigDict(hide_input_in_errors=True)

    api_key: SecretStr = Field(repr=False)
    model: str = "gpt-6.1-sol"
    embedding_model: str = "text-embedding-3-small"
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = "low"
    max_tokens: int = Field(default=1024, gt=0)
    timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)

    @field_validator("api_key")
    @classmethod
    def validar_clave(cls, value: SecretStr) -> SecretStr:
        """Exigir una clave con contenido sin comprobarla contra la API."""
        if not value.get_secret_value().strip():
            raise ValueError("Falta OPENAI_API_KEY en la configuración")
        return SecretStr(value.get_secret_value().strip())

    @field_validator("model", "embedding_model")
    @classmethod
    def validar_modelo(cls, value: str) -> str:
        """Rechazar identificadores vacíos y quitar espacios exteriores."""
        if not value.strip():
            raise ValueError("El identificador del modelo no puede estar vacío")
        return value.strip()


def cargar_configuracion() -> Configuracion:
    """Leer el entorno; los scripts se encargarán de cargar .env al ejecutarse."""
    model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-6.1-sol"
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "").strip() or "text-embedding-3-small"
    default_effort = "low" if model == "gpt-6.1-sol" else ""
    effort = os.getenv("OPENAI_REASONING_EFFORT", default_effort).strip() or None
    return Configuracion(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=model,
        embedding_model=embedding_model,
        reasoning_effort=effort,
        max_tokens=os.getenv("LLM_MAX_TOKENS", "1024"),
        timeout_seconds=os.getenv("LLM_TIMEOUT_SECONDS", "30"),
    )
