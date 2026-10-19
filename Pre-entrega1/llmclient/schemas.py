"""Estructuras de mensajes, configuración y respuestas del cliente."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class Provider(str, Enum):
    """Proveedores que puede seleccionar el cliente."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatMessage(BaseModel):
    """Mensaje de entrada con un rol reconocido y contenido de texto."""

    role: Literal["user", "assistant", "system"] = Field(description="user, assistant o system")
    content: str

    @field_validator("content")
    @classmethod
    def validar_contenido(cls, value: str) -> str:
        """Rechazar mensajes sin texto y conservar el contenido válido original."""
        if not value.strip():
            raise ValueError("El contenido del mensaje no puede estar vacío")
        return value


class LLMConfig(BaseModel):
    """Configuración validada para un proveedor y modelo."""

    provider: Provider
    model: str
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validar_credenciales(self) -> "LLMConfig":
        """Exigir una clave no vacía para el proveedor seleccionado."""
        if self.provider == Provider.OPENAI:
            secret = self.openai_api_key
            key_name = "openai_api_key"
        else:
            secret = self.anthropic_api_key
            key_name = "anthropic_api_key"
        if secret is None or not secret.get_secret_value().strip():
            raise ValueError(f"Se requiere {key_name} para el proveedor seleccionado")
        return self

    @field_validator("model")
    @classmethod
    def validar_modelo(cls, value: str) -> str:
        """Quitar espacios exteriores y rechazar un identificador vacío."""
        model = value.strip()
        if not model:
            raise ValueError("El modelo no puede estar vacío")
        return model


class ModelResponse(BaseModel):
    """Respuesta completa o error controlado, según el contrato docente."""

    provider: Provider
    model: str
    content: str
    error: str | None = None
