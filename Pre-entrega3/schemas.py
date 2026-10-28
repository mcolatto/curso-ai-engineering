"""Contratos de salida para la respuesta del LLM y el resultado RAG."""

from pydantic import BaseModel, Field, field_validator


class RespuestaLLM(BaseModel):
    """Texto generado por el modelo y validado por PydanticOutputParser."""

    respuesta: str = Field(
        description=(
            "Respuesta basada exclusivamente en el contexto proporcionado. "
            "Si no contiene información suficiente, responder exactamente: No lo sé"
        )
    )

    @field_validator("respuesta")
    @classmethod
    def validar_respuesta(cls, value: str) -> str:
        """Exigir contenido real sin modificar el texto válido."""
        if not value.strip():
            raise ValueError("La respuesta no puede estar vacía")
        return value


class RAGResponse(RespuestaLLM):
    """Agregar a la respuesta validada los metadatos del contexto recuperado."""

    fuentes: list[str] = Field(
        description="Archivos de origen de los fragmentos proporcionados como contexto"
    )
    fragmentos_recuperados: int = Field(
        ge=0, description="Cantidad de fragmentos recuperados para la consulta"
    )

    @field_validator("fuentes")
    @classmethod
    def validar_fuentes(cls, value: list[str]) -> list[str]:
        """Permitir una lista vacía, pero no nombres sin contenido."""
        for fuente in value:
            if not fuente.strip():
                raise ValueError("Cada fuente debe contener un nombre no vacío")
        return value
