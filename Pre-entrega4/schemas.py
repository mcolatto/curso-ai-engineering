"""Contratos de los metadatos de ingesta y los casos de evaluación."""

from pydantic import BaseModel, Field, field_validator


class MetadataChunk(BaseModel):
    """Metadatos obligatorios que acompañan cada vector en Pinecone."""

    text: str = Field(description="Texto original del fragmento")
    source: str = Field(description="Nombre original del archivo Markdown")
    categoria: str = Field(description="Categoría derivada del nombre del archivo")
    chunk_id: str = Field(description="Identificador determinista del fragmento")

    @field_validator("text", "source", "categoria", "chunk_id")
    @classmethod
    def validar_texto(cls, value: str) -> str:
        """Exigir contenido real sin modificar el texto recibido."""
        if not value.strip():
            raise ValueError("El campo debe contener texto no vacío")
        return value


class CasoGolden(BaseModel):
    """Pregunta y única fuente esperada para un caso de evaluación."""

    pregunta: str = Field(description="Consulta que se enviará al recuperador híbrido")
    documento_id_esperado: str = Field(description="Nombre del archivo fuente esperado")

    @field_validator("pregunta", "documento_id_esperado")
    @classmethod
    def validar_texto(cls, value: str) -> str:
        """Rechazar preguntas o fuentes sin contenido real."""
        if not value.strip():
            raise ValueError("El campo debe contener texto no vacío")
        return value
