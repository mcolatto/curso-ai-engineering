"""Configuración compartida por la ingesta, recuperación y evaluación."""

import math
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GOLDEN_SET_PATH = BASE_DIR / "golden_dataset.json"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
TOKEN_ENCODING = "cl100k_base"
TOP_K = 5


class Configuracion(BaseModel):
    """Validar parámetros y ocultar credenciales en representaciones y errores."""

    model_config = ConfigDict(hide_input_in_errors=True)

    openai_api_key: SecretStr = Field(repr=False)
    pinecone_api_key: SecretStr = Field(repr=False)
    index_name: str = "preentrega4-rag-hibrido"
    namespace: str = "langgraph-docs"
    cloud: str = "aws"
    region: str = "us-east-1"
    chunk_size: int = Field(default=600, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    bm25_weight: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    vector_weight: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)

    @field_validator("openai_api_key", "pinecone_api_key")
    @classmethod
    def validar_clave(cls, value: SecretStr) -> SecretStr:
        """Exigir contenido sin comprobar las claves contra servicios externos."""
        if not value.get_secret_value().strip():
            raise ValueError("La API key es obligatoria y no puede estar vacía")
        return SecretStr(value.get_secret_value().strip())

    @field_validator("index_name", "namespace", "cloud", "region")
    @classmethod
    def validar_texto(cls, value: str) -> str:
        """Evitar nombres vacíos, especialmente el namespace predeterminado global."""
        if not value.strip():
            raise ValueError("El parámetro debe contener texto no vacío")
        return value.strip()

    @model_validator(mode="after")
    def validar_relaciones(self) -> "Configuracion":
        """Comprobar el solapamiento y la suma de los dos pesos."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP debe ser menor que CHUNK_SIZE")
        if not math.isclose(self.bm25_weight + self.vector_weight, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("BM25_WEIGHT y VECTOR_WEIGHT deben sumar 1")
        return self

    @property
    def weights(self) -> list[float]:
        """Mantener el orden BM25, Pinecone esperado por EnsembleRetriever."""
        return [self.bm25_weight, self.vector_weight]


def cargar_configuracion() -> Configuracion:
    """Leer el entorno; los scripts cargarán .env al ejecutarse."""
    return Configuracion(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        index_name=os.getenv("INDEX_NAME", "preentrega4-rag-hibrido"),
        namespace=os.getenv("NAMESPACE", "langgraph-docs"),
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_REGION", "us-east-1"),
        chunk_size=os.getenv("CHUNK_SIZE", "600"),
        chunk_overlap=os.getenv("CHUNK_OVERLAP", "100"),
        bm25_weight=os.getenv("BM25_WEIGHT", "0.5"),
        vector_weight=os.getenv("VECTOR_WEIGHT", "0.5"),
    )
