"""Contrato de salida validado para la extracción de entidades técnicas."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class NivelDeCriticidad(str, Enum):
    """Niveles de criticidad admitidos para el texto analizado."""

    baja = "baja"
    media = "media"
    alta = "alta"


class EntidadesTecnicas(BaseModel):
    """Tecnologías, criticidad y resumen extraídos de un texto."""

    tecnologias: list[str] = Field(
        description="Tecnologías identificadas en el texto; lista vacía si no se mencionan."
    )
    nivel_de_criticidad: NivelDeCriticidad = Field(
        description="Criticidad del texto analizado: baja, media o alta."
    )
    resumen_tecnico: str = Field(
        description="Resumen técnico basado en el texto, sin inventar información."
    )

    @field_validator("tecnologias")
    @classmethod
    def validar_tecnologias(cls, value: list[str]) -> list[str]:
        """Permitir una lista vacía y normalizar el texto de cada tecnología."""
        tecnologias_limpias = []
        for tecnologia in value:
            limpia = tecnologia.strip()
            if not limpia:
                raise ValueError("Cada tecnología debe contener texto")
            tecnologias_limpias.append(limpia)
        return tecnologias_limpias

    @field_validator("resumen_tecnico")
    @classmethod
    def validar_resumen(cls, value: str) -> str:
        """Rechazar un resumen vacío sin modificar el texto válido."""
        if not value.strip():
            raise ValueError("El resumen técnico debe contener texto")
        return value
