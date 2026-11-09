"""Contratos de solicitud, investigación y evaluación del viaje."""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator


# Reutilizar la misma regla para nombres, fuentes, IDs y explicaciones.
Texto = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Componente = Literal["alojamiento", "actividades", "gastronomia"]
SiguienteAgente = Literal["hoteleria", "turismo", "analista", "finalizar"]
EvaluacionSupervisor = Literal["aprobado", "aprobado_con_advertencias", "desaprobado"]


class SolicitudParcial(BaseModel):
    """Representar lo extraído del usuario, incluidos datos todavía ausentes."""

    destino: Texto | None = None
    fecha_inicio: date | None = None
    fecha_fin: date | None = None
    cantidad_personas: int | None = Field(default=None, gt=0)
    componentes: list[Componente] = Field(default_factory=list)
    preferencias: list[Texto] = Field(default_factory=list)

    @model_validator(mode="after")
    def validar_fechas(self) -> "SolicitudParcial":
        """Rechazar intervalos de viaje invertidos o sin duración."""
        if self.fecha_inicio is not None and self.fecha_fin is not None:
            if self.fecha_fin <= self.fecha_inicio:
                raise ValueError("La fecha de fin debe ser posterior a la de inicio")
        return self

    def datos_faltantes(self) -> list[str]:
        """Identificar los cuatro datos mínimos sin consultar servicios."""
        fields = ("destino", "fecha_inicio", "fecha_fin", "cantidad_personas")
        return [name for name in fields if getattr(self, name) is None]


class SolicitudViaje(SolicitudParcial):
    """Solicitud completa que habilita las investigaciones solicitadas."""

    destino: Texto
    fecha_inicio: date
    fecha_fin: date
    cantidad_personas: int = Field(gt=0)
    componentes: list[Componente] = Field(min_length=1)


class Coordenadas(BaseModel):
    """Ubicación geográfica conocida de una opción."""

    latitud: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitud: float = Field(ge=-180, le=180, allow_inf_nan=False)


class Periodo(BaseModel):
    """Intervalo inclusivo informado por una fuente, sin inferir disponibilidad."""

    inicio: date
    fin: date

    @model_validator(mode="after")
    def validar_orden(self) -> "Periodo":
        if self.fin < self.inicio:
            raise ValueError("El fin del período no puede preceder a su inicio")
        return self


class Observacion(BaseModel):
    """Limitación, incertidumbre o inconsistencia con descripción explícita."""

    tipo: Literal["limitacion", "incertidumbre", "inconsistencia"]
    descripcion: Texto


class OpcionHotel(BaseModel):
    """Alojamiento recuperado con datos originales y procedencia."""

    id: Texto
    nombre: Texto
    fuente: Texto
    direccion: Texto | None = None
    coordenadas: Coordenadas | None = None
    estrellas: int | None = Field(default=None, ge=1, le=5)
    capacidad: int | None = Field(default=None, gt=0)
    disponibilidad: Literal["disponible_en_fuente", "no_disponible", "sin_datos"]
    periodos_disponibles: list[Periodo] | None = None
    observaciones: list[Observacion] = Field(default_factory=list)


class OpcionTuristica(BaseModel):
    """Actividad o establecimiento, con restricciones sólo si constan en la fuente."""

    id: Texto
    nombre: Texto
    fuente: Texto
    categoria: Literal["gastronomy", "attraction", "viewpoint"]
    direccion: Texto | None = None
    coordenadas: Coordenadas | None = None
    periodos_disponibles: list[Periodo] | None = None
    restricciones: list[Texto] = Field(default_factory=list)
    observaciones: list[Observacion] = Field(default_factory=list)


class ResultadoHoteleria(BaseModel):
    """Búsqueda completada, incluso vacía, o fallo descrito en observaciones."""

    estado_busqueda: Literal["completada", "error"]
    hoteles: list[OpcionHotel] = Field(default_factory=list)
    observaciones: list[Observacion] = Field(default_factory=list)


class ResultadoTurismo(BaseModel):
    """Opciones recuperadas sin clasificarlas todavía por compatibilidad."""

    estado_busqueda: Literal["completada", "error"]
    opciones: list[OpcionTuristica] = Field(default_factory=list)
    observaciones: list[Observacion] = Field(default_factory=list)


class OpcionEvaluada(BaseModel):
    """Clasificación del Analista vinculada a la opción original y su dominio."""

    opcion_id: Texto
    dominio: Literal["hoteleria", "turismo"]
    clasificacion: Literal["compatible", "pendiente_verificacion", "incompatible"]
    justificacion: Texto
    distancia_km: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class ResultadoAnalisis(BaseModel):
    """Evaluación de las opciones disponibles sin alterar la investigación."""

    sintesis_analisis: Texto
    resultados_evaluados: list[OpcionEvaluada] = Field(default_factory=list)
    observaciones: list[Observacion] = Field(default_factory=list)


class DecisionSupervisor(BaseModel):
    """Ruta elegida y tarea concreta; la respuesta final se publicará en messages."""

    next_agent: SiguienteAgente
    instruccion: Texto
    evaluacion_supervisor: EvaluacionSupervisor | None = None


class SalidaSupervisor(BaseModel):
    """Extracción inicial y decisión del modelo, incluida su respuesta de cierre."""

    es_turistica: bool
    solicitud: SolicitudParcial | None = None
    decision: DecisionSupervisor
    respuesta_final: Texto | None = None
