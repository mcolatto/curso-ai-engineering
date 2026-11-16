"""Estado compartido: cada especialista actualiza su propio resultado."""

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

from schemas import (
    DecisionHumana, DecisionSupervisor, PropuestaReserva, ResultadoAnalisis, ResultadoHoteleria, ResultadoTurismo,
    SolicitudParcial, SolicitudViaje,
)


class EstadoViaje(MessagesState):
    """None distingue resultados aún no producidos de búsquedas vacías o fallidas."""

    solicitud_original: SolicitudParcial | SolicitudViaje | None
    resultado_hoteleria: ResultadoHoteleria | None
    resultado_turismo: ResultadoTurismo | None
    resultado_analisis: ResultadoAnalisis | None
    refinamientos_analista: int
    decision_supervisor: DecisionSupervisor | None
    solicita_reserva_simulada: bool
    propuesta_reserva: PropuestaReserva | None
    decision_aprobacion: DecisionHumana | None


def crear_estado_inicial(texto: str) -> EstadoViaje:
    """Conservar el mensaje original sin anticipar extracción ni delegaciones."""
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("La solicitud debe contener texto no vacío")
    return {
        "messages": [HumanMessage(content=texto)],
        "solicitud_original": None,
        "resultado_hoteleria": None,
        "resultado_turismo": None,
        "resultado_analisis": None,
        "refinamientos_analista": 0,
        "decision_supervisor": None,
        "solicita_reserva_simulada": False,
        "propuesta_reserva": None,
        "decision_aprobacion": None,
    }
