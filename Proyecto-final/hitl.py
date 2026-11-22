"""Proponer una reserva compatible, pausar y registrar sólo tras la aprobación."""

from collections.abc import Awaitable, Callable
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from redis.asyncio import Redis

from agents.analista import comprobar_opcion
from schemas import PropuestaReserva, RespuestaAprobacion, SolicitudViaje
from state import EstadoViaje

RegistrarReserva = Callable[[UUID, PropuestaReserva], Awaitable[str]]
PREFIJO_RESERVA = "proyecto-final:reserva:"


def proponer_reserva(state: EstadoViaje) -> PropuestaReserva | None:
    """Elegir el primer hotel del catálogo que cumple las verificaciones."""
    solicitud = state["solicitud_original"]
    hoteles = state["resultado_hoteleria"]
    analisis = state["resultado_analisis"]
    if (
        not state.get("solicita_reserva_simulada")
        or not isinstance(solicitud, SolicitudViaje)
        or "alojamiento" not in solicitud.componentes
        or hoteles is None or hoteles.estado_busqueda != "completada"
        or analisis is None
    ):
        return None

    compatibles = {
        opcion.opcion_id for opcion in analisis.resultados_evaluados
        if opcion.dominio == "hoteleria" and opcion.clasificacion == "compatible"
    }
    for hotel in hoteles.hoteles:
        if hotel.id not in compatibles:
            continue
        evidencia = comprobar_opcion(solicitud, hotel)
        if evidencia["capacidad_suficiente"] is True and evidencia["compatibilidad_temporal"] is True:
            return PropuestaReserva(
                hotel_id=hotel.id, hotel=hotel.nombre,
                fecha_inicio=solicitud.fecha_inicio, fecha_fin=solicitud.fecha_fin,
                cantidad_personas=solicitud.cantidad_personas,
            )
    return None


def agregar_resultado_reserva(state: EstadoViaje, texto: str) -> dict:
    """Completar la respuesta del Supervisor conservando el ID de su mensaje."""
    mensaje = state["messages"][-1]
    return {"messages": [mensaje.model_copy(update={"content": f"{mensaje.text}\n\n{texto}"})]}


def aprobar_reserva(state: EstadoViaje) -> dict:
    """Pausar con una propuesta concreta. Esta parte se repite al reanudar."""
    propuesta = proponer_reserva(state)
    if propuesta is None:
        return agregar_resultado_reserva(
            state, "No se pudo proponer una reserva simulada con las opciones compatibles verificadas.",
        )

    respuesta = RespuestaAprobacion.model_validate(interrupt(propuesta.model_dump(mode="json")))
    if respuesta.propuesta != propuesta:
        raise ValueError("La propuesta reanudada no coincide con la propuesta presentada")
    decision = respuesta.decision
    if decision == "rechazar":
        return {
            "decision_aprobacion": decision,
            **agregar_resultado_reserva(state, "La propuesta fue rechazada. No se realizó ninguna reserva."),
        }
    return {"decision_aprobacion": decision, "propuesta_reserva": propuesta}


async def reservar_hotel_simulado(redis: Redis, job_id: UUID, propuesta: PropuestaReserva) -> str:
    """Registrar la simulación una sola vez y devolver siempre la misma referencia."""
    clave = PREFIJO_RESERVA + str(job_id)
    registro = propuesta.model_dump_json()
    creado = await redis.set(clave, registro, nx=True)
    if not creado and await redis.get(clave) != registro:
        raise ValueError("Ya existe una reserva diferente para este trabajo")
    return f"SIM-{job_id}"


def crear_nodo_reserva(registrar: RegistrarReserva | None):
    """Mantener el acceso a Redis separado del momento en que se pide aprobación."""
    async def reservar(state: EstadoViaje, config: RunnableConfig) -> dict:
        propuesta = state.get("propuesta_reserva")
        if state.get("decision_aprobacion") != "aprobar" or propuesta is None:
            raise ValueError("La reserva requiere una propuesta aprobada")
        if registrar is None:
            raise RuntimeError("No se configuró el registro de reservas")

        job_id = UUID(config["configurable"]["thread_id"])
        referencia = await registrar(job_id, propuesta)
        return agregar_resultado_reserva(
            state, f"Se registró la reserva simulada en {propuesta.hotel}. Referencia: {referencia}.",
        )

    return reservar
