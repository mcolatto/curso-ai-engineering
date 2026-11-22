"""Grafo jerárquico secuencial con decisiones dinámicas del Supervisor."""

from collections.abc import Awaitable, Callable
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agents.analista import crear_agente_analista
from agents.hoteleria import crear_agente_hoteleria
from agents.supervisor import crear_supervisor
from agents.turismo import crear_agente_turismo
from hitl import RegistrarReserva, aprobar_reserva, crear_nodo_reserva
from schemas import SiguienteAgente, SolicitudViaje
from state import EstadoViaje


def siguiente_agente(state: EstadoViaje) -> SiguienteAgente | Literal["aprobar_reserva"]:
    """Traducir la decisión validada a una arista del grafo."""
    decision = state["decision_supervisor"]
    if decision is None:
        raise ValueError("Falta la decisión del Supervisor")
    if (
        decision.next_agent == "finalizar"
        and state.get("solicita_reserva_simulada")
        and isinstance(state.get("solicitud_original"), SolicitudViaje)
    ):
        return "aprobar_reserva"
    return decision.next_agent


def despues_de_aprobar(state: EstadoViaje) -> Literal["reservar", "finalizar"]:
    return "reservar" if state.get("decision_aprobacion") == "aprobar" else "finalizar"


def crear_grafo(
    llm: ChatOpenAI, *,
    nodo_analista: Callable[[EstadoViaje], Awaitable[dict]] | None = None,
    registrar_reserva: RegistrarReserva | None = None,
) -> StateGraph:
    """Construir nodos sin invocarlos; el ejecutor configurará persistencia y límites."""
    graph = StateGraph(EstadoViaje)
    graph.add_node("supervisor", crear_supervisor(llm))
    graph.add_node("hoteleria", crear_agente_hoteleria(llm))
    graph.add_node("turismo", crear_agente_turismo(llm))
    graph.add_node("analista", nodo_analista if nodo_analista is not None else crear_agente_analista(llm))
    graph.add_node("aprobar_reserva", aprobar_reserva)
    graph.add_node("reservar", crear_nodo_reserva(registrar_reserva))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", siguiente_agente, {
        "hoteleria": "hoteleria", "turismo": "turismo",
        "analista": "analista", "finalizar": END,
        "aprobar_reserva": "aprobar_reserva",
    })
    for name in ("hoteleria", "turismo", "analista"):
        graph.add_edge(name, "supervisor")
    graph.add_conditional_edges("aprobar_reserva", despues_de_aprobar, {"reservar": "reservar", "finalizar": END})
    graph.add_edge("reservar", END)
    return graph
