"""Grafo jerárquico secuencial con decisiones dinámicas del Supervisor."""

from collections.abc import Awaitable, Callable

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agents.analista import crear_agente_analista
from agents.hoteleria import crear_agente_hoteleria
from agents.supervisor import crear_supervisor
from agents.turismo import crear_agente_turismo
from schemas import SiguienteAgente
from state import EstadoViaje


def siguiente_agente(state: EstadoViaje) -> SiguienteAgente:
    """Traducir la decisión validada a una arista del grafo."""
    decision = state["decision_supervisor"]
    if decision is None:
        raise ValueError("Falta la decisión del Supervisor")
    return decision.next_agent


def crear_grafo(
    llm: ChatOpenAI, *, nodo_analista: Callable[[EstadoViaje], Awaitable[dict]] | None = None
) -> StateGraph:
    """Construir nodos sin invocarlos; el ejecutor configurará persistencia y límites."""
    graph = StateGraph(EstadoViaje)
    graph.add_node("supervisor", crear_supervisor(llm))
    graph.add_node("hoteleria", crear_agente_hoteleria(llm))
    graph.add_node("turismo", crear_agente_turismo(llm))
    graph.add_node("analista", nodo_analista if nodo_analista is not None else crear_agente_analista(llm))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", siguiente_agente, {
        "hoteleria": "hoteleria", "turismo": "turismo",
        "analista": "analista", "finalizar": END,
    })
    for name in ("hoteleria", "turismo", "analista"):
        graph.add_edge(name, "supervisor")
    return graph
