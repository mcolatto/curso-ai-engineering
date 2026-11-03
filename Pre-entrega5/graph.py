"""Grafo ReAct: el modelo decide si consulta herramientas o termina el turno."""

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import herramientas


INSTRUCCIONES = (
    "Sos un asistente que consulta clientes y pedidos mediante las herramientas disponibles. "
    "Usá sus resultados para responder; no inventes IDs, cantidades ni montos. "
    "Si una herramienta informa un error o falta información, pedí una aclaración "
    "o corregí la consulta sólo si el contexto permite hacerlo. "
    "Utilizá el historial para interpretar las preguntas de seguimiento."
)


def crear_grafo(llm: Runnable[LanguageModelInput, BaseMessage]) -> StateGraph:
    """Construir el ciclo; quien lo ejecute elegirá el checkpointer al compilar."""

    async def nodo_modelo(state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Enviar instrucciones e historial y agregar únicamente la nueva respuesta."""
        messages = [SystemMessage(content=INSTRUCCIONES), *state["messages"]]
        respuesta = await llm.ainvoke(messages)
        return {"messages": [respuesta]}

    grafo = StateGraph(MessagesState)
    grafo.add_node("modelo", nodo_modelo)
    grafo.add_node("herramientas", ToolNode(herramientas, handle_tool_errors=True))
    grafo.add_edge(START, "modelo")
    grafo.add_conditional_edges(
        "modelo", tools_condition, {"tools": "herramientas", END: END}
    )
    grafo.add_edge("herramientas", "modelo")
    return grafo
