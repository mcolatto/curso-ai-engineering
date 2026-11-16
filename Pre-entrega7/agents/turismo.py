"""Investigación de actividades, miradores y gastronomía mediante su herramienta especializada."""

from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from schemas import ResultadoTurismo, SolicitudViaje
from state import EstadoViaje
from tools import buscar_turismo, normalizar_destino


INSTRUCCIONES = (
    "Sos el investigador de turismo. Consultá actividades, miradores y gastronomía usando únicamente la herramienta asignada. "
    "Usá el destino de la solicitud. No evalúes compatibilidad, no descartes opciones "
    "por fechas o capacidad y no inventes datos. La fuente contiene información simulada."
)


def crear_agente_turismo(llm: ChatOpenAI) -> Callable[[EstadoViaje], Awaitable[dict]]:
    """Vincular exclusivamente la herramienta del rol y devolver su nodo asíncrono."""
    model = llm.bind_tools([buscar_turismo], tool_choice="buscar_turismo")

    async def investigar(state: EstadoViaje) -> dict:
        """Consultar y validar la fuente, actualizando sólo el resultado del rol."""
        partial = state["solicitud_original"]
        decision = state["decision_supervisor"]
        if partial is None or decision is None or decision.next_agent != "turismo":
            raise ValueError("Se requiere una solicitud completa y una delegación a turismo")
        request = SolicitudViaje.model_validate(partial.model_dump())
        response = await model.ainvoke([
            SystemMessage(content=INSTRUCCIONES),
            HumanMessage(content=(
                f"Solicitud: {request.model_dump_json()}\n"
                f"Tarea delegada: {decision.instruccion}"
            )),
        ])
        calls = response.tool_calls
        if len(calls) != 1 or calls[0]["name"] != "buscar_turismo":
            raise ValueError("El investigador debe emitir una única consulta a buscar_turismo")
        arguments = calls[0]["args"]
        destination = arguments.get("destino")
        if not isinstance(destination, str) or normalizar_destino(destination) != normalizar_destino(request.destino):
            raise ValueError("La consulta debe conservar el destino solicitado")
        data = await buscar_turismo.ainvoke(arguments)
        result = ResultadoTurismo.model_validate(data)
        return {"resultado_turismo": result}

    return investigar
