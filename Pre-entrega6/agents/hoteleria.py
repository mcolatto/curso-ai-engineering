"""Investigación de alojamientos mediante su herramienta especializada."""

from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from schemas import ResultadoHoteleria, SolicitudViaje
from state import EstadoViaje
from tools import buscar_hoteles, normalizar_destino


INSTRUCCIONES = (
    "Sos el investigador de hoteleria. Consultá alojamientos usando únicamente la herramienta asignada. "
    "Usá el destino de la solicitud. No evalúes compatibilidad, no descartes opciones "
    "por fechas o capacidad y no inventes datos. La fuente contiene información simulada."
)


def crear_agente_hoteleria(llm: ChatOpenAI) -> Callable[[EstadoViaje], Awaitable[dict]]:
    """Vincular exclusivamente la herramienta del rol y devolver su nodo asíncrono."""
    model = llm.bind_tools([buscar_hoteles], tool_choice="buscar_hoteles")

    async def investigar(state: EstadoViaje) -> dict:
        """Consultar y validar la fuente, actualizando sólo el resultado del rol."""
        partial = state["solicitud_original"]
        decision = state["decision_supervisor"]
        if partial is None or decision is None or decision.next_agent != "hoteleria":
            raise ValueError("Se requiere una solicitud completa y una delegación a hoteleria")
        request = SolicitudViaje.model_validate(partial.model_dump())
        response = await model.ainvoke([
            SystemMessage(content=INSTRUCCIONES),
            HumanMessage(content=(
                f"Solicitud: {request.model_dump_json()}\n"
                f"Tarea delegada: {decision.instruccion}"
            )),
        ])
        calls = response.tool_calls
        if len(calls) != 1 or calls[0]["name"] != "buscar_hoteles":
            raise ValueError("El investigador debe emitir una única consulta a buscar_hoteles")
        arguments = calls[0]["args"]
        destination = arguments.get("destino")
        if not isinstance(destination, str) or normalizar_destino(destination) != normalizar_destino(request.destino):
            raise ValueError("La consulta debe conservar el destino solicitado")
        data = await buscar_hoteles.ainvoke(arguments)
        result = ResultadoHoteleria.model_validate(data)
        return {"resultado_hoteleria": result}

    return investigar
