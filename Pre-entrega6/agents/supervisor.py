"""Delegación dinámica y validación de suficiencia con una revisión del análisis."""

import json
from collections.abc import Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from schemas import DecisionSupervisor, SalidaSupervisor, SolicitudParcial, SolicitudViaje
from state import EstadoViaje

MAX_REFINAMIENTOS_ANALISTA = 1


INSTRUCCIONES = (
    "Sos el Supervisor de un asistente turístico. Interpretá el pedido inicial sin inventar "
    "destino, fechas, viajeros ni preferencias. Indicá es_turistica=false para pedidos ajenos al dominio. "
    "Extraé solicitud en la primera intervención; después usá la solicitud ya guardada. "
    "Elegí dinámicamente entre las rutas permitidas, con una instrucción concreta al especialista. "
    "Hotelería investiga alojamiento; Turismo investiga actividades y gastronomía. Podés elegir "
    "cualquiera de las investigaciones pendientes primero. No repitas investigaciones completadas "
    "ni fallidas. Una búsqueda vacía o fallida no bloquea las otras investigaciones solicitadas. "
    "Cuando terminen, delegá al Analista. Después de su primer resultado podés finalizar "
    "o devolverlo una única vez al Analista si presenta una omisión o un defecto concreto. "
    "Para devolverlo, elegí analista e indicá en instruccion qué opción omitió o qué evaluación "
    "debe corregir o justificar, usando sólo la solicitud y las investigaciones existentes. "
    "No pidas nuevas búsquedas ni repitas investigadores. Una limitación de la fuente, "
    "como hoteles sin fechas compatibles, no justifica por sí sola refinar el análisis. "
    "No refines un análisis suficiente sólo para utilizar la revisión disponible. "
    "Cuando refinamientos_analista alcance el máximo permitido, debés finalizar; "
    "si persisten omisiones relevantes, comunicá las limitaciones y no emitas aprobado. "
    "Las instrucciones delegadas deben limitarse a las condiciones de la solicitud original: "
    "no agregues requisitos ni anticipes clasificaciones de opciones. No exijas precios, horarios "
    "exactos, cupos o reservas que no se hayan solicitado. La capacidad del alojamiento sí debe "
    "cubrir al grupo; no traslades ese requisito automáticamente a actividades o gastronomía. "
    "Antes de finalizar aplicá esta rúbrica: aprobado si los componentes solicitados están resueltos "
    "y el análisis es suficiente, sin advertencias relevantes; aprobado_con_advertencias si hay "
    "información útil pero componentes sin resolver, omisiones relevantes, incertidumbres "
    "que impiden verificar condiciones solicitadas o fallos; desaprobado "
    "si no hay evidencia suficientemente útil o confiable. Examiná la cobertura de opciones del análisis, "
    "las clasificaciones y sus justificaciones. No confundas falta de datos con incompatibilidad. "
    "Las aclaraciones informativas sobre precios, horarios, cupos o reservas no solicitados "
    "no impiden aprobado. Si se cubren alojamiento, actividades y gastronomía solicitados, "
    "con análisis suficiente y sin incumplimientos relevantes, corresponde aprobado; "
    "no rebajes la evaluación sólo porque no se confirman reservas. "
    "Conservá recomendaciones válidas aunque otro componente no tenga opciones compatibles. "
    "La ausencia de hoteles en estas fuentes no implica ausencia de hoteles en todo el destino. "
    "No confirmes reservas. "
    "No exijas reservas gastronómicas sin día y horario concretos ni rechaces por distancia. "
    "No presentes opciones incompatibles como recomendaciones válidas. "
    "Al finalizar devolvé respuesta_final y la evaluación cualitativa en sus campos separados. "
    "respuesta_final está destinada al viajero: incluí recomendaciones y limitaciones relevantes "
    "en lenguaje natural. No incluyas la calificación interna (aprobado, aprobado_con_advertencias "
    "o desaprobado, tampoco sus variantes con espacios), nombres de archivos JSON, rutas de archivos, "
    "IDs técnicos ni nombres de campos del estado. Conservá esos metadatos en los campos internos. "
    "Podés decir 'según la información consultada' sin enumerar archivos ni agregar secciones "
    "de evaluación técnica. Para solicitudes incompletas "
    "o ajenas al dominio, finalizá sin delegar ni evaluar una investigación inexistente."
)


def rutas_permitidas(state: EstadoViaje, solicitud: SolicitudViaje) -> list[str]:
    """Restringir rutas por trabajo pendiente, sin imponer orden a los investigadores."""
    pending = []
    if "alojamiento" in solicitud.componentes and state["resultado_hoteleria"] is None:
        pending.append("hoteleria")
    if any(item in solicitud.componentes for item in ("actividades", "gastronomia")):
        if state["resultado_turismo"] is None:
            pending.append("turismo")
    if pending:
        return pending
    if state["resultado_analisis"] is None:
        return ["analista"]
    if state.get("refinamientos_analista", 0) < MAX_REFINAMIENTOS_ANALISTA:
        return ["analista", "finalizar"]
    return ["finalizar"]


def crear_supervisor(llm: ChatOpenAI) -> Callable[[EstadoViaje], Awaitable[dict]]:
    """Preparar decisiones estructuradas sin realizar solicitudes al construir el nodo."""
    model = llm.with_structured_output(SalidaSupervisor)

    async def supervisar(state: EstadoViaje) -> dict:
        stored = state["solicitud_original"]
        payload = {
            "mensajes": [{"tipo": message.type, "contenido": message.text} for message in state["messages"]],
            "solicitud": stored.model_dump(mode="json") if stored is not None else None,
            "rutas_permitidas": rutas_permitidas(state, stored)
            if isinstance(stored, SolicitudViaje) else None,
            "refinamientos_analista": state.get("refinamientos_analista", 0),
            "max_refinamientos_analista": MAX_REFINAMIENTOS_ANALISTA,
        }
        for field in ("resultado_hoteleria", "resultado_turismo", "resultado_analisis", "decision_supervisor"):
            value = state[field]
            payload[field] = value.model_dump(mode="json") if value is not None else None
        response = await model.ainvoke([
            SystemMessage(content=INSTRUCCIONES),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])
        output = SalidaSupervisor.model_validate(response)
        # La extracción se guarda una vez: no sustituir el pedido original entre nodos.
        request = stored if stored is not None else output.solicitud
        if not output.es_turistica:
            if stored is not None:
                raise ValueError("No se puede reclasificar una solicitud turística ya iniciada")
            if output.decision.next_agent != "finalizar" or output.respuesta_final is None:
                raise ValueError("Una consulta ajena al dominio debe finalizar con una respuesta")
            decision = output.decision.model_copy(update={"evaluacion_supervisor": None})
            return {"decision_supervisor": decision, "messages": [AIMessage(content=output.respuesta_final, name="supervisor")]}
        if request is None:
            request = SolicitudParcial()
        missing = request.datos_faltantes()
        if not request.componentes:
            missing.append("componentes solicitados (alojamiento, actividades o gastronomía)")
        if missing:
            decision = DecisionSupervisor(next_agent="finalizar", instruccion="Informar datos faltantes")
            return {
                "solicitud_original": request, "decision_supervisor": decision,
                "messages": [AIMessage(content="Para procesar la solicitud faltan: " + ", ".join(missing) + ".", name="supervisor")],
            }
        request = SolicitudViaje.model_validate(request.model_dump())
        allowed = rutas_permitidas(state, request)
        decision = output.decision
        if decision.next_agent not in allowed:
            raise ValueError("La delegación no corresponde al trabajo pendiente del estado")
        update = {"solicitud_original": request, "decision_supervisor": decision}
        if decision.next_agent == "finalizar":
            if output.respuesta_final is None or decision.evaluacion_supervisor is None:
                raise ValueError("El cierre requiere respuesta y evaluación del Supervisor")
            update["messages"] = [AIMessage(content=output.respuesta_final, name="supervisor")]
        elif output.respuesta_final is not None or decision.evaluacion_supervisor is not None:
            raise ValueError("La respuesta final y la evaluación corresponden únicamente al cierre")
        elif decision.next_agent == "analista" and state["resultado_analisis"] is not None:
            update["refinamientos_analista"] = state.get("refinamientos_analista", 0) + 1
        return update

    return supervisar
