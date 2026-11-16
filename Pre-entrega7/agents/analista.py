"""Análisis de compatibilidad usando investigaciones existentes y cálculos locales."""

import json
from collections.abc import Awaitable, Callable
from datetime import timedelta

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from schemas import OpcionHotel, OpcionTuristica, ResultadoAnalisis, SolicitudViaje
from state import EstadoViaje
from tools import TOKYO_STATION, calcular_distancia_km, normalizar_destino


INSTRUCCIONES = (
    "Sos el Analista. Evaluá las opciones recuperadas contra la solicitud original, "
    "usando los datos y comprobaciones adjuntos. No investigues ni inventes opciones. "
    "Clasificá cada opción con su dominio e ID original y justificá el resultado. "
    "Una incompatibilidad requiere evidencia: capacidad insuficiente, falta de cobertura "
    "temporal o restricciones conocidas que contradigan la solicitud. "
    "La falta de información necesaria implica pendiente_verificacion, no incompatible. "
    "Necesaria significa indispensable para verificar una condición expresamente solicitada, "
    "no cualquier campo ausente de la fuente. Evaluá cada opción de forma independiente: "
    "que fallen los hoteles no cambia el criterio aplicado a las mismas opciones turísticas. "
    "La comprobación capacidad_suficiente corresponde sólo al alojamiento; en turismo su valor "
    "null no indica un problema ni un cupo insuficiente. No exijas cupos o admisión para el grupo "
    "en recomendaciones generales sin una restricción conocida o una confirmación solicitada. "
    "No exijas precios si no se pidió presupuesto o comparación de costos; no exijas horarios "
    "exactos si no se pidió itinerario horario, ni reservas o acceso confirmado si no se solicitaron. "
    "Esas ausencias, por sí solas, no justifican incompatible ni pendiente_verificacion. "
    "Usá compatible cuando se satisfacen las condiciones solicitadas según la evidencia disponible; "
    "puede conservar observaciones informativas sin confirmar reservas. "
    "Contrastá preferencias sólo con información existente; explicá lo que no pueda verificarse. "
    "Considerá actividades válidas si pueden realizarse en algún momento del viaje; "
    "no exijas que una actividad esté disponible durante toda la estadía. "
    "Para gastronomía, no exijas reserva o disponibilidad de mesas si no se pidió "
    "un día y horario concretos. No conviertas esa incertidumbre, por sí sola, "
    "en incompatible ni pendiente_verificacion. "
    "La distancia es informativa: usá el cálculo recibido, no impongas umbrales. "
    "Una búsqueda fallida no equivale a ausencia de opciones en el destino. "
    "Si recibís analisis_anterior, revisalo según la corrección concreta indicada en tarea. "
    "Devolvé el análisis completo revisado, conservando las evaluaciones válidas y corrigiendo "
    "las omisiones señaladas. Utilizá únicamente las investigaciones existentes; "
    "no realices nuevas búsquedas ni modifiques los datos originales. "
    "No emitas la evaluación global del Supervisor ni una respuesta final al usuario."
)


def comprobar_opcion(
    solicitud: SolicitudViaje, opcion: OpcionHotel | OpcionTuristica
) -> dict:
    """Producir evidencia determinista; None significa que no hay datos suficientes."""
    periods = opcion.periodos_disponibles
    temporal = None
    if periods is not None:
        if isinstance(opcion, OpcionHotel):
            # Unir períodos contiguos permite cubrir la estadía con más de un tramo.
            covered_until = solicitud.fecha_inicio - timedelta(days=1)
            for period in sorted(periods, key=lambda item: item.inicio):
                if period.inicio > covered_until + timedelta(days=1):
                    break
                covered_until = max(covered_until, period.fin)
            temporal = covered_until >= solicitud.fecha_fin
        else:
            temporal = any(period.inicio <= solicitud.fecha_fin and period.fin >= solicitud.fecha_inicio
                           for period in periods)
    capacity = None
    if isinstance(opcion, OpcionHotel):
        if opcion.capacidad is not None:
            capacity = opcion.capacidad >= solicitud.cantidad_personas
        if opcion.disponibilidad == "no_disponible":
            temporal = False
        elif opcion.disponibilidad == "sin_datos":
            temporal = None
    distance = None
    # La referencia acordada corresponde a Tokio, no a otros destinos.
    if opcion.coordenadas is not None and normalizar_destino(solicitud.destino) in ("tokio", "tokyo"):
        distance = calcular_distancia_km(TOKYO_STATION, opcion.coordenadas)
    return {
        "compatibilidad_temporal": temporal,
        "capacidad_suficiente": capacity,
        "distancia_km": distance,
    }


def crear_agente_analista(llm: ChatOpenAI) -> Callable[[EstadoViaje], Awaitable[dict]]:
    """Preparar el nodo sin herramientas de investigación ni decisiones de ruteo."""
    model = llm.with_structured_output(ResultadoAnalisis)

    async def analizar(state: EstadoViaje) -> dict:
        partial = state["solicitud_original"]
        decision = state["decision_supervisor"]
        if partial is None or decision is None or decision.next_agent != "analista":
            raise ValueError("Se requiere una solicitud completa y una delegación al Analista")
        request = SolicitudViaje.model_validate(partial.model_dump())
        investigations = {}
        options = {}
        hotels = state["resultado_hoteleria"]
        tourism = state["resultado_turismo"]
        for domain, result, items in (
            ("hoteleria", hotels, hotels.hoteles if hotels is not None else []),
            ("turismo", tourism, tourism.opciones if tourism is not None else []),
        ):
            investigations[domain] = result.model_dump(mode="json") if result is not None else None
            for option in items:
                key = (domain, option.id)
                if key in options:
                    raise ValueError("La investigación contiene IDs duplicados en un dominio")
                options[key] = comprobar_opcion(request, option)
        evidence = [dict(dominio=domain, opcion_id=option_id, **facts)
                    for (domain, option_id), facts in options.items()]
        payload = {
            "solicitud": request.model_dump(mode="json"),
            "tarea": decision.instruccion,
            "investigaciones": investigations,
            "comprobaciones": evidence,
            "analisis_anterior": state["resultado_analisis"].model_dump(mode="json")
            if state["resultado_analisis"] is not None else None,
        }
        response = await model.ainvoke([
            SystemMessage(content=INSTRUCCIONES),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])
        result = ResultadoAnalisis.model_validate(response).model_copy(deep=True)
        seen = set()
        for evaluated in result.resultados_evaluados:
            key = (evaluated.dominio, evaluated.opcion_id)
            if key not in options or key in seen:
                raise ValueError("El análisis contiene una opción desconocida o repetida")
            seen.add(key)
            # La distancia publicada procede del cálculo Python, no del LLM.
            evaluated.distancia_km = options[key]["distancia_km"]
        return {"resultado_analisis": result}

    return analizar
