"""Inyección de una omisión y verificación externa de la prueba controlada."""

import re
import unicodedata
from collections.abc import Awaitable, Callable

from langchain_openai import ChatOpenAI

from agents.analista import crear_agente_analista
from state import EstadoViaje

OPCION_OMITIDA = ("turismo", "turismo-tokio-03")


def crear_analista_con_omision(llm: ChatOpenAI, registro: dict) -> Callable[[EstadoViaje], Awaitable[dict]]:
    """Alterar sólo una copia del primer análisis, sin instrucciones al Supervisor."""
    analizar = crear_agente_analista(llm)
    primera = True
    registro.update({"omision_aplicada": False, "opcion_omitida": list(OPCION_OMITIDA)})

    async def analizar_con_omision(state: EstadoViaje) -> dict:
        nonlocal primera
        update = await analizar(state)
        if not primera:
            return update
        primera = False
        hotels = state["resultado_hoteleria"]
        tourism = state["resultado_turismo"]
        if hotels is None or tourism is None or hotels.estado_busqueda != "completada" or tourism.estado_busqueda != "completada":
            raise ValueError("Precondición: ambas investigaciones deben estar completadas")
        expected = {("hoteleria", item.id) for item in hotels.hoteles}
        expected.update(("turismo", item.id) for item in tourism.opciones)
        result = update["resultado_analisis"]
        evaluated = {(item.dominio, item.opcion_id) for item in result.resultados_evaluados}
        registro["opciones_esperadas"] = [list(key) for key in sorted(expected)]
        registro["cantidad_antes"] = len(result.resultados_evaluados)
        if len(expected) != 5 or OPCION_OMITIDA not in expected or evaluated != expected or len(result.resultados_evaluados) != 5:
            raise ValueError("Precondición: el primer análisis debe cubrir las cinco opciones")
        copy = result.model_copy(deep=True)
        copy.resultados_evaluados = [item for item in copy.resultados_evaluados
                                    if (item.dominio, item.opcion_id) != OPCION_OMITIDA]
        registro.update({"omision_aplicada": True, "cantidad_despues": len(copy.resultados_evaluados)})
        return {**update, "resultado_analisis": copy}

    return analizar_con_omision


def feedback_identifica_omision(texto: str) -> bool:
    """Criterio textual conservador: objetivo identificable y evaluación faltante.

    No es un juez semántico: conservar siempre la instrucción para revisión humana.
    Las formulaciones que no cumplen estas señales se consideran no satisfactorias.
    """
    normalized = "".join(character for character in unicodedata.normalize("NFD", texto.casefold())
                         if not unicodedata.combining(character))
    if any(term in normalized for term in ("no falta", "no hay omision", "sin omisiones", "no se omit")):
        return False
    target = any(term in normalized for term in (
        "turismo-tokio-03", "gastronom", "restaurante", "jardin sakura",
    ))
    coverage = bool(re.search(r"\b(cuatro|4)\b", normalized) and re.search(r"\b(cinco|5)\b", normalized))
    omission = any(term in normalized for term in (
        "omit", "falt", "no evalu", "sin evalu", "incomplet", "complet", "incorpor", "inclu", "agreg",
    ))
    evaluation = any(term in normalized for term in ("eval", "analis", "clasific", "justific"))
    return (target or coverage) and omission and evaluation


def verificar_prueba(eventos: list[dict], registro: dict, error: str | None = None) -> dict:
    """Evaluar el recorrido observado; nunca decidir ni corregir rutas del grafo."""
    failures = []
    if not registro.get("omision_aplicada"):
        failures.append("No se pudo aplicar la omisión bajo las precondiciones acordadas")
    if error:
        failures.append(f"La ejecución se interrumpió ({error})")
    names = [event["nodo"] for event in eventos]
    analyses = [index for index, name in enumerate(names) if name == "analista"]
    feedback = None
    if len(analyses) != 2:
        failures.append("No se observaron exactamente dos intervenciones del Analista")
    if analyses:
        following = eventos[analyses[0] + 1:]
        if following and following[0]["nodo"] == "supervisor":
            decision = following[0]["actualizacion"]["decision_supervisor"]
            if decision["next_agent"] == "analista":
                feedback = decision["instruccion"]
        if feedback is None:
            failures.append("El Supervisor no solicitó refinamiento después de la omisión")
        elif not feedback_identifica_omision(feedback):
            failures.append("El feedback no identifica concretamente la evaluación omitida")
    for researcher in ("hoteleria", "turismo"):
        if names.count(researcher) != 1:
            failures.append(f"La investigación de {researcher} no ocurrió exactamente una vez")
    if len(analyses) == 2:
        first = eventos[analyses[0]]["actualizacion"]["resultado_analisis"]["resultados_evaluados"]
        second = eventos[analyses[1]]["actualizacion"]["resultado_analisis"]["resultados_evaluados"]
        expected = {tuple(key) for key in registro.get("opciones_esperadas", [])}
        if len(first) != 4 or len(second) != 5 or {(item["dominio"], item["opcion_id"]) for item in second} != expected:
            failures.append("El análisis revisado no recuperó la cobertura de las cinco opciones")
    supervisors = [event["actualizacion"] for event in eventos if event["nodo"] == "supervisor"]
    refinements = [update for update in supervisors if "refinamientos_analista" in update]
    if len(refinements) != 1 or refinements[0]["refinamientos_analista"] != 1:
        failures.append("No se registró una única devolución al Analista")
    if not eventos or eventos[-1]["nodo"] != "supervisor" or supervisors[-1]["decision_supervisor"]["next_agent"] != "finalizar":
        failures.append("No se observó el cierre final del Supervisor")
    return {"satisfactoria": not failures, "motivos": failures, "feedback_observado": feedback,
            "criterio_feedback": "Objetivo gastronómico o cobertura de cuatro/cinco opciones y evaluación faltante explícitos; revisión textual conservadora"}
