"""Demostración del orquestador con persistencia SQLite y eventos del grafo."""

import asyncio
import json
from uuid import uuid4

import aiosqlite
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel
from openai import LengthFinishReasonError

from config import BASE_DIR, CHECKPOINT_PATH, TRACE_PATH, Configuracion, cargar_configuracion
from graph import crear_grafo
from demo_refinamiento import crear_analista_con_omision, verificar_prueba
from llm import crear_llm
from state import crear_estado_inicial


ESCENARIOS = [
    ("respuesta_completa", "Quiero viajar a Tokio del 12 al 20 de febrero de 2027. "
     "Somos cuatro personas y buscamos alojamiento, actividades turísticas y gastronomía."),
    ("respuesta_parcial", "Quiero viajar a Tokio del 12 al 20 de marzo de 2027. "
     "Somos cuatro personas y buscamos alojamiento, actividades turísticas y gastronomía."),
]
NOMBRE_PRUEBA = "prueba_refinamiento_controlado"

# Tipos propios que SQLite puede reconstruir al recuperar el estado del grafo.
TIPOS_CHECKPOINT = [("schemas", name) for name in (
    "SolicitudParcial", "SolicitudViaje", "ResultadoHoteleria", "ResultadoTurismo",
    "ResultadoAnalisis", "DecisionSupervisor",
)]


def serializar(valor):
    """Convertir actualizaciones a JSON sin volcar metadatos internos de mensajes."""
    if isinstance(valor, BaseMessage):
        return {"tipo": valor.type, "contenido": valor.text, "nombre": valor.name}
    if isinstance(valor, BaseModel):
        return valor.model_dump(mode="json")
    if isinstance(valor, dict):
        return {key: serializar(value) for key, value in valor.items()}
    if isinstance(valor, list):
        return [serializar(value) for value in valor]
    return valor


def mostrar_evento(nodo: str, actualizacion: dict) -> None:
    """Mostrar decisiones, contribuciones y cierre a medida que termina cada nodo."""
    print(f"\n[{nodo}]")
    if nodo == "supervisor":
        decision = actualizacion["decision_supervisor"]
        print(f"Decisión: {decision.next_agent}. {decision.instruccion}")
        if decision.evaluacion_supervisor is not None:
            print(f"Evaluación: {decision.evaluacion_supervisor}")
        for message in actualizacion.get("messages", []):
            print(f"Respuesta: {message.text}")
    elif nodo in ("hoteleria", "turismo"):
        result = actualizacion[f"resultado_{nodo}"]
        options = result.hoteles if nodo == "hoteleria" else result.opciones
        print(f"Búsqueda: {result.estado_busqueda}. Opciones: {len(options)}")
        for option in options:
            print(f"- {option.id}: {option.nombre} (fuente: {option.fuente})")
        for observation in result.observaciones:
            print(f"- {observation.descripcion}")
    elif nodo == "analista":
        result = actualizacion["resultado_analisis"]
        print(result.sintesis_analisis)
        for option in result.resultados_evaluados:
            distance = f" Distancia: {option.distancia_km} km." if option.distancia_km is not None else ""
            print(f"- {option.dominio}/{option.opcion_id}: {option.clasificacion}. {option.justificacion}{distance}")
        for observation in result.observaciones:
            print(f"- {observation.descripcion}")


async def ejecutar_escenario(app, name: str, question: str, invocation: dict, *, capturar_error: bool = False) -> dict:
    """Registrar un recorrido; la prueba controlada conserva también ejecuciones interrumpidas."""
    print(f"\n=== {name} ===\nUsuario: {question}")
    events = []
    error = None
    try:
        async for update in app.astream(crear_estado_inicial(question), config=invocation, stream_mode="updates"):
            for node, changes in update.items():
                events.append({"nodo": node, "actualizacion": serializar(changes)})
                mostrar_evento(node, changes)
    except Exception as exception:
        if not capturar_error:
            raise
        error = type(exception).__name__
    state = {}
    try:
        snapshot = await app.aget_state(invocation)
        state = snapshot.values
        decision = state.get("decision_supervisor")
        if snapshot.next or decision is None or decision.next_agent != "finalizar":
            raise ValueError("El escenario no llegó al cierre del Supervisor")
    except Exception as exception:
        if not capturar_error:
            raise
        error = error or type(exception).__name__
    case = {"solicitud_usuario": question, "thread_id": invocation["configurable"]["thread_id"],
            "eventos": events, "estado_final": serializar(state)}
    if error:
        case["error_ejecucion"] = error
    return case


async def ejecutar_demo(config: Configuracion, *, incluir_prueba_refinamiento: bool = False) -> dict:
    """Ejecutar los casos normales y, opcionalmente, la omisión controlada aislada."""
    llm = crear_llm(config)
    graph = crear_grafo(llm)
    execution_id = uuid4().hex
    trace = {}
    async with aiosqlite.connect(str(CHECKPOINT_PATH)) as connection:
        saver = AsyncSqliteSaver(connection, serde=JsonPlusSerializer(allowed_msgpack_modules=TIPOS_CHECKPOINT))
        app = graph.compile(checkpointer=saver)
        for name, question in ESCENARIOS:
            thread_id = f"{name}-{execution_id}"
            invocation = {"configurable": {"thread_id": thread_id}, "recursion_limit": config.recursion_limit}
            trace[name] = await ejecutar_escenario(app, name, question, invocation)
        if incluir_prueba_refinamiento:
            control = {}
            # Sólo se sustituye la implementación del mismo nodo en este grafo de demostración.
            controlled_graph = crear_grafo(llm, nodo_analista=crear_analista_con_omision(llm, control))
            controlled_app = controlled_graph.compile(checkpointer=saver)
            invocation = {"configurable": {"thread_id": f"{NOMBRE_PRUEBA}-{execution_id}"},
                          "recursion_limit": config.recursion_limit}
            case = await ejecutar_escenario(controlled_app, NOMBRE_PRUEBA, ESCENARIOS[0][1],
                                           invocation, capturar_error=True)
            case["control_prueba"] = control
            case["resultado_prueba"] = verificar_prueba(case["eventos"], control, case.get("error_ejecucion"))
            trace[NOMBRE_PRUEBA] = case
    return trace


def guardar_traza(trace: dict) -> None:
    """Publicar la traza completa sin sobrescribir evidencia anterior con una parcial."""
    text = json.dumps(trace, ensure_ascii=False, indent=2)
    temporary_path = TRACE_PATH.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(text + "\n", encoding="utf-8")
        temporary_path.replace(TRACE_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


async def main() -> int:
    """Cargar configuración y comunicar fallos sin traceback ni detalles sensibles."""
    try:
        load_dotenv(BASE_DIR / ".env")
        trace = await ejecutar_demo(cargar_configuracion(), incluir_prueba_refinamiento=True)
        guardar_traza(trace)
    except LengthFinishReasonError:
        print(
            "Demostración incompleta: el modelo alcanzó el límite de salida antes de completar "
            "la respuesta estructurada. Revisá OPENAI_MAX_TOKENS. No se guardó una nueva traza."
        )
        return 1
    except Exception as error:
        print(f"Demostración incompleta ({type(error).__name__}). Revisá configuración, conexión y límite de recursión.")
        return 1
    print(f"\nTraza guardada en {TRACE_PATH.name}.")
    result = trace[NOMBRE_PRUEBA]["resultado_prueba"]
    if not result["satisfactoria"]:
        print("Prueba de refinamiento NO satisfactoria:")
        for reason in result["motivos"]:
            print(f"- {reason}")
        return 1
    print("Prueba de refinamiento satisfactoria. Demostración finalizada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
