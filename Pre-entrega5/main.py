"""Demostración asíncrona del agente con conversaciones persistidas en SQLite."""

import asyncio
import json
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import BASE_DIR, CHECKPOINT_PATH, TRACE_PATH, Configuracion, cargar_configuracion
from graph import crear_grafo
from llm import crear_llm


async def ejecutar_demo(config: Configuracion) -> dict[str, list[BaseMessage]]:
    """Ejecutar los tres escenarios y devolver sólo los mensajes nuevos de cada uno."""
    llm = crear_llm(config)
    grafo = crear_grafo(llm)
    execution_id = uuid4().hex
    conversation = {
        "configurable": {"thread_id": f"demo-{execution_id}"},
        "recursion_limit": config.recursion_limit,
    }
    error_conversation = {
        "configurable": {"thread_id": f"error-{execution_id}"},
        "recursion_limit": config.recursion_limit,
    }
    scenarios = [
        (
            "turno_1_multi_paso",
            "¿Cuántos pedidos tuvo Ana García y cuál fue el total?",
            conversation,
        ),
        ("turno_2_memoria", "¿Y Juan Pérez?", conversation),
        (
            "prueba_error_y_aclaracion",
            "¿Cuántos pedidos tuvo el cliente Roberto Sánchez?",
            error_conversation,
        ),
    ]
    results = {}
    processed_counts = {}
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_PATH)) as checkpointer:
        app = grafo.compile(checkpointer=checkpointer)
        for name, question, invocation_config in scenarios:
            print(f"\n=== {name} ===")
            print(f"Usuario: {question}")
            result = await app.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                config=invocation_config,
            )
            print(f"Asistente: {result['messages'][-1].text}")
            thread_id = invocation_config["configurable"]["thread_id"]
            previous_count = processed_counts.get(thread_id, 0)
            results[name] = result["messages"][previous_count:]
            processed_counts[thread_id] = len(result["messages"])
    return results


def serializar_traza(messages: list[BaseMessage]) -> list[dict]:
    """Conservar mensajes y llamadas observables, sin metadatos del proveedor."""
    trace = []
    for message in messages:
        entry = {"tipo": type(message).__name__, "contenido": message.text}
        if isinstance(message, AIMessage) and message.tool_calls:
            entry["tool_calls"] = [
                {"id": call["id"], "nombre": call["name"], "argumentos": call["args"]}
                for call in message.tool_calls
            ]
        if isinstance(message, ToolMessage):
            entry["herramienta"] = message.name
            entry["tool_call_id"] = message.tool_call_id
        trace.append(entry)
    return trace


def guardar_traza(results: dict[str, list[BaseMessage]]) -> None:
    """Reemplazar la evidencia anterior únicamente cuando el JSON esté escrito."""
    trace = {name: serializar_traza(messages) for name, messages in results.items()}
    text = json.dumps(trace, ensure_ascii=False, indent=2)
    temporary_path = TRACE_PATH.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(text + "\n", encoding="utf-8")
        temporary_path.replace(TRACE_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


async def main() -> int:
    """Cargar el entorno y comunicar fallos sin traceback ni credenciales."""
    try:
        load_dotenv(BASE_DIR / ".env")
        results = await ejecutar_demo(cargar_configuracion())
        guardar_traza(results)
    except Exception as error:
        print(
            f"Demostración incompleta ({type(error).__name__}). "
            "Revisá configuración, conexión y límite de recursión."
        )
        return 1
    print(f"\nTraza guardada en {TRACE_PATH.name}.")
    print("Demostración finalizada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
