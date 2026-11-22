"""Demostrar una omisión controlada sin dirigir la decisión del Supervisor."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from checkpoints import abrir_checkpointer
from config import BASE_DIR, Configuracion, cargar_configuracion
from demo_refinamiento import crear_analista_con_omision, verificar_prueba
from graph import crear_grafo
from llm import crear_llm
from observability import Observabilidad, configurar_invocacion
from state import crear_estado_inicial

PROMPT = (
    "Quiero viajar a Tokio del 12 al 20 de febrero de 2027. "
    "Somos cuatro personas y buscamos alojamiento, actividades turísticas y gastronomía."
)
ESCENARIO = "refinamiento_controlado"


def serializar(valor):
    """Conservar datos del recorrido sin volcar metadatos internos de los mensajes."""
    if isinstance(valor, BaseMessage):
        return {"tipo": valor.type, "contenido": valor.text, "nombre": valor.name}
    if isinstance(valor, BaseModel):
        return valor.model_dump(mode="json")
    if isinstance(valor, dict):
        return {clave: serializar(dato) for clave, dato in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [serializar(dato) for dato in valor]
    return valor


def guardar_traza(traza: dict, destino: Path, *, inicial: bool = False) -> None:
    """Reservar una salida nueva y reemplazarla de forma atómica al terminar."""
    contenido = json.dumps(traza, ensure_ascii=False, indent=2) + "\n"
    if inicial:
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("x", encoding="utf-8") as archivo:
            archivo.write(contenido)
        return
    temporal = destino.with_suffix(destino.suffix + ".tmp")
    try:
        temporal.write_text(contenido, encoding="utf-8")
        temporal.replace(destino)
    finally:
        temporal.unlink(missing_ok=True)


def mostrar_evento(nodo: str, actualizacion: dict) -> None:
    """Mostrar la delegación y la cobertura del análisis durante la ejecución."""
    if nodo == "supervisor":
        decision = actualizacion["decision_supervisor"]
        print(f"[supervisor] {decision.next_agent}: {decision.instruccion}", flush=True)
    elif nodo == "analista":
        cantidad = len(actualizacion["resultado_analisis"].resultados_evaluados)
        print(f"[analista] Opciones evaluadas: {cantidad}", flush=True)
    else:
        print(f"[{nodo}] Investigación finalizada.", flush=True)


async def ejecutar_prueba(config: Configuracion, destino: Path | None = None) -> tuple[dict, Path]:
    """Ejecutar sólo el escenario controlado, con Redis y trazas correlacionadas."""
    job_id = uuid4()
    destino = destino or BASE_DIR / "evidencias" / f"refinamiento-{job_id}.json"
    invocacion = configurar_invocacion(job_id, config.recursion_limit, es_reanudacion=False)
    invocacion["run_name"] = "orquestador_viajes_refinamiento_controlado"
    invocacion["tags"].append(ESCENARIO)
    invocacion["metadata"]["escenario"] = ESCENARIO
    control = {}
    traza = {
        "escenario": ESCENARIO, "job_id": str(job_id), "thread_id": str(job_id),
        "run_id": str(invocacion["run_id"]), "solicitud_usuario": PROMPT,
        "iniciado_en": datetime.now(UTC).isoformat(), "estado": "EN_EJECUCION",
        "eventos": [], "estado_final": {}, "control_prueba": control,
        "error_ejecucion": None,
    }
    # Comprobar la salida antes de crear clientes o consumir API; nunca sobrescribir otra corrida.
    await asyncio.to_thread(guardar_traza, traza, destino, inicial=True)
    observabilidad = None
    try:
        observabilidad = Observabilidad(config)
        async with abrir_checkpointer(config.redis_url.get_secret_value()) as saver:
            llm = crear_llm(config)
            analista = crear_analista_con_omision(llm, control)
            grafo = crear_grafo(llm, nodo_analista=analista).compile(checkpointer=saver)
            with observabilidad.contexto():
                try:
                    async with asyncio.timeout(config.job_timeout_seconds):
                        async for actualizacion in grafo.astream(
                            crear_estado_inicial(PROMPT), config=invocacion,
                            stream_mode="updates", durability="sync",
                        ):
                            if "__interrupt__" in actualizacion:
                                raise ValueError("La demostración no debe solicitar una reserva")
                            for nodo, cambios in actualizacion.items():
                                traza["eventos"].append({"nodo": nodo, "actualizacion": serializar(cambios)})
                                mostrar_evento(nodo, cambios)
                except Exception as error:
                    traza["error_ejecucion"] = type(error).__name__
            # Recuperar también el estado parcial si un nodo falló o se agotó el límite.
            snapshot = await grafo.aget_state(invocacion)
            traza["estado_final"] = serializar(snapshot.values)
            decision = snapshot.values.get("decision_supervisor")
            if snapshot.next or decision is None or decision.next_agent != "finalizar":
                traza["error_ejecucion"] = traza["error_ejecucion"] or "EjecucionIncompleta"
    except Exception as error:
        traza["error_ejecucion"] = traza["error_ejecucion"] or type(error).__name__
    finally:
        if observabilidad is not None:
            try:
                await observabilidad.cerrar()
            except Exception as error:
                traza["error_ejecucion"] = traza["error_ejecucion"] or type(error).__name__
        traza["resultado_prueba"] = verificar_prueba(traza["eventos"], control, traza["error_ejecucion"])
        traza["estado"] = "SATISFACTORIA" if traza["resultado_prueba"]["satisfactoria"] else "NO_SATISFACTORIA"
        traza["finalizado_en"] = datetime.now(UTC).isoformat()
        await asyncio.to_thread(guardar_traza, traza, destino)
    return traza, destino


async def main(argv: list[str] | None = None) -> int:
    """Cargar configuración y comunicar el resultado sin exponer excepciones completas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, help="Archivo JSON nuevo; se conservan también las pruebas fallidas.")
    args = parser.parse_args(argv)
    try:
        load_dotenv(BASE_DIR / ".env")
        traza, destino = await ejecutar_prueba(cargar_configuracion(), args.salida)
    except FileExistsError:
        print("La salida ya existe. Elegí otro archivo para conservar la corrida anterior.")
        return 1
    except Exception as error:
        print(f"No se pudo preparar o guardar la prueba ({type(error).__name__}).")
        return 1
    print(f"Traza guardada en {destino}. Hilo: {traza['thread_id']}.")
    print(f"Prueba {traza['estado']}.")
    resultado = traza["resultado_prueba"]
    if resultado["feedback_observado"]:
        print(f"Feedback observado: {resultado['feedback_observado']}")
    for motivo in resultado["motivos"]:
        print(f"- {motivo}")
    return 0 if resultado["satisfactoria"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
