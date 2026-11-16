"""Ejecutar el grafo en tareas supervisadas por el proceso de la API."""

import asyncio
import logging
from uuid import UUID

from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from redis.exceptions import RedisError

from jobs import ESTADOS_ACTIVOS, TrabajosRedis
from observability import Observabilidad, configurar_invocacion
from schemas import PropuestaReserva, ResultadoTrabajo
from state import crear_estado_inicial

logger = logging.getLogger(__name__)
MAX_TRABAJOS_ACTIVOS = 5


class Worker:
    def __init__(
        self, trabajos: TrabajosRedis, grafo: CompiledStateGraph, observabilidad: Observabilidad,
        *, recursion_limit: int, timeout_seconds: float,
    ) -> None:
        self.trabajos = trabajos
        self.grafo = grafo
        self.observabilidad = observabilidad
        self.recursion_limit = recursion_limit
        self.timeout_seconds = timeout_seconds
        self.cupos = asyncio.Semaphore(MAX_TRABAJOS_ACTIVOS)
        self.tareas: set[asyncio.Task[None]] = set()

    def agendar(self, job_id: UUID, entrada: str | Command, *, lote_id: UUID | None = None) -> None:
        """Iniciar o reanudar una tarea en segundo plano."""
        tarea = asyncio.create_task(self.ejecutar(job_id, entrada, lote_id=lote_id))
        self.tareas.add(tarea)
        tarea.add_done_callback(self.tareas.discard)

    async def ejecutar(self, job_id: UUID, entrada: str | Command, *, lote_id: UUID | None = None) -> None:
        try:
            async with self.cupos:
                inicio = await self.trabajos.transicionar(job_id, desde=("PENDING",), hacia="RUNNING")
                if not inicio:
                    return
                invocacion = configurar_invocacion(
                    job_id, self.recursion_limit,
                    es_reanudacion=isinstance(entrada, Command), lote_id=lote_id,
                )
                with self.observabilidad.contexto():
                    async with asyncio.timeout(self.timeout_seconds):
                        estado = await self.grafo.ainvoke(
                            crear_estado_inicial(entrada) if isinstance(entrada, str) else entrada,
                            config=invocacion, durability="sync",
                        )
                interrupciones = estado.get("__interrupt__", ())
                if interrupciones:
                    if len(interrupciones) != 1 or isinstance(entrada, Command):
                        raise ValueError("El trabajo sólo admite una propuesta de reserva")
                    propuesta = PropuestaReserva.model_validate(interrupciones[0].value)
                    await self.trabajos.transicionar(
                        job_id, desde=("RUNNING",), hacia="WAITING_APPROVAL", accion_pendiente=propuesta,
                    )
                    return
                decision = estado.get("decision_supervisor")
                mensajes = estado["messages"]
                if isinstance(entrada, Command) and estado.get("decision_aprobacion") != entrada.resume["decision"]:
                    raise ValueError("El grafo no aplicó la decisión sobre la propuesta")
                if (
                    decision is None or decision.next_agent != "finalizar"
                    or not mensajes or not isinstance(mensajes[-1], AIMessage)
                ):
                    raise ValueError("El grafo no produjo una respuesta final")
                resultado = ResultadoTrabajo(respuesta=mensajes[-1].text)
                await self.trabajos.transicionar(job_id, desde=("RUNNING",), hacia="DONE", resultado=resultado)
        except asyncio.CancelledError:
            await self.marcar_fallido(job_id, "La API se detuvo antes de completar el trabajo.")
            raise
        except TimeoutError:
            await self.marcar_fallido(job_id, "El trabajo superó el tiempo máximo de ejecución.")
        except Exception as error:
            logger.warning("Falló el trabajo %s (%s)", job_id, type(error).__name__)
            await self.marcar_fallido(job_id, "No se pudo completar el trabajo.")

    async def marcar_fallido(self, job_id: UUID, error: str) -> None:
        try:
            await self.trabajos.transicionar(job_id, desde=ESTADOS_ACTIVOS, hacia="FAILED", error=error)
        except RedisError:
            logger.error("Redis no pudo guardar FAILED para el trabajo %s", job_id)

    async def cerrar(self) -> None:
        tareas = list(self.tareas)
        for tarea in tareas:
            tarea.cancel()
        await asyncio.gather(*tareas, return_exceptions=True)
