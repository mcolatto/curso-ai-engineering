"""API HTTP del orquestador de viajes."""

from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated
from uuid import UUID, uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from langgraph.types import Command
from redis.asyncio import Redis
from redis.exceptions import RedisError

from checkpoints import abrir_checkpointer
from config import BASE_DIR, cargar_configuracion
from graph import crear_grafo
from hitl import reservar_hotel_simulado
from jobs import TrabajosRedis
from llm import crear_llm
from observability import Observabilidad
from schemas import AprobarTrabajo, CrearTrabajo, RespuestaAprobacion, Trabajo, TrabajoAceptado
from worker import Worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Abrir los recursos compartidos y cerrarlos después de las tareas."""
    load_dotenv(BASE_DIR / ".env")
    config = cargar_configuracion()
    url = config.redis_url.get_secret_value()
    try:
        async with (
            Redis.from_url(url, decode_responses=True, socket_connect_timeout=5, socket_timeout=5) as redis,
            abrir_checkpointer(url) as checkpointer,
        ):
            trabajos = TrabajosRedis(redis)
            await trabajos.recuperar_huerfanos()
            registrar = partial(reservar_hotel_simulado, redis)
            grafo = crear_grafo(crear_llm(config), registrar_reserva=registrar).compile(checkpointer=checkpointer)
            observabilidad = Observabilidad(config)
            worker = Worker(
                trabajos, grafo, observabilidad, recursion_limit=config.recursion_limit,
                timeout_seconds=config.job_timeout_seconds,
            )
            app.state.trabajos = trabajos
            app.state.worker = worker
            try:
                yield
            finally:
                await worker.cerrar()
                await observabilidad.cerrar()
    except RedisError:
        raise RuntimeError("No se pudo acceder a Redis durante el ciclo de vida de la API.") from None


app = FastAPI(title="Orquestador de viajes", lifespan=lifespan)


@app.post("/tasks", status_code=202)
async def crear_trabajo(
    pedido: CrearTrabajo, request: Request,
    lote_id: Annotated[UUID | None, Header(alias="X-Batch-ID")] = None,
) -> TrabajoAceptado:
    job_id = uuid4()
    try:
        await request.app.state.trabajos.crear(job_id, lote_id=lote_id)
    except RedisError:
        raise HTTPException(status_code=503, detail="Redis no está disponible.") from None
    request.app.state.worker.agendar(job_id, pedido.prompt, lote_id=lote_id)
    return TrabajoAceptado(job_id=job_id)


@app.get("/tasks/{job_id}", response_model_exclude_none=True)
async def consultar_trabajo(job_id: UUID, request: Request) -> Trabajo:
    try:
        trabajo = await request.app.state.trabajos.obtener(job_id)
    except RedisError:
        raise HTTPException(status_code=503, detail="Redis no está disponible.") from None
    if trabajo is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
    return trabajo


@app.post("/tasks/{job_id}/approve", status_code=202)
async def decidir_reserva(job_id: UUID, pedido: AprobarTrabajo, request: Request) -> TrabajoAceptado:
    trabajos = request.app.state.trabajos
    try:
        trabajo = await trabajos.obtener(job_id)
        if trabajo is None:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
        if trabajo.status != "WAITING_APPROVAL" or trabajo.accion_pendiente is None:
            raise HTTPException(status_code=409, detail="El trabajo no está esperando aprobación.")
        respuesta = RespuestaAprobacion(decision=pedido.decision, propuesta=trabajo.accion_pendiente)
        reanudacion = Command(resume=respuesta.model_dump(mode="json"))
        aceptada = await trabajos.transicionar(
            job_id, desde=("WAITING_APPROVAL",), hacia="PENDING", decision_aprobacion=pedido.decision,
        )
    except RedisError:
        raise HTTPException(status_code=503, detail="Redis no está disponible.") from None
    if not aceptada:
        raise HTTPException(status_code=409, detail="El trabajo no está esperando aprobación.")
    request.app.state.worker.agendar(job_id, reanudacion, lote_id=trabajo.lote_id)
    return TrabajoAceptado(job_id=job_id)
