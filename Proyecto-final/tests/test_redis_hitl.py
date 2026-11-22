"""Recuperación de HITL con Redis real y respuestas del modelo simuladas."""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from redis.asyncio import Redis

from config import Configuracion
from hitl import PREFIJO_RESERVA, reservar_hotel_simulado
from main import app, lifespan
from schemas import (
    Coordenadas, DecisionSupervisor, OpcionEvaluada, Periodo, PropuestaReserva,
    ResultadoAnalisis, ResultadoHoteleria, SalidaSupervisor, SolicitudViaje,
)


def crear_modelo_simulado():
    """Sustituir sólo el modelo; los agentes, herramientas y grafo siguen reales."""
    solicitud = SolicitudViaje(
        destino="Tokio", fecha_inicio="2027-02-12", fecha_fin="2027-02-20",
        cantidad_personas=4, componentes=["alojamiento"],
    )
    supervisor = MagicMock()
    supervisor.ainvoke = AsyncMock(side_effect=[
        SalidaSupervisor(
            es_turistica=True, solicitud=solicitud, solicita_reserva_simulada=True,
            decision=DecisionSupervisor(next_agent="hoteleria", instruccion="Consultar hoteles"),
        ),
        SalidaSupervisor(
            es_turistica=True,
            decision=DecisionSupervisor(next_agent="analista", instruccion="Evaluar los hoteles"),
        ),
        SalidaSupervisor(
            es_turistica=True,
            decision=DecisionSupervisor(
                next_agent="finalizar", instruccion="Responder", evaluacion_supervisor="aprobado",
            ),
            respuesta_final="Hay dos alojamientos compatibles para el viaje.",
        ),
    ])
    investigador = MagicMock()
    investigador.ainvoke = AsyncMock(return_value=AIMessage(content="", tool_calls=[{
        "name": "buscar_hoteles", "args": {"destino": "Tokio"}, "id": "consulta-hoteles",
    }]))
    analista = MagicMock()
    analista.ainvoke = AsyncMock(return_value=ResultadoAnalisis(
        sintesis_analisis="Los dos hoteles cubren las fechas y la capacidad solicitadas.",
        resultados_evaluados=[
            OpcionEvaluada(
                opcion_id=identificador, dominio="hoteleria", clasificacion="compatible",
                justificacion="Capacidad suficiente y fechas cubiertas por el catálogo.",
            )
            for identificador in ("hotel-tokio-01", "hotel-tokio-02")
        ],
    ))
    modelo = MagicMock()
    modelo.bind_tools.return_value = investigador
    modelo.with_structured_output.side_effect = (
        lambda esquema: supervisor if esquema is SalidaSupervisor else analista
    )
    return modelo, supervisor, investigador, analista


def crear_api_de_prueba():
    """Reutilizar las rutas reales con un estado de aplicación nuevo."""
    api = FastAPI()
    api.include_router(app.router)
    return api


async def esperar_worker(api):
    """Esperar las tareas ya agendadas, con límite para detectar bloqueos."""
    await asyncio.wait_for(asyncio.gather(*tuple(api.state.worker.tareas)), timeout=10)


@unittest.skipUnless(os.getenv("TEST_REDIS_URL"), "Requiere un Redis 8 descartable")
class RedisHITLTests(unittest.IsolatedAsyncioTestCase):
    async def test_pausa_y_reanudacion_tras_recrear_la_api_con_ambas_decisiones(self):
        config = Configuracion(
            openai_api_key="clave-ficticia", redis_url=os.environ["TEST_REDIS_URL"],
            job_timeout_seconds=15,
        )
        for decision in ("aprobar", "rechazar"):
            with self.subTest(decision=decision):
                modelo, supervisor, investigador, analista = crear_modelo_simulado()
                with (
                    patch("main.load_dotenv"),
                    patch("main.cargar_configuracion", return_value=config),
                    patch("main.crear_llm", return_value=modelo),
                ):
                    primera_api = crear_api_de_prueba()
                    lote_id = uuid4()
                    prompt = (
                        "Quiero viajar a Tokio del 12 al 20 de febrero de 2027. Somos cuatro personas. "
                        "Buscamos alojamiento y gestionar una reserva hotelera simulada."
                    )
                    async with lifespan(primera_api):
                        async with AsyncClient(
                            transport=ASGITransport(app=primera_api), base_url="http://test",
                        ) as cliente:
                            aceptado = await cliente.post(
                                "/tasks", json={"prompt": prompt}, headers={"X-Batch-ID": str(lote_id)},
                            )
                            self.assertEqual(aceptado.status_code, 202)
                            job_id = aceptado.json()["job_id"]
                            await esperar_worker(primera_api)
                            respuesta = await cliente.get(f"/tasks/{job_id}")
                            self.assertEqual(respuesta.status_code, 200)
                            pausado = respuesta.json()
                            self.assertEqual(pausado["status"], "WAITING_APPROVAL")
                            self.assertEqual(pausado["accion_pendiente"]["hotel_id"], "hotel-tokio-01")
                            self.assertNotIn("finished_at", pausado)
                            clave_reserva = PREFIJO_RESERVA + job_id
                            self.assertIsNone(await primera_api.state.trabajos.redis.get(clave_reserva))

                            grafo_anterior = primera_api.state.worker.grafo
                            hilo = {"configurable": {"thread_id": job_id}}
                            snapshot = await grafo_anterior.aget_state(hilo)
                            self.assertEqual(snapshot.next, ("aprobar_reserva",))
                            checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
                            solicitud = snapshot.values["solicitud_original"].model_dump(mode="json")
                            hoteles = snapshot.values["resultado_hoteleria"].model_dump(mode="json")
                            self.assertEqual(supervisor.ainvoke.await_count, 3)
                            investigador.ainvoke.assert_awaited_once()
                            analista.ainvoke.assert_awaited_once()

                    # Nuevos clientes Redis y nuevo grafo: no reutilizar app.state ni memoria del saver.
                    segunda_api = crear_api_de_prueba()
                    async with lifespan(segunda_api):
                        self.assertIsNot(segunda_api.state.worker.grafo, grafo_anterior)
                        async with AsyncClient(
                            transport=ASGITransport(app=segunda_api), base_url="http://test",
                        ) as cliente:
                            recuperado = (await cliente.get(f"/tasks/{job_id}")).json()
                            self.assertEqual(recuperado, pausado)
                            snapshot = await segunda_api.state.worker.grafo.aget_state(hilo)
                            self.assertEqual(snapshot.next, ("aprobar_reserva",))
                            self.assertEqual(snapshot.config["configurable"]["checkpoint_id"], checkpoint_id)
                            self.assertIsInstance(snapshot.values["solicitud_original"], SolicitudViaje)
                            self.assertEqual(snapshot.values["solicitud_original"].model_dump(mode="json"), solicitud)
                            resultado = snapshot.values["resultado_hoteleria"]
                            self.assertIsInstance(resultado, ResultadoHoteleria)
                            self.assertIsInstance(resultado.hoteles[0].coordenadas, Coordenadas)
                            self.assertIsInstance(resultado.hoteles[0].periodos_disponibles[0], Periodo)
                            self.assertEqual(resultado.model_dump(mode="json"), hoteles)
                            self.assertIsNone(await segunda_api.state.trabajos.redis.get(clave_reserva))

                            respuesta = await cliente.post(f"/tasks/{job_id}/approve", json={"decision": decision})
                            self.assertEqual(respuesta.status_code, 202)
                            self.assertEqual(respuesta.json()["job_id"], job_id)
                            await esperar_worker(segunda_api)
                            terminado = (await cliente.get(f"/tasks/{job_id}")).json()
                            self.assertEqual(terminado["status"], "DONE")
                            self.assertEqual(terminado["lote_id"], str(lote_id))
                            self.assertEqual(terminado["created_at"], pausado["created_at"])
                            self.assertEqual(terminado["started_at"], pausado["started_at"])
                            self.assertIn("finished_at", terminado)
                            self.assertEqual(terminado["decision_aprobacion"], decision)
                            self.assertNotIn("accion_pendiente", terminado)
                            registro = await segunda_api.state.trabajos.redis.get(clave_reserva)
                            if decision == "aprobar":
                                self.assertIn(f"SIM-{job_id}", terminado["result"]["respuesta"])
                                self.assertEqual(PropuestaReserva.model_validate_json(registro).model_dump(mode="json"),
                                                 pausado["accion_pendiente"])
                            else:
                                self.assertIn("rechazada", terminado["result"]["respuesta"])
                                self.assertIsNone(registro)

                            final = await segunda_api.state.worker.grafo.aget_state(hilo)
                            self.assertFalse(final.next)
                            self.assertEqual(final.values["messages"][0].content, prompt)
                            self.assertEqual(final.values["decision_aprobacion"], decision)
                            repetida = await cliente.post(f"/tasks/{job_id}/approve", json={"decision": decision})
                            self.assertEqual(repetida.status_code, 409)
                            self.assertEqual((await cliente.get(f"/tasks/{job_id}")).json(), terminado)
                            self.assertEqual(await segunda_api.state.trabajos.redis.get(clave_reserva), registro)
                            # La reanudación no vuelve a investigar ni consulta al modelo.
                            self.assertEqual(supervisor.ainvoke.await_count, 3)
                            investigador.ainvoke.assert_awaited_once()
                            analista.ainvoke.assert_awaited_once()

    async def test_registro_de_reserva_idempotente_y_aislado_por_trabajo(self):
        propuesta = PropuestaReserva(
            hotel_id="hotel-tokio-01", hotel="Hotel Sakura Central",
            fecha_inicio="2027-02-12", fecha_fin="2027-02-20", cantidad_personas=4,
        )
        job_id = uuid4()
        otro_id = uuid4()
        async with Redis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True) as redis:
            referencias = await asyncio.wait_for(asyncio.gather(*(
                reservar_hotel_simulado(redis, job_id, propuesta) for _ in range(5)
            )), timeout=5)
            self.assertEqual(referencias, [f"SIM-{job_id}"] * 5)
            clave = PREFIJO_RESERVA + str(job_id)
            registro = await redis.get(clave)
            self.assertEqual(PropuestaReserva.model_validate_json(registro), propuesta)
            distinta = propuesta.model_copy(update={"hotel_id": "hotel-tokio-02"})
            with self.assertRaises(ValueError):
                await reservar_hotel_simulado(redis, job_id, distinta)
            self.assertEqual(await redis.get(clave), registro)
            self.assertIsNone(await redis.get(PREFIJO_RESERVA + str(otro_id)))
            self.assertEqual(await reservar_hotel_simulado(redis, otro_id, propuesta), f"SIM-{otro_id}")
            self.assertEqual(await redis.get(clave), registro)
