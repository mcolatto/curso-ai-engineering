"""Contratos HTTP y ejecución en segundo plano sin servicios externos."""

import asyncio
import unittest
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.types import Interrupt
from redis.exceptions import ConnectionError

from config import Configuracion
from main import app, lifespan
from schemas import DecisionSupervisor, Trabajo


class AlmacenDePrueba:
    """Doble de almacenamiento para observar el avance a través de la API."""

    def __init__(self):
        self.registros = {}
        self.termino = asyncio.Event()

    async def recuperar_huerfanos(self):
        pass

    async def crear(self, job_id, *, lote_id=None):
        self.registros[job_id] = Trabajo(
            job_id=job_id, lote_id=lote_id, status="PENDING", created_at=datetime.now(UTC),
        )

    async def obtener(self, job_id):
        return self.registros.get(job_id)

    async def transicionar(
        self, job_id, *, desde, hacia, resultado=None, error=None,
        accion_pendiente=None, decision_aprobacion=None,
    ):
        trabajo = self.registros[job_id]
        if trabajo.status not in desde:
            return False
        trabajo.status = hacia
        trabajo.result = resultado
        trabajo.error = error
        trabajo.accion_pendiente = accion_pendiente
        if decision_aprobacion is not None:
            trabajo.decision_aprobacion = decision_aprobacion
        if hacia in ("DONE", "FAILED", "WAITING_APPROVAL"):
            self.termino.set()
        return True


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.trabajos = AlmacenDePrueba()
        self.grafo = AsyncMock()

        @asynccontextmanager
        async def recurso(*args, **kwargs):
            yield object()

        grafo_builder = Mock()
        grafo_builder.compile.return_value = self.grafo
        config = Configuracion(openai_api_key="clave-ficticia", job_timeout_seconds=2)
        for patcher in (
            patch("main.load_dotenv"),
            patch("main.cargar_configuracion", return_value=config),
            patch("main.Redis.from_url", side_effect=recurso),
            patch("main.abrir_checkpointer", side_effect=recurso),
            patch("main.TrabajosRedis", return_value=self.trabajos),
            patch("main.crear_llm"),
            patch("main.crear_grafo", return_value=grafo_builder),
        ):
            self.enterContext(patcher)

        await self.enterAsyncContext(lifespan(app))
        self.client = await self.enterAsyncContext(
            AsyncClient(transport=ASGITransport(app=app), base_url="http://test"),
        )

    async def enviar(self, *, lote_id=None):
        headers = {"X-Batch-ID": str(lote_id)} if lote_id else {}
        response = await asyncio.wait_for(
            self.client.post("/tasks", json={"prompt": "Viaje a Tokio"}, headers=headers), 1,
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "PENDING")
        return response.json()["job_id"]

    async def test_acepta_y_permite_polling_antes_de_terminar_el_grafo(self):
        inicio = asyncio.Event()
        seguir = asyncio.Event()

        async def ejecutar(*args, **kwargs):
            inicio.set()
            await seguir.wait()
            return {
                "decision_supervisor": DecisionSupervisor(next_agent="finalizar", instruccion="Responder"),
                "messages": [AIMessage(content="Opciones para Tokio.")],
            }

        self.grafo.ainvoke.side_effect = ejecutar
        lote_id = uuid4()
        job_id = await self.enviar(lote_id=lote_id)
        await asyncio.wait_for(inicio.wait(), 1)
        response = await asyncio.wait_for(self.client.get(f"/tasks/{job_id}"), 1)
        self.assertEqual(response.json()["status"], "RUNNING")
        self.assertEqual(response.json()["lote_id"], str(lote_id))
        seguir.set()
        await asyncio.wait_for(self.trabajos.termino.wait(), 1)
        response = await self.client.get(f"/tasks/{job_id}")
        self.assertEqual(response.json()["status"], "DONE")
        self.assertEqual(response.json()["result"], {"respuesta": "Opciones para Tokio."})
        llamada = self.grafo.ainvoke.await_args
        self.assertEqual(llamada.kwargs["config"]["configurable"]["thread_id"], job_id)
        self.assertEqual(llamada.kwargs["config"]["metadata"]["lote_id"], str(lote_id))
        self.assertIn(f"lote:{lote_id}", llamada.kwargs["config"]["tags"])
        self.assertEqual(llamada.args[0]["messages"][0].content, "Viaje a Tokio")

    async def test_error_del_grafo_se_persiste_sin_detalles_sensibles(self):
        self.grafo.ainvoke.side_effect = RuntimeError("dato sensible ficticio")
        with self.assertLogs("worker", level="WARNING") as logs:
            job_id = await self.enviar()
            await asyncio.wait_for(self.trabajos.termino.wait(), 1)
        response = await self.client.get(f"/tasks/{job_id}")
        self.assertEqual(response.json()["status"], "FAILED")
        self.assertEqual(response.json()["error"], "No se pudo completar el trabajo.")
        self.assertNotIn("dato sensible ficticio", response.text + str(logs.output))

    async def test_timeout_se_persiste_como_failed(self):
        app.state.worker.timeout_seconds = 0.01

        async def ejecutar(*args, **kwargs):
            await asyncio.Event().wait()

        self.grafo.ainvoke.side_effect = ejecutar
        job_id = await self.enviar()
        await asyncio.wait_for(self.trabajos.termino.wait(), 1)
        response = await self.client.get(f"/tasks/{job_id}")
        self.assertEqual(response.json()["status"], "FAILED")
        self.assertIn("tiempo máximo", response.json()["error"])

    async def test_404_y_422_no_agendan_trabajos(self):
        self.assertEqual((await self.client.get(f"/tasks/{uuid4()}")).status_code, 404)
        self.assertEqual((await self.client.get("/tasks/id-invalido")).status_code, 422)
        for body in ({}, {"prompt": " \n "}, {"prompt": "Viaje", "solicitar_reserva_simulada": True}):
            with self.subTest(body=body):
                self.assertEqual((await self.client.post("/tasks", json=body)).status_code, 422)
        self.assertEqual(self.trabajos.registros, {})
        self.grafo.ainvoke.assert_not_awaited()

    async def test_redis_inaccesible_devuelve_503_sin_agendar(self):
        error = ConnectionError("rediss://:secreto-ficticio@redis")
        with patch.object(self.trabajos, "crear", side_effect=error):
            crear = await self.client.post("/tasks", json={"prompt": "Viaje a Tokio"})
        with patch.object(self.trabajos, "obtener", side_effect=error):
            consultar = await self.client.get(f"/tasks/{uuid4()}")
        for response in (crear, consultar):
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("secreto-ficticio", response.text)
        self.grafo.ainvoke.assert_not_awaited()

    async def test_aprobacion_reanuda_el_mismo_trabajo_y_las_decisiones_duplicadas_dan_409(self):
        propuesta = {
            "accion": "reserva_hotelera_simulada", "hotel_id": "hotel-tokio-01",
            "hotel": "Hotel Sakura Central", "fecha_inicio": "2027-02-12",
            "fecha_fin": "2027-02-20", "cantidad_personas": 4,
        }
        self.grafo.ainvoke.side_effect = [
            {"__interrupt__": [Interrupt(value=propuesta, id="prueba")]},
            {
                "decision_supervisor": DecisionSupervisor(next_agent="finalizar", instruccion="Responder"),
                "messages": [AIMessage(content="Resultado de la reserva.")],
                "decision_aprobacion": "aprobar",
            },
        ]
        job_id = await self.enviar()
        url = f"/tasks/{job_id}"
        await asyncio.wait_for(self.trabajos.termino.wait(), 1)
        esperando = (await self.client.get(url)).json()
        self.assertEqual(esperando["status"], "WAITING_APPROVAL")
        self.assertEqual(esperando["accion_pendiente"], propuesta)
        self.assertNotIn("result", esperando)
        invalida = await self.client.post(f"{url}/approve", json={"decision": "tal_vez"})
        self.assertEqual(invalida.status_code, 422)
        self.assertEqual((await self.client.get(url)).json()["status"], "WAITING_APPROVAL")

        self.trabajos.termino.clear()
        primera, segunda = await asyncio.gather(*(
            self.client.post(f"{url}/approve", json={"decision": "aprobar"}) for _ in range(2)
        ))
        self.assertCountEqual([primera.status_code, segunda.status_code], [202, 409])
        await asyncio.wait_for(self.trabajos.termino.wait(), 1)
        final = (await self.client.get(url)).json()
        self.assertEqual(final["status"], "DONE")
        self.assertEqual(final["decision_aprobacion"], "aprobar")
        self.assertNotIn("accion_pendiente", final)
        self.assertEqual(self.grafo.ainvoke.await_count, 2)
        for llamada in self.grafo.ainvoke.await_args_list:
            self.assertEqual(llamada.kwargs["config"]["configurable"]["thread_id"], job_id)
        self.assertEqual(self.grafo.ainvoke.await_args.args[0].resume, {
            "decision": "aprobar", "propuesta": propuesta,
        })

    async def test_aprobacion_de_id_inexistente_da_404(self):
        response = await self.client.post(f"/tasks/{uuid4()}/approve", json={"decision": "rechazar"})
        self.assertEqual(response.status_code, 404)
        self.grafo.ainvoke.assert_not_awaited()
