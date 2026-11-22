"""Integración con Redis 8 descartable, habilitada mediante TEST_REDIS_URL."""

import asyncio
import os
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from config import Configuracion
from main import app, lifespan
from schemas import DecisionSupervisor
from state import EstadoViaje


@unittest.skipUnless(os.getenv("TEST_REDIS_URL"), "Requiere un Redis 8 descartable")
class RedisJobsTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrencia_checkpoints_y_recuperacion_de_huerfanos(self):
        config = Configuracion(
            openai_api_key="clave-ficticia", redis_url=os.environ["TEST_REDIS_URL"],
            job_timeout_seconds=15,
        )
        comenzaron = []
        cinco_activos = asyncio.Event()
        seguir = asyncio.Event()

        async def responder(estado):
            pregunta = estado["messages"][0].content
            comenzaron.append(pregunta)
            if len(comenzaron) == 5:
                cinco_activos.set()
            await seguir.wait()
            return {
                "decision_supervisor": DecisionSupervisor(next_agent="finalizar", instruccion="Responder"),
                "messages": [AIMessage(content=f"Resultado de {pregunta}")],
            }

        grafo = StateGraph(EstadoViaje)
        grafo.add_node("responder", responder)
        grafo.add_edge(START, "responder")
        grafo.add_edge("responder", END)
        for patcher in (
            patch("main.load_dotenv"),
            patch("main.cargar_configuracion", return_value=config),
            patch("main.crear_llm"),
            patch("main.crear_grafo", return_value=grafo),
        ):
            self.enterContext(patcher)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            async with lifespan(app):
                prompts = [f"viaje {i}" for i in range(5)]
                respuestas = await asyncio.wait_for(asyncio.gather(*(
                    client.post("/tasks", json={"prompt": prompt}) for prompt in prompts
                )), 5)
                self.assertEqual([r.status_code for r in respuestas], [202] * 5)
                ids = [r.json()["job_id"] for r in respuestas]
                self.assertEqual(len(set(ids)), 5)
                await asyncio.wait_for(cinco_activos.wait(), 5)
                for job_id in ids:
                    trabajo = (await client.get(f"/tasks/{job_id}")).json()
                    self.assertEqual(trabajo["status"], "RUNNING")
                    self.assertIn("started_at", trabajo)

                tareas = list(app.state.worker.tareas)
                seguir.set()
                await asyncio.wait_for(asyncio.gather(*tareas), 10)
                for job_id, prompt in zip(ids, prompts, strict=True):
                    trabajo = (await client.get(f"/tasks/{job_id}")).json()
                    self.assertEqual(trabajo["status"], "DONE")
                    self.assertEqual(trabajo["result"]["respuesta"], f"Resultado de {prompt}")
                    self.assertIn("finished_at", trabajo)
                    snapshot = await app.state.worker.grafo.aget_state({"configurable": {"thread_id": job_id}})
                    self.assertEqual(snapshot.values["messages"][0].content, prompt)

                pendientes = [uuid4(), uuid4()]
                for job_id in pendientes:
                    await app.state.trabajos.crear(job_id)
                reclamados = await asyncio.gather(*(
                    app.state.trabajos.transicionar(pendientes[1], desde=("PENDING",), hacia="RUNNING")
                    for _ in range(2)
                ))
                self.assertCountEqual(reclamados, [True, False])

            async with lifespan(app):
                for job_id in pendientes:
                    trabajo = await app.state.trabajos.obtener(job_id)
                    self.assertEqual(trabajo.status, "FAILED")
                    self.assertIn("reinició", trabajo.error)
                terminado = await app.state.trabajos.obtener(UUID(ids[0]))
                self.assertEqual(terminado.status, "DONE")
