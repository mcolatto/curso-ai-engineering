"""Comprobar la carga y sus métricas sin OpenAI ni LangSmith reales."""

import asyncio
import json
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import httpx
from langsmith import AsyncClient as LangSmithAsyncClient
from langsmith._httpx import httpx as httpx_langsmith
from langsmith.types.trace import Trace

from scripts import prueba_concurrencia as carga


class CargaHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_cinco_post_solapados_y_un_solo_lote(self):
        reporte = carga.Reporte(lote_id=uuid4(), proyecto_id=uuid4(), iniciado_en=datetime.now(UTC))
        solicitudes = []
        llegaron_cinco = asyncio.Event()
        ids = [uuid4() for _ in range(5)]

        async def servidor(request):
            if request.method == "POST":
                self.assertEqual(request.headers["X-Batch-ID"], str(reporte.lote_id))
                self.assertEqual(json.loads(request.content), {"prompt": carga.PROMPT})
                indice = len(solicitudes)
                solicitudes.append(request)
                if len(solicitudes) == 5:
                    llegaron_cinco.set()
                await asyncio.wait_for(llegaron_cinco.wait(), 1)
                return httpx.Response(202, json={"job_id": str(ids[indice]), "status": "PENDING"})
            if request.url.path == "/openapi.json":
                return httpx.Response(200, json={"paths": {"/tasks": {"post": {
                    "parameters": [{"in": "header", "name": "X-Batch-ID"}],
                }}}})
            return httpx.Response(200, json={
                "job_id": request.url.path.split("/")[-1], "lote_id": str(reporte.lote_id),
                "status": "DONE", "result": {"respuesta": "Viaje de prueba"},
            })

        cliente = httpx.AsyncClient(transport=httpx.MockTransport(servidor), base_url="http://test")
        with patch.object(carga.httpx, "AsyncClient", return_value=cliente):
            mediciones = await carga.lanzar_lote("http://test", reporte)
        self.assertEqual([m.job_id for m in mediciones], ids)
        self.assertTrue(all(m.estado == "DONE" and m.aceptacion_ms is not None for m in mediciones))

    async def test_api_anterior_no_lanza_trabajos(self):
        reporte = carga.Reporte(lote_id=uuid4(), proyecto_id=uuid4(), iniciado_en=datetime.now(UTC))
        peticiones = []

        def servidor(request):
            peticiones.append(request.method)
            return httpx.Response(200, json={"paths": {"/tasks": {"post": {}}}})

        cliente = httpx.AsyncClient(transport=httpx.MockTransport(servidor), base_url="http://test")
        with patch.object(carga.httpx, "AsyncClient", return_value=cliente), self.assertRaisesRegex(
            ValueError, "Reconstruí",
        ):
            await carga.lanzar_lote("http://test", reporte)
        self.assertEqual(peticiones, ["GET"])


async def paginar(trazas):
    for traza in trazas:
        yield traza


class MetricasTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.reporte = carga.Reporte(
            lote_id=uuid4(), proyecto_id=uuid4(), iniciado_en=datetime.now(UTC),
            mediciones=[carga.Medicion(peticion=i, job_id=uuid4(), estado="DONE") for i in range(1, 6)],
        )

    def traza(self, medicion, duracion, *, costo=0.025, tag=None):
        return Trace.model_validate({
            "root_run": {
                "id": str(medicion.job_id), "start_time": self.reporte.iniciado_en,
                "end_time": self.reporte.iniciado_en + timedelta(seconds=duracion),
                "tags": [tag or f"lote:{self.reporte.lote_id}"],
            },
            "trace_aggregates": {"total_tokens": 10, "total_cost": costo},
        })

    async def test_costo_de_cada_traza_y_un_feedback_p95_por_lote(self):
        trazas = [
            self.traza(m, segundos) for m, segundos in zip(self.reporte.mediciones, [45, 42, 39, 52, 43])
        ]
        cliente = Mock(create_feedback=AsyncMock())
        cliente.traces.query.return_value = paginar(trazas)
        await carga.reunir_metricas(cliente, self.reporte, espera_s=0)
        self.assertEqual(self.reporte.estado, "COMPLETO")
        self.assertEqual(self.reporte.latencia_p95_s, 52)
        self.assertEqual([str(m.costo_usd) for m in self.reporte.mediciones], ["0.025"] * 5)
        self.assertEqual([m.tokens for m in self.reporte.mediciones], [10] * 5)
        consulta = cliente.traces.query.call_args.kwargs
        self.assertEqual(consulta["trace_ids"], [str(m.job_id) for m in self.reporte.mediciones])

        await carga.publicar_p95(cliente, self.reporte)
        cliente.create_feedback.assert_awaited_once()
        feedback = cliente.create_feedback.call_args.kwargs
        self.assertEqual(feedback["key"], "latencia_p95_s")
        self.assertEqual(feedback["score"], 52)
        self.assertTrue(self.reporte.feedback_publicado)

    async def test_traza_sin_costo_o_de_otro_lote_no_cierra_el_reporte(self):
        cliente = Mock()
        cliente.traces.query.return_value = paginar([
            self.traza(m, 42) for m in self.reporte.mediciones[:3]
        ] + [
            self.traza(self.reporte.mediciones[3], 42, costo=None),
            self.traza(self.reporte.mediciones[4], 42, tag="lote:otro"),
        ])
        await carga.reunir_metricas(cliente, self.reporte, espera_s=0)
        self.assertEqual(self.reporte.estado, "INCOMPLETO")
        self.assertIsNone(self.reporte.latencia_p95_s)
        self.assertIsNone(self.reporte.mediciones[-1].costo_usd)

        cliente.traces.query.return_value = paginar([self.traza(m, 42) for m in self.reporte.mediciones[-2:]])
        await carga.reunir_metricas(cliente, self.reporte, espera_s=0)
        self.assertEqual(self.reporte.estado, "COMPLETO")
        self.assertEqual(len(cliente.traces.query.call_args.kwargs["trace_ids"]), 2)

    async def test_recolecta_con_el_paginador_del_sdk_real(self):
        datos = [self.traza(m, 40).model_dump(mode="json") for m in self.reporte.mediciones]

        async def responder(http, request, **kwargs):
            if request.url.path.endswith("/info"):
                return httpx_langsmith.Response(200, request=request, json={"version": "0.18.0"})
            self.assertTrue(request.url.path.endswith("/api/v2/traces/query"), request.url.path)
            consulta = json.loads(request.content)
            self.assertEqual(consulta["trace_ids"], [str(m.job_id) for m in self.reporte.mediciones])
            return httpx_langsmith.Response(200, request=request, json={"items": datos})

        with patch.object(httpx_langsmith.AsyncClient, "send", new=responder):
            async with LangSmithAsyncClient(
                api_url="https://langsmith.test", api_key="ficticia", disable_prompt_cache=True,
            ) as cliente:
                await carga.reunir_metricas(cliente, self.reporte, espera_s=0)
        self.assertEqual(self.reporte.estado, "COMPLETO")
        self.assertEqual(self.reporte.latencia_p95_s, 40)
