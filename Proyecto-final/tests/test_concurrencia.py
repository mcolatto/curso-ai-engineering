"""Comprobar la carga y sus métricas sin OpenAI ni LangSmith reales."""

import asyncio
import json
import os
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import httpx
from langsmith import AsyncClient as LangSmithAsyncClient
from langsmith.utils import LangSmithConflictError
from langsmith._httpx import httpx as httpx_langsmith
from langsmith.types.trace import Trace

from scripts import prueba_concurrencia as carga

CONFIGURACION_PRUEBA = {
    "model": "modelo-prueba", "timeout_seconds": 30, "max_tokens": 4096,
    "reasoning_effort": None, "recursion_limit": 10, "job_timeout_seconds": 300,
}


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
                "status": "DONE", "result": {"respuesta": f"Viaje de prueba: {request.url.path.split('/')[-1]}"},
            })

        cliente = httpx.AsyncClient(transport=httpx.MockTransport(servidor), base_url="http://test")
        with patch.object(carga.httpx, "AsyncClient", return_value=cliente):
            mediciones = await carga.lanzar_lote("http://test", reporte)
        self.assertEqual([m.job_id for m in mediciones], ids)
        self.assertTrue(all(m.estado == "DONE" and m.aceptacion_ms is not None for m in mediciones))
        self.assertEqual([m.respuesta for m in mediciones], [f"Viaje de prueba: {job_id}" for job_id in ids])
        self.assertEqual(reporte.prompt, carga.PROMPT)

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
            prompt=carga.PROMPT,
            mediciones=[carga.Medicion(peticion=i, job_id=uuid4(), estado="DONE", respuesta=f"Respuesta {i}")
                        for i in range(1, 6)],
        )

    def traza(self, medicion, duracion, *, costo=0.025, tag=None, configuracion=CONFIGURACION_PRUEBA):
        return Trace.model_validate({
            "root_run": {
                "id": str(medicion.job_id), "start_time": self.reporte.iniciado_en,
                "end_time": self.reporte.iniciado_en + timedelta(seconds=duracion),
                "tags": [tag or f"lote:{self.reporte.lote_id}"],
                "metadata": {"configuracion_ejecucion": configuracion},
            },
            "trace_aggregates": {"total_tokens": 10, "total_cost": costo},
        })

    async def test_costo_de_cada_traza_y_un_feedback_p95_por_lote(self):
        trazas = [
            self.traza(m, segundos) for m, segundos in zip(self.reporte.mediciones, [45, 42, 39, 52, 43])
        ]
        cliente = Mock(create_feedback=AsyncMock())
        cliente.traces.query.return_value = paginar(trazas)
        await carga.reunir_metricas(cliente, self.reporte, espera_s=0, confirmaciones=1)
        self.assertEqual(self.reporte.estado, "COMPLETO")
        self.assertEqual(self.reporte.latencia_p95_s, 52)
        self.assertEqual([str(m.costo_usd) for m in self.reporte.mediciones], ["0.025"] * 5)
        self.assertEqual([m.tokens for m in self.reporte.mediciones], [10] * 5)
        self.assertEqual([m.configuracion for m in self.reporte.mediciones], [CONFIGURACION_PRUEBA] * 5)
        consulta = cliente.traces.query.call_args.kwargs
        self.assertEqual(consulta["trace_ids"], [str(m.job_id) for m in self.reporte.mediciones])
        self.assertIn("METADATA", consulta["selects"])

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
        await carga.reunir_metricas(cliente, self.reporte, espera_s=0, confirmaciones=1)
        self.assertEqual(self.reporte.estado, "INCOMPLETO")
        self.assertIsNone(self.reporte.latencia_p95_s)
        self.assertIsNone(self.reporte.mediciones[-1].costo_usd)

        cliente.traces.query.return_value = paginar([self.traza(m, 42) for m in self.reporte.mediciones[-2:]])
        await carga.reunir_metricas(cliente, self.reporte, espera_s=0, confirmaciones=1)
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
                await carga.reunir_metricas(cliente, self.reporte, espera_s=0, confirmaciones=1)
        self.assertEqual(self.reporte.estado, "COMPLETO")
        self.assertEqual(self.reporte.latencia_p95_s, 40)

    async def test_configuracion_ausente_o_distinta_no_cierra_el_lote(self):
        original = self.reporte.model_copy(deep=True)
        for configuracion in (None, {**CONFIGURACION_PRUEBA, "model": "otro-modelo"}):
            with self.subTest(configuracion=configuracion):
                self.reporte = original.model_copy(deep=True)
                cliente = Mock()
                cliente.traces.query.return_value = paginar([
                    self.traza(m, 40) for m in self.reporte.mediciones[:-1]
                ] + [self.traza(self.reporte.mediciones[-1], 40, configuracion=configuracion)])
                await carga.reunir_metricas(cliente, self.reporte, espera_s=0, confirmaciones=1)
                self.assertEqual(self.reporte.estado, "INCOMPLETO")
                self.assertIsNone(self.reporte.latencia_p95_s)

    async def test_fallos_respuestas_ausentes_o_ids_repetidos_no_completan_el_lote(self):
        original = self.reporte.model_copy(deep=True)
        for caso in ("failed", "sin_respuesta", "id_repetido"):
            with self.subTest(caso=caso):
                self.reporte = original.model_copy(deep=True)
                primera = self.reporte.mediciones[0]
                if caso == "failed":
                    primera.estado = "FAILED"
                    primera.error = "Falló el trabajo."
                elif caso == "sin_respuesta":
                    primera.respuesta = None
                else:
                    primera.job_id = self.reporte.mediciones[1].job_id
                cliente = Mock()
                cliente.traces.query.return_value = paginar([self.traza(m, 40) for m in self.reporte.mediciones])
                await carga.reunir_metricas(cliente, self.reporte, espera_s=0, confirmaciones=1)
                self.assertEqual(self.reporte.estado, "INCOMPLETO")
                self.assertIsNone(self.reporte.latencia_p95_s)

    async def test_reporte_solo_guarda_los_parametros_publicos_de_la_traza(self):
        cliente = Mock()
        parametros = {**CONFIGURACION_PRUEBA, "openai_api_key": "secreto-ficticio", "redis_url": "url-privada"}
        cliente.traces.query.return_value = paginar([
            self.traza(m, 40, configuracion=parametros) for m in self.reporte.mediciones
        ])
        await carga.reunir_metricas(cliente, self.reporte, espera_s=0, confirmaciones=1)
        self.assertEqual(self.reporte.estado, "COMPLETO")
        for medicion in self.reporte.mediciones:
            self.assertEqual(medicion.configuracion, CONFIGURACION_PRUEBA)
        self.assertNotIn("secreto-ficticio", self.reporte.model_dump_json())
        self.assertNotIn("url-privada", self.reporte.model_dump_json())

    async def test_recolectar_completa_el_archivo_sin_enviar_trabajos_ni_perder_respuestas(self):
        cliente = MagicMock()
        cliente.__aenter__ = AsyncMock(return_value=cliente)
        cliente.__aexit__ = AsyncMock(return_value=False)
        cliente.create_feedback = AsyncMock()
        cliente.traces.query.side_effect = lambda **kwargs: paginar([self.traza(m, 40) for m in self.reporte.mediciones])
        with TemporaryDirectory() as carpeta:
            destino = Path(carpeta) / "lote.json"
            carga.guardar(self.reporte, destino)
            with (
                patch.object(carga, "AsyncClient", return_value=cliente),
                patch.object(carga, "load_dotenv"),
                patch.object(carga.asyncio, "sleep", new_callable=AsyncMock),
                patch.object(carga, "lanzar_lote", new_callable=AsyncMock) as lanzar,
                patch.dict(os.environ, {"LANGSMITH_API_KEY": "clave-ficticia"}, clear=True),
                patch("builtins.print"),
            ):
                codigo = await carga.main(["--recolectar", str(destino)])
            self.assertEqual(codigo, 0)
            lanzar.assert_not_awaited()
            cliente.read_project.assert_not_called()
            guardado = carga.Reporte.model_validate_json(destino.read_text(encoding="utf-8"))
            self.assertEqual(guardado.lote_id, self.reporte.lote_id)
            self.assertEqual(guardado.estado, "COMPLETO")
            self.assertTrue(guardado.feedback_publicado)
            self.assertEqual([m.respuesta for m in guardado.mediciones], [m.respuesta for m in self.reporte.mediciones])
            self.assertTrue(all(m.observaciones_estables == 3 and m.metricas_confirmadas_en for m in guardado.mediciones))
            cliente.create_feedback.assert_awaited_once()

    async def test_no_publica_p95_si_el_reporte_esta_incompleto(self):
        cliente = Mock(create_feedback=AsyncMock())
        with self.assertRaises(ValueError):
            await carga.publicar_p95(cliente, self.reporte)
        cliente.create_feedback.assert_not_awaited()

    async def test_agregados_parciales_no_se_confunden_con_metricas_confirmadas(self):
        parciales = [self.traza(m, 40, costo=0.025) for m in self.reporte.mediciones]
        finales = [self.traza(m, 40, costo=0.05) for m in self.reporte.mediciones]
        for traza in finales:
            traza.trace_aggregates.total_tokens = 20
        cliente = Mock()
        cliente.traces.query.side_effect = [
            paginar(parciales), paginar(finales), paginar(finales), paginar(finales),
        ]
        with patch.object(carga.asyncio, "sleep", new_callable=AsyncMock):
            await carga.reunir_metricas(cliente, self.reporte)
        self.assertEqual([m.tokens for m in self.reporte.mediciones], [20] * 5)
        self.assertEqual([str(m.costo_usd) for m in self.reporte.mediciones], ["0.05"] * 5)
        self.assertEqual(cliente.traces.query.call_count, 4)

    async def test_reporte_completo_puede_refrescarse_sin_repetir_las_consultas(self):
        self.reporte.estado = "COMPLETO"
        self.reporte.latencia_p95_s = 40
        self.reporte.feedback_publicado = True
        for medicion in self.reporte.mediciones:
            medicion.latencia_traza_s = 40
            medicion.tokens = 1
        cliente = Mock()
        cliente.traces.query.return_value = paginar([self.traza(m, 40) for m in self.reporte.mediciones])
        await carga.reunir_metricas(cliente, self.reporte, espera_s=0, confirmaciones=1)
        self.assertEqual([m.tokens for m in self.reporte.mediciones], [10] * 5)
        self.assertEqual(self.reporte.estado, "COMPLETO")
        self.assertTrue(self.reporte.feedback_publicado)

    async def test_actualiza_el_mismo_feedback_si_ya_existe(self):
        self.reporte.estado = "COMPLETO"
        self.reporte.latencia_p95_s = 40
        cliente = Mock(create_feedback=AsyncMock(side_effect=LangSmithConflictError("Ya existe")))
        with patch.object(carga, "Client") as sincronico:
            await carga.publicar_p95(cliente, self.reporte)
        llamada = sincronico.return_value.update_feedback.call_args
        self.assertEqual(llamada.kwargs["score"], 40)
        self.assertEqual(llamada.args[0], cliente.create_feedback.call_args.kwargs["feedback_id"])
        sincronico.return_value.close.assert_called_once_with(timeout=5)
        self.assertTrue(self.reporte.feedback_publicado)
