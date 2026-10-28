"""Pruebas de la demostración; no comprueban grounding de un LLM real."""

import io
import json
import os
import runpy
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

import httpx2
from langchain_core.exceptions import OutputParserException
from openai import (
    APIConnectionError, APIError, APITimeoutError,
    AuthenticationError, PermissionDeniedError, RateLimitError,
)

import main as demo
from config import BASE_DIR
from schemas import RAGResponse


class MainTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        for target in ("main.load_dotenv", "main.logging.basicConfig", "main.get_rag_response"):
            patcher = patch(target)
            mock = patcher.start()
            self.addCleanup(patcher.stop)
            if target.endswith("load_dotenv"):
                self.load_dotenv = mock
            elif target.endswith("basicConfig"):
                self.logging_config = mock
            else:
                self.respond = mock

    async def ejecutar_demo(self):
        output = io.StringIO()
        with redirect_stdout(output):
            await demo.main()
        return output.getvalue()

    def respuesta_simulada(self, text="Respuesta simulada", fuentes=None):
        return RAGResponse(
            respuesta=text, fuentes=["rag.txt"] if fuentes is None else fuentes,
            fragmentos_recuperados=4,
        )

    async def test_ejecuta_seis_preguntas_en_orden_y_muestra_json_completo(self):
        questions = [
            "¿Cuáles son los beneficios de la RAG?",
            "¿Qué es la curación de datos?",
            "¿Qué son los vectores?",
            "¿Qué es Pinecone?",
            "¿Qué algoritmo utiliza Pinecone para distribuir sus índices entre servidores?",
            "¿Qué es HNSW?",
        ]
        responses = [
            self.respuesta_simulada("Beneficios simulados", ["rag.txt"]),
            self.respuesta_simulada("Definición simulada", ["datasets.txt"]),
            self.respuesta_simulada("Vectores simulados", ["vectordb.txt"]),
            *[self.respuesta_simulada("No lo sé") for _ in range(3)],
        ]
        self.respond.side_effect = responses
        output = await self.ejecutar_demo()
        self.assertEqual([call.args[0] for call in self.respond.await_args_list], questions)
        self.assertEqual(output.count("=== "), 6)
        self.assertEqual(output.count('"respuesta": "No lo sé"'), 3)
        # Validar cada JSON mostrado sin depender de la redacción del modelo.
        sections = output.split("\n=== ")[1:]
        for section, question, response in zip(sections, questions, responses):
            self.assertIn(question, section)
            body = section[section.index("{"):]
            self.assertEqual(json.loads(body), response.model_dump())
        self.load_dotenv.assert_called_once_with(BASE_DIR / ".env")
        self.logging_config.assert_called_once()

    async def test_error_de_parseo_no_impide_ejecutar_los_casos_siguientes(self):
        self.respond.side_effect = [
            OutputParserException("contenido-privado clave-ficticia"),
            *[self.respuesta_simulada() for _ in range(5)],
        ]
        output = await self.ejecutar_demo()
        self.assertEqual(self.respond.await_count, 6)
        self.assertEqual(output.count('"respuesta": "Respuesta simulada"'), 5)
        self.assertIn("no cumple el formato esperado", output)
        self.assertNotIn("contenido-privado", output)
        self.assertNotIn("clave-ficticia", output)
        self.assertNotIn("Traceback", output)

    async def test_configuracion_o_indice_invalido_se_informan_sin_exponer_detalles(self):
        self.respond.side_effect = ValueError("detalle-privado clave-ficticia")
        output = await self.ejecutar_demo()
        self.assertEqual(self.respond.await_count, 6)
        self.assertIn(".env.example", output)
        self.assertIn("ingest.py", output)
        self.assertNotIn("detalle-privado", output)
        self.assertNotIn("clave-ficticia", output)

    async def test_errores_api_se_informan_y_continuan_sin_reintentar_desde_main(self):
        request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
        cases = [
            (AuthenticationError("clave-ficticia", response=httpx2.Response(401, request=request), body=None), "autenticación"),
            (PermissionDeniedError("clave-ficticia", response=httpx2.Response(403, request=request), body=None), "permisos"),
            (RateLimitError("clave-ficticia", response=httpx2.Response(429, request=request), body=None), "cuota"),
            (APITimeoutError(request=request), "tiempo de espera"),
            (APIConnectionError(request=request), "conexión"),
            (APIError("clave-ficticia", request=request, body=None), "Error de OpenAI"),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                self.respond.reset_mock()
                self.respond.side_effect = error
                output = await self.ejecutar_demo()
                self.assertEqual(self.respond.await_count, 6)
                self.assertIn(expected, output)
                self.assertNotIn("clave-ficticia", output)
                self.assertNotIn("Traceback", output)

    def test_importar_main_no_ejecuta_consultas_ni_carga_env(self):
        with patch("chain.get_rag_response", new_callable=AsyncMock) as respond:
            runpy.run_module("main", run_name="main_import_test")
            respond.assert_not_called()
            self.load_dotenv.assert_not_called()
            self.logging_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
