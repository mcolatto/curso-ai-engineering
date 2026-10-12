"""Pruebas del cliente OpenAI sin solicitudes de red."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from llmclient.providers.openai_client import OpenAIClient
from llmclient.schemas import ChatMessage, Provider


class OpenAIClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sdk_patch = patch("llmclient.providers.openai_client.AsyncOpenAI")
        self.sdk_constructor = self.sdk_patch.start()
        self.addCleanup(self.sdk_patch.stop)
        self.create = AsyncMock()
        self.sdk_constructor.return_value.chat.completions.create = self.create
        self.client = OpenAIClient("clave-ficticia", "modelo-de-prueba", 0.7, 200)
        self.messages = [ChatMessage(role="user", content="Hola")]

    def errores_simulados(self):
        request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
        return [
            AuthenticationError(
                "clave-ficticia", response=httpx2.Response(401, request=request), body=None
            ),
            RateLimitError(
                "clave-ficticia", response=httpx2.Response(429, request=request), body=None
            ),
            APITimeoutError(request=request),
            APIConnectionError(request=request),
            APIError("clave-ficticia", request=request, body=None),
        ]

    def test_configura_timeout_y_deshabilita_reintentos(self):
        self.sdk_constructor.assert_called_once_with(
            api_key="clave-ficticia", timeout=30, max_retries=0
        )
        OpenAIClient("otra-clave-ficticia", "modelo-de-prueba", 0.7, 200, 12)
        self.sdk_constructor.assert_called_with(
            api_key="otra-clave-ficticia", timeout=12, max_retries=0
        )


    async def test_razonamiento_openai_en_ambos_modos(self):
        from inspect import signature
        from openai.resources.chat.completions import AsyncCompletions

        for effort in ("low", None):
            with self.subTest(effort=effort):
                client = OpenAIClient(
                    "clave-ficticia", "gpt-6.1-sol", 0.7, 1024,
                    reasoning_effort=effort,
                )
                self.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Hola"))]
                )
                await client.generate(self.messages)
                normal = self.create.call_args.kwargs
                stream = MagicMock()
                stream.__aenter__.return_value = stream
                stream.__aiter__.return_value = []
                self.create.return_value = stream
                _ = [part async for part in client.generate_stream(self.messages)]
                streaming = self.create.call_args.kwargs
                for parameters in (normal, streaming):
                    signature(AsyncCompletions.create).bind(None, **parameters)
                    self.assertEqual(parameters["max_completion_tokens"], 1024)
                    self.assertNotIn("temperature", parameters)
                    self.assertNotIn("max_tokens", parameters)
                    self.assertEqual(parameters.get("reasoning_effort"), effort)
                    if effort is None:
                        self.assertNotIn("reasoning_effort", parameters)

    async def test_genera_respuesta_y_espera_al_sdk(self):
        self.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hola de vuelta"))]
        )
        response = await self.client.generate(self.messages)
        self.assertEqual(response.provider, Provider.OPENAI)
        self.assertEqual(response.model, "modelo-de-prueba")
        self.assertEqual(response.content, "Hola de vuelta")
        self.assertIsNone(response.error)
        self.create.assert_awaited_once_with(
            model="modelo-de-prueba",
            messages=[{"role": "user", "content": "Hola"}],
            temperature=0.7,
            max_completion_tokens=200,
        )

    async def test_respuesta_sin_texto_es_error_controlado(self):
        for choices in ([], [SimpleNamespace(message=SimpleNamespace(content=None))]):
            with self.subTest(choices=choices):
                self.create.return_value = SimpleNamespace(choices=choices)
                response = await self.client.generate(self.messages)
                self.assertEqual(response.content, "")
                self.assertEqual(response.error, "OpenAI devolvió una respuesta sin texto.")

    async def test_errores_normales_son_controlados_y_no_exponen_secretos(self):
        descriptions = ("autenticación", "cuota", "tiempo", "conexión", "API")
        for error, description in zip(self.errores_simulados(), descriptions):
            with self.subTest(error=type(error).__name__):
                self.create.reset_mock()
                self.create.side_effect = error
                response = await self.client.generate(self.messages)
                self.assertEqual(response.content, "")
                self.assertIn(description, response.error)
                self.assertNotIn("clave-ficticia", response.error)
                self.create.assert_awaited_once()

    async def test_stream_emite_texto_e_ignora_fragmentos_sin_texto(self):
        stream = MagicMock()
        stream.__aenter__.return_value = stream
        stream.__aiter__.return_value = [
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hola"))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" mundo"))]),
        ]
        self.create.return_value = stream
        fragments = [part async for part in self.client.generate_stream(self.messages)]
        self.assertEqual(fragments, ["Hola", " mundo"])
        self.create.assert_awaited_once_with(
            model="modelo-de-prueba",
            messages=[{"role": "user", "content": "Hola"}],
            temperature=0.7,
            max_completion_tokens=200,
            stream=True,
        )
        stream.__aexit__.assert_awaited_once()

    async def test_errores_al_abrir_stream_se_emiten_como_texto(self):
        for error in self.errores_simulados():
            with self.subTest(error=type(error).__name__):
                self.create.side_effect = error
                fragments = [part async for part in self.client.generate_stream(self.messages)]
                self.assertEqual(len(fragments), 1)
                self.assertIn("Error durante el streaming", fragments[0])
                self.assertNotIn("clave-ficticia", fragments[0])

    async def test_error_durante_stream_conserva_texto_previo(self):
        async def fragments_with_error():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Texto parcial"))]
            )
            raise self.errores_simulados()[2]

        stream = MagicMock()
        stream.__aenter__.return_value = stream
        stream.__aiter__.side_effect = fragments_with_error
        self.create.return_value = stream
        fragments = [part async for part in self.client.generate_stream(self.messages)]
        self.assertEqual(fragments[0], "Texto parcial")
        self.assertIn("tiempo de espera", fragments[1])
        stream.__aexit__.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
