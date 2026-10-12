"""Pruebas del cliente Anthropic sin solicitudes de red."""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from anthropic.resources.messages import AsyncMessages

from llmclient.providers.anthropic_client import AnthropicClient
from llmclient.schemas import ChatMessage, Provider


class AnthropicClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sdk_patch = patch("llmclient.providers.anthropic_client.AsyncAnthropic")
        self.sdk_constructor = self.sdk_patch.start()
        self.addCleanup(self.sdk_patch.stop)
        self.sdk = self.sdk_constructor.return_value
        self.create = AsyncMock()
        self.sdk.messages.create = self.create
        self.client = AnthropicClient("clave-ficticia", "modelo-de-prueba", 0.7, 200)
        self.messages = [ChatMessage(role="user", content="Hola")]

    def errores_simulados(self):
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
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
        AnthropicClient("otra-clave-ficticia", "modelo-de-prueba", 0.7, 200, 12)
        self.sdk_constructor.assert_called_with(
            api_key="otra-clave-ficticia", timeout=12, max_retries=0
        )

    async def test_genera_texto_y_separa_multiples_mensajes_system(self):
        messages = [
            ChatMessage(role="system", content="Sé breve."),
            ChatMessage(role="user", content="Hola"),
            ChatMessage(role="system", content="Respondé en español."),
            ChatMessage(role="assistant", content="Hola de vuelta"),
        ]
        self.create.return_value = SimpleNamespace(content=[
            SimpleNamespace(type="text", text="Primera parte"),
            SimpleNamespace(type="tool_use"),
            SimpleNamespace(type="text", text=" y segunda parte"),
        ])
        response = await self.client.generate(messages)
        self.assertEqual(response.provider, Provider.ANTHROPIC)
        self.assertEqual(response.model, "modelo-de-prueba")
        self.assertEqual(response.content, "Primera parte y segunda parte")
        self.assertIsNone(response.error)
        # Comprobar los argumentos contra la firma real, sin ejecutar el SDK.
        inspect.signature(AsyncMessages.create).bind(None, **self.create.call_args.kwargs)
        self.create.assert_awaited_once_with(
            model="modelo-de-prueba", max_tokens=200,
            messages=[
                {"role": "user", "content": "Hola"},
                {"role": "assistant", "content": "Hola de vuelta"},
            ],
            system="Sé breve.\nRespondé en español.",
        )

    async def test_respuesta_sin_texto_es_error_controlado(self):
        for content in ([], [SimpleNamespace(type="tool_use")]):
            with self.subTest(content=content):
                self.create.return_value = SimpleNamespace(content=content)
                response = await self.client.generate(self.messages)
                self.assertEqual(response.content, "")
                self.assertEqual(response.error, "Anthropic devolvió una respuesta sin texto.")

    async def test_errores_normales_son_controlados_y_no_exponen_secretos(self):
        for error, description in zip(
            self.errores_simulados(), ("autenticación", "cuota", "tiempo", "conexión", "API")
        ):
            with self.subTest(error=type(error).__name__):
                self.create.reset_mock()
                self.create.side_effect = error
                response = await self.client.generate(self.messages)
                self.assertEqual(response.content, "")
                self.assertIn(description, response.error)
                self.assertNotIn("clave-ficticia", response.error)
                self.create.assert_awaited_once()

    async def test_diagnostico_http_en_ambos_modos_sin_exponer_mensaje_original(self):
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        for status, message, expected in (
            (400, "Your credit balance is too low clave-ficticia", "Saldo de créditos"),
            (404, "model clave-ficticia not found", "modelo configurado"),
            (403, "clave-ficticia forbidden", "Acceso denegado"),
            (400, "invalid clave-ficticia", "Solicitud inválida"),
            (529, "overloaded clave-ficticia", "Fallo del servicio"),
        ):
            with self.subTest(status=status, message=message):
                error = APIStatusError(
                    message, response=httpx2.Response(status, request=request),
                    body={"error": {"message": message}},
                )
                self.create.side_effect = error
                response = await self.client.generate(self.messages)
                self.sdk.messages.stream.return_value.__aenter__.side_effect = error
                fragments = [part async for part in self.client.generate_stream(self.messages)]
                for text in (response.error, fragments[0]):
                    self.assertIn(f"HTTP {status}", text)
                    self.assertIn(expected, text)
                    self.assertNotIn("clave-ficticia", text)

    async def test_stream_emite_fragmentos_y_separa_system(self):
        stream = MagicMock()
        stream.text_stream.__aiter__.return_value = ["Hola", "", " mundo"]
        context = self.sdk.messages.stream.return_value
        context.__aenter__.return_value = stream
        messages = [ChatMessage(role="system", content="Sé breve."), *self.messages]
        fragments = [part async for part in self.client.generate_stream(messages)]
        self.assertEqual(fragments, ["Hola", " mundo"])
        inspect.signature(AsyncMessages.stream).bind(
            None, **self.sdk.messages.stream.call_args.kwargs
        )
        self.sdk.messages.stream.assert_called_once_with(
            model="modelo-de-prueba", max_tokens=200,
            messages=[{"role": "user", "content": "Hola"}], system="Sé breve.",
        )
        context.__aexit__.assert_awaited_once()

    async def test_sin_system_envia_instruccion_vacia(self):
        self.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Hola")]
        )
        await self.client.generate(self.messages)
        self.assertEqual(self.create.call_args.kwargs["system"], "")

    async def test_errores_al_abrir_stream_se_emiten_como_texto(self):
        for error in self.errores_simulados():
            with self.subTest(error=type(error).__name__):
                self.sdk.messages.stream.return_value.__aenter__.side_effect = error
                fragments = [part async for part in self.client.generate_stream(self.messages)]
                self.assertEqual(len(fragments), 1)
                self.assertIn("Error durante el streaming", fragments[0])
                self.assertNotIn("clave-ficticia", fragments[0])

    async def test_error_durante_stream_conserva_texto_previo(self):
        async def fragments_with_error():
            yield "Texto parcial"
            raise self.errores_simulados()[2]

        stream = MagicMock()
        stream.text_stream.__aiter__.side_effect = fragments_with_error
        context = self.sdk.messages.stream.return_value
        context.__aenter__.return_value = stream
        fragments = [part async for part in self.client.generate_stream(self.messages)]
        self.assertEqual(fragments[0], "Texto parcial")
        self.assertIn("tiempo de espera", fragments[1])
        context.__aexit__.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
