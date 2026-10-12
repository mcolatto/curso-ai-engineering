"""Pruebas de selección y delegación con clientes simulados."""

import unittest
from unittest.mock import AsyncMock, patch

from llmclient.manager import AsyncLLMManager
from llmclient.schemas import ChatMessage, LLMConfig, ModelResponse, Provider


class AsyncLLMManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clients = {}
        for name in ("OpenAIClient", "AnthropicClient"):
            client_patch = patch(f"llmclient.manager.{name}")
            self.clients[name] = client_patch.start()
            self.addCleanup(client_patch.stop)
        self.cases = [
            (Provider.OPENAI, "openai_api_key", "OpenAIClient"),
            (Provider.ANTHROPIC, "anthropic_api_key", "AnthropicClient"),
        ]
        self.messages = [ChatMessage(role="user", content="Hola")]

    def test_selecciona_solo_el_cliente_elegido_y_transmite_configuracion(self):
        for provider, key_name, client_name in self.cases:
            with self.subTest(provider=provider):
                for constructor in self.clients.values():
                    constructor.reset_mock()
                config = LLMConfig(
                    provider=provider,
                    model="modelo-de-prueba",
                    temperature=0.5,
                    max_tokens=123,
                    timeout_seconds=12,
                    **{key_name: "clave-ficticia"},
                )
                AsyncLLMManager(config)
                extra = {"reasoning_effort": None} if provider == Provider.OPENAI else {}
                self.clients[client_name].assert_called_once_with(
                    api_key="clave-ficticia",
                    model="modelo-de-prueba",
                    temperature=0.5,
                    max_tokens=123,
                    timeout_seconds=12,
                    **extra,
                )
                for name, constructor in self.clients.items():
                    if name != client_name:
                        constructor.assert_not_called()

    def test_rechaza_clave_ausente_vacia_o_con_espacios_antes_de_crear_cliente(self):
        for provider, key_name, _ in self.cases:
            for key in (None, "", " \n\t"):
                with self.subTest(provider=provider, key=key):
                    config = LLMConfig(
                        provider=provider, model="modelo-de-prueba", **{key_name: key}
                    )
                    with self.assertRaisesRegex(ValueError, f"Falta {key_name}"):
                        AsyncLLMManager(config)
                    for constructor in self.clients.values():
                        constructor.assert_not_called()


    def test_transmite_razonamiento_solo_a_openai(self):
        for provider, key_name, client_name in self.cases:
            with self.subTest(provider=provider):
                AsyncLLMManager(LLMConfig(
                    provider=provider, model="modelo", reasoning_effort="low",
                    **{key_name: "clave-ficticia"},
                ))
                kwargs = self.clients[client_name].call_args.kwargs
                if provider == Provider.OPENAI:
                    self.assertEqual(kwargs["reasoning_effort"], "low")
                else:
                    self.assertNotIn("reasoning_effort", kwargs)


    def test_clave_de_otro_proveedor_no_sustituye_la_clave_requerida(self):
        config = LLMConfig(
            provider=Provider.ANTHROPIC,
            model="modelo-de-prueba",
            openai_api_key="clave-ficticia-de-otro-proveedor",
        )
        with self.assertRaisesRegex(ValueError, "Falta anthropic_api_key"):
            AsyncLLMManager(config)
        for constructor in self.clients.values():
            constructor.assert_not_called()

    async def test_delega_generate_y_conserva_respuesta_o_error(self):
        for provider, key_name, client_name in self.cases:
            for error in (None, "Error de conexión"):
                with self.subTest(provider=provider, error=error):
                    response = ModelResponse(
                        provider=provider,
                        model="modelo-de-prueba",
                        content="Respuesta simulada" if error is None else "",
                        error=error,
                    )
                    client = self.clients[client_name].return_value
                    client.generate = AsyncMock(return_value=response)
                    manager = AsyncLLMManager(LLMConfig(
                        provider=provider,
                        model="modelo-de-prueba",
                        **{key_name: "clave-ficticia"},
                    ))
                    result = await manager.generate(self.messages)
                    self.assertIs(result, response)
                    client.generate.assert_awaited_once_with(self.messages)

    async def test_delega_stream_y_conserva_fragmentos_y_error(self):
        async def simulated_stream(messages):
            yield "Texto "
            yield "parcial"
            yield "\n[Error durante el streaming: error simulado]"

        for provider, key_name, client_name in self.cases:
            with self.subTest(provider=provider):
                client = self.clients[client_name].return_value
                client.generate_stream.side_effect = simulated_stream
                manager = AsyncLLMManager(LLMConfig(
                    provider=provider,
                    model="modelo-de-prueba",
                    **{key_name: "clave-ficticia"},
                ))
                fragments = [part async for part in manager.generate_stream(self.messages)]
                self.assertEqual(fragments, [
                    "Texto ", "parcial", "\n[Error durante el streaming: error simulado]"
                ])
                client.generate_stream.assert_called_once_with(self.messages)


if __name__ == "__main__":
    unittest.main()
