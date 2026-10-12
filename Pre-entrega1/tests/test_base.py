"""Verificación del contrato abstracto mediante un cliente local de prueba."""

import unittest

from llmclient.base import BaseLLMClient
from llmclient.schemas import ChatMessage, ModelResponse, Provider


class ClienteDePrueba(BaseLLMClient):
    """Implementación mínima usada solo en tests, sin llamadas a APIs."""

    async def generate(self, messages):
        return ModelResponse(
            provider=Provider.OPENAI,
            model="modelo-de-prueba",
            content=messages[0].content,
        )

    async def generate_stream(self, messages):
        yield "Respuesta: "
        yield messages[0].content


class BaseLLMClientTests(unittest.IsolatedAsyncioTestCase):
    def test_no_permite_instanciar_clase_base(self):
        with self.assertRaises(TypeError):
            BaseLLMClient()

    def test_exige_implementar_ambos_metodos(self):
        class ClienteSinStreaming(BaseLLMClient):
            async def generate(self, messages):
                return await ClienteDePrueba().generate(messages)

        class ClienteSinGeneracion(BaseLLMClient):
            async def generate_stream(self, messages):
                yield "Respuesta simulada"

        for client_class in (ClienteSinStreaming, ClienteSinGeneracion):
            with self.subTest(client_class=client_class.__name__):
                with self.assertRaises(TypeError):
                    client_class()

    async def test_cliente_concreto_devuelve_respuesta_completa(self):
        client = ClienteDePrueba()
        messages = [ChatMessage(role="user", content="Hola")]

        response = await client.generate(messages)

        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.content, "Hola")
        self.assertIsNone(response.error)

    async def test_cliente_concreto_emite_fragmentos_en_orden(self):
        client = ClienteDePrueba()
        messages = [ChatMessage(role="user", content="Hola")]

        fragments = []
        async for fragment in client.generate_stream(messages):
            fragments.append(fragment)

        self.assertEqual(fragments, ["Respuesta: ", "Hola"])


if __name__ == "__main__":
    unittest.main()
