"""Comprobar que una llamada pendiente permite avanzar a otra tarea."""

import asyncio
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from anthropic.resources.messages import AsyncMessages
from anthropic import APITimeoutError, BadRequestError
import httpx2

from llmclient.manager import AsyncLLMManager
from llmclient.schemas import ChatMessage, LLMConfig, Provider


class AsyncBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_anthropic_conserva_texto_y_error_a_traves_del_manager(self):
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        error = APITimeoutError(request=request)

        async def simulated_text():
            yield "Texto parcial"
            raise error

        with patch("llmclient.providers.anthropic_client.AsyncAnthropic") as constructor:
            context = constructor.return_value.messages.stream.return_value
            context.__aenter__.return_value = SimpleNamespace(text_stream=simulated_text())
            manager = AsyncLLMManager(LLMConfig(
                provider=Provider.ANTHROPIC, model="modelo-de-prueba",
                anthropic_api_key="clave-ficticia",
            ))
            received = [chunk async for chunk in manager.generate_stream([
                ChatMessage(role="user", content="Hola")
            ])]
            self.assertEqual(received, [
                "Texto parcial",
                "\n[Error durante el streaming: Se agotó el tiempo de espera de Anthropic.]",
            ])
            context.__aexit__.assert_awaited_once()
            self.assertIs(context.__aexit__.call_args.args[1], error)

    async def test_stream_anthropic_lista_vacia_propaga_rechazo_simulado(self):
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        error = BadRequestError(
            "rechazo simulado", response=httpx2.Response(400, request=request),
            body={"error": {"message": "rechazo simulado"}},
        )
        with patch("llmclient.providers.anthropic_client.AsyncAnthropic") as constructor:
            call = constructor.return_value.messages.stream
            call.return_value.__aenter__.side_effect = error
            manager = AsyncLLMManager(LLMConfig(
                provider=Provider.ANTHROPIC, model="modelo-de-prueba",
                anthropic_api_key="clave-ficticia",
            ))
            received = [chunk async for chunk in manager.generate_stream([])]
            call.assert_called_once_with(
                model="modelo-de-prueba", max_tokens=1024, messages=[],
            )
            self.assertEqual(received, [
                "\n[Error durante el streaming: Error de la API de Anthropic (HTTP 400). "
                "Solicitud inválida. Revisá el modelo y sus parámetros.]",
            ])
            call.return_value.__aenter__.assert_awaited_once()
            call.return_value.__aexit__.assert_not_awaited()

    async def test_stream_anthropic_propaga_primer_fragmento_antes_del_cierre(self):
        cases = [
            ([ChatMessage(role="user", content="Respondé con una palabra.")], ""),
            ([
                ChatMessage(role="system", content="Sos un asistente breve."),
                ChatMessage(role="user", content="Hola"),
            ], "Sos un asistente breve."),
        ]
        for messages, system in cases:
            with self.subTest(system=system), patch(
                "llmclient.providers.anthropic_client.AsyncAnthropic"
            ) as constructor:
                release = asyncio.Event()
                first_received = asyncio.Event()
                waiting_for_release = asyncio.Event()
                received = []

                async def simulated_text():
                    yield "Hola"
                    waiting_for_release.set()
                    await release.wait()
                    yield " mundo"

                stream = SimpleNamespace(text_stream=simulated_text())
                context = constructor.return_value.messages.stream.return_value
                context.__aenter__.return_value = stream
                manager = AsyncLLMManager(LLMConfig(
                    provider=Provider.ANTHROPIC, model="modelo-de-prueba",
                    anthropic_api_key="clave-ficticia",
                ))

                async def consume():
                    async for chunk in manager.generate_stream(messages):
                        received.append(chunk)
                        if len(received) == 1:
                            first_received.set()

                consumer = asyncio.create_task(consume())
                try:
                    await asyncio.wait_for(first_received.wait(), timeout=1)
                    await asyncio.wait_for(waiting_for_release.wait(), timeout=1)
                    self.assertEqual(received, ["Hola"])
                    self.assertFalse(consumer.done())
                    context.__aenter__.assert_awaited_once()
                    context.__aexit__.assert_not_awaited()

                    call = constructor.return_value.messages.stream
                    call.assert_called_once_with(
                        model="modelo-de-prueba", max_tokens=1024,
                        messages=[m.model_dump() for m in messages if m.role != "system"],
                        **({"system": system} if system else {}),
                    )
                    inspect.signature(AsyncMessages.stream).bind(None, **call.call_args.kwargs)

                    release.set()
                    await asyncio.wait_for(consumer, timeout=1)
                    self.assertEqual(received, ["Hola", " mundo"])
                    for chunk in received:
                        self.assertIsInstance(chunk, str)
                        self.assertTrue(chunk)
                    context.__aexit__.assert_awaited_once_with(None, None, None)
                finally:
                    release.set()
                    consumer.cancel()
                    await asyncio.gather(consumer, return_exceptions=True)
                    await stream.text_stream.aclose()

    async def test_otra_tarea_avanza_mientras_el_sdk_espera(self):
        cases = [
            (Provider.OPENAI, "openai_api_key", "llmclient.providers.openai_client.AsyncOpenAI"),
            (Provider.ANTHROPIC, "anthropic_api_key", "llmclient.providers.anthropic_client.AsyncAnthropic"),
        ]
        for provider, key_field, target in cases:
            with self.subTest(provider=provider), patch(target) as constructor:
                started = asyncio.Event()
                release = asyncio.Event()
                other_task_ran = asyncio.Event()

                async def delayed_response(**kwargs):
                    started.set()
                    await release.wait()
                    if provider == Provider.OPENAI:
                        return SimpleNamespace(choices=[
                            SimpleNamespace(message=SimpleNamespace(content="Hola"))
                        ])
                    if provider == Provider.ANTHROPIC:
                        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Hola")])

                sdk = constructor.return_value
                call = AsyncMock(side_effect=delayed_response)
                if provider == Provider.OPENAI:
                    sdk.chat.completions.create = call
                else:
                    sdk.messages.create = call

                manager = AsyncLLMManager(LLMConfig(
                    provider=provider, model="modelo-de-prueba", **{key_field: "clave-ficticia"}
                ))

                async def other_task():
                    await started.wait()
                    other_task_ran.set()

                generation = asyncio.create_task(manager.generate([
                    ChatMessage(role="user", content="Hola")
                ]))
                observer = asyncio.create_task(other_task())
                try:
                    await asyncio.wait_for(other_task_ran.wait(), timeout=1)
                    self.assertFalse(generation.done())
                    release.set()
                    response = await asyncio.wait_for(generation, timeout=1)
                    self.assertEqual(response.content, "Hola")
                    call.assert_awaited_once()
                finally:
                    release.set()
                    generation.cancel()
                    observer.cancel()
                    await asyncio.gather(generation, observer, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
