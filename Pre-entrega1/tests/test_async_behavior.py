"""Comprobar que una llamada pendiente permite avanzar a otra tarea."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from llmclient.manager import AsyncLLMManager
from llmclient.schemas import ChatMessage, LLMConfig, Provider


class AsyncBehaviorTests(unittest.IsolatedAsyncioTestCase):
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
