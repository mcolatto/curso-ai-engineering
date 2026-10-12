"""Demostración secuencial de generación normal y streaming por proveedor."""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from llmclient.manager import AsyncLLMManager
from llmclient.schemas import ChatMessage, LLMConfig, Provider


async def main() -> None:
    """Consultar los proveedores configurados y mostrar respuestas o errores."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    messages = [
        ChatMessage(role="user", content="¿Qué es la entropía? Respondé en 2 líneas.")
    ]
    providers = [
        (Provider.OPENAI, "OPENAI_API_KEY", "OPENAI_MODEL", "openai_api_key"),
        (Provider.ANTHROPIC, "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "anthropic_api_key"),
    ]

    for provider, key_variable, model_variable, key_field in providers:
        print(f"\n=== {provider.value.upper()} ===")
        api_key = os.getenv(key_variable, "").strip()
        if not api_key:
            print(f"Omitido: no se configuró {key_variable}.")
            continue

        try:
            config = LLMConfig(
                provider=provider,
                model=os.getenv(model_variable, ""),
                temperature=os.getenv("LLM_TEMPERATURE", "0.7"),
                max_tokens=os.getenv("LLM_MAX_TOKENS", "1024"),
                timeout_seconds=os.getenv("LLM_TIMEOUT_SECONDS", "30"),
                reasoning_effort=(
                    os.getenv("OPENAI_REASONING_EFFORT", "").strip() or None
                    if provider == Provider.OPENAI else None
                ),
                **{key_field: api_key},
            )
            manager = AsyncLLMManager(config)
        except ValidationError as error:
            fields = [".".join(str(part) for part in item["loc"]) for item in error.errors()]
            print(f"Error de configuración. Revisá: {', '.join(fields)}.")
            continue
        except ValueError:
            print("Error de configuración: no se pudo crear el cliente del proveedor.")
            continue

        print("Modo normal:")
        print()
        response = await manager.generate(messages)
        if response.error:
            print(f"Error: {response.error}")
        else:
            print(response.content)

        print()
        print("Modo streaming:")
        print()
        async for chunk in manager.generate_stream(messages):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
