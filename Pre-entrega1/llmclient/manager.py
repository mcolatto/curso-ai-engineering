"""Selección del proveedor y acceso a la interfaz común de generación."""

from collections.abc import AsyncGenerator

from llmclient.base import BaseLLMClient
from llmclient.providers.anthropic_client import AnthropicClient
from llmclient.providers.openai_client import OpenAIClient
from llmclient.schemas import ChatMessage, LLMConfig, ModelResponse, Provider


class AsyncLLMManager:
    """Crear el cliente configurado y delegar las llamadas de generación."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: BaseLLMClient = self._crear_cliente()

    def _crear_cliente(self) -> BaseLLMClient:
        """Seleccionar el cliente y verificar su clave antes de instanciarlo."""
        if self.config.provider == Provider.OPENAI:
            client_class = OpenAIClient
            secret = self.config.openai_api_key
            key_name = "openai_api_key"
        elif self.config.provider == Provider.ANTHROPIC:
            client_class = AnthropicClient
            secret = self.config.anthropic_api_key
            key_name = "anthropic_api_key"
        else:
            raise ValueError("Proveedor no soportado")

        if secret is None or not secret.get_secret_value().strip():
            raise ValueError(f"Falta {key_name} en la configuración")

        if self.config.provider == Provider.OPENAI:
            return OpenAIClient(
                api_key=secret.get_secret_value(),
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout_seconds=self.config.timeout_seconds,
                reasoning_effort=self.config.reasoning_effort,
            )

        return client_class(
            api_key=secret.get_secret_value(),
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout_seconds=self.config.timeout_seconds,
        )

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Delegar la generación de una respuesta completa en el cliente."""
        return await self._client.generate(messages)

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Delegar la generación y emitir cada fragmento recibido del cliente."""
        async for chunk in self._client.generate_stream(messages):
            yield chunk
