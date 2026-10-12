"""Generación asíncrona de texto mediante la API de OpenAI."""

from collections.abc import AsyncGenerator

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from llmclient.base import BaseLLMClient
from llmclient.schemas import ChatMessage, ModelResponse, Provider


def _mensaje_error(error: APIError) -> str:
    """Describir el error sin exponer el contenido de la excepción ni secretos."""
    if isinstance(error, AuthenticationError):
        return f"Error de autenticación con OpenAI. Revisá la clave configurada."
    if isinstance(error, RateLimitError):
        return f"Límite de solicitudes o cuota de OpenAI excedido."
    if isinstance(error, APITimeoutError):
        return f"Se agotó el tiempo de espera de OpenAI."
    if isinstance(error, APIConnectionError):
        return f"Error de conexión con OpenAI."
    return f"Error de la API de OpenAI."


class OpenAIClient(BaseLLMClient):
    """Adaptar las llamadas y respuestas de OpenAI a la interfaz común."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float = 30,
        reasoning_effort: str | None = None,
    ):
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def _parametros_generacion(self) -> dict:
        """Adaptar tokens y razonamiento al modelo elegido."""
        parameters = {"max_completion_tokens": self.max_tokens}
        if self.reasoning_effort is not None:
            parameters["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_effort is None and self.model != "gpt-6.1-sol" and not self.model.startswith("gpt-6.1-sol-"):
            parameters["temperature"] = self.temperature
        return parameters

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa (modo normal, no streaming)."""
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                **self._parametros_generacion(),
            )
            content = response.choices[0].message.content if response.choices else None
            if not content:
                return ModelResponse(
                    provider=Provider.OPENAI,
                    model=self.model,
                    content="",
                    error="OpenAI devolvió una respuesta sin texto.",
                )
            return ModelResponse(
                provider=Provider.OPENAI, model=self.model, content=content
            )
        except APIError as error:
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model,
                content="",
                error=_mensaje_error(error),
            )

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Genera la respuesta token a token (modo streaming)."""
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                **self._parametros_generacion(),
                stream=True,
            )
            async with stream:
                async for chunk in stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta
        except APIError as error:
            yield f"\n[Error durante el streaming: {_mensaje_error(error)}]"
