"""Generación asíncrona de texto mediante la API de Anthropic."""

from collections.abc import AsyncGenerator

from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    RateLimitError,
)

from llmclient.base import BaseLLMClient
from llmclient.schemas import ChatMessage, ModelResponse, Provider


def _mensaje_error(error: APIError) -> str:
    """Describir el error sin exponer el contenido de la excepción ni secretos."""
    if isinstance(error, AuthenticationError):
        return "Error de autenticación con Anthropic. Revisá la clave configurada."
    if isinstance(error, RateLimitError):
        return "Límite de solicitudes o cuota de Anthropic excedido."
    if isinstance(error, APITimeoutError):
        return "Se agotó el tiempo de espera de Anthropic."
    if isinstance(error, APIConnectionError):
        return "Error de conexión con Anthropic."
    if isinstance(error, APIStatusError):
        prefix = f"Error de la API de Anthropic (HTTP {error.status_code})."
        # Usar el mensaje para clasificar la causa sin mostrar su contenido.
        body = error.body if isinstance(error.body, dict) else {}
        detail = body.get("error", body)
        message = detail.get("message", "") if isinstance(detail, dict) else ""
        message = message.lower() if isinstance(message, str) else ""
        if "credit balance" in message or "insufficient credit" in message:
            return f"{prefix} Saldo de créditos insuficiente. Revisá la facturación."
        if error.status_code == 404:
            return f"{prefix} Recurso no encontrado. Revisá el modelo configurado."
        if error.status_code == 403:
            return f"{prefix} Acceso denegado. Revisá los permisos de la cuenta."
        if error.status_code == 400:
            return f"{prefix} Solicitud inválida. Revisá el modelo y sus parámetros."
        if error.status_code >= 500:
            return f"{prefix} Fallo del servicio de Anthropic."
        return prefix
    return "Error de la API de Anthropic."


class AnthropicClient(BaseLLMClient):
    """Adaptar las llamadas y respuestas de Anthropic a la interfaz común."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float = 30,
    ):
        self._client = AsyncAnthropic(
            api_key=api_key, timeout=timeout_seconds, max_retries=0
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _convertir_mensajes(
        self, messages: list[ChatMessage]
    ) -> tuple[list[dict[str, str]], str]:
        """Separar las instrucciones de sistema del historial de conversación."""
        conversation = []
        system_messages = []
        for message in messages:
            if message.role == "system":
                system_messages.append(message.content)
            else:
                conversation.append(message.model_dump())
        return conversation, "\n".join(system_messages)

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa (modo normal, no streaming)."""
        try:
            conversation, system = self._convertir_mensajes(messages)
            system_parameters = {"system": system} if system.strip() else {}
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,  # obligatorio en Anthropic, a diferencia de OpenAI
                messages=conversation,
                **system_parameters,
            )
            content = "".join(
                block.text for block in response.content if block.type == "text"
            )
            if not content:
                return ModelResponse(
                    provider=Provider.ANTHROPIC,
                    model=self.model,
                    content="",
                    error="Anthropic devolvió una respuesta sin texto.",
                )
            return ModelResponse(
                provider=Provider.ANTHROPIC, model=self.model, content=content
            )
        except APIError as error:
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model,
                content="",
                error=_mensaje_error(error),
            )

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Genera la respuesta token a token (modo streaming)."""
        try:
            conversation, system = self._convertir_mensajes(messages)
            system_parameters = {"system": system} if system.strip() else {}
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=conversation,
                **system_parameters,
            ) as stream:
                async for texto in stream.text_stream:
                    if texto:
                        yield texto
        except APIError as error:
            yield f"\n[Error durante el streaming: {_mensaje_error(error)}]"
