"""Interfaz común para los clientes asíncronos de modelos de lenguaje."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from llmclient.schemas import ChatMessage, ModelResponse


class BaseLLMClient(ABC):
    """Contrato que deben implementar todos los clientes de proveedores."""

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Devolver una respuesta completa o un error controlado."""
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Emitir fragmentos de texto o un mensaje de error controlado."""
        raise NotImplementedError
        yield  # nunca se ejecuta; solo le indica a Python que este método es un generador
