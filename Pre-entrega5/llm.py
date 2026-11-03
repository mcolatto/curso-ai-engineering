"""Construcción del modelo OpenAI vinculado a las herramientas del agente."""

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from config import Configuracion
from tools import herramientas


def crear_llm(config: Configuracion) -> Runnable[LanguageModelInput, BaseMessage]:
    """Configurar el modelo y permitirle decidir cuándo utilizar herramientas."""
    optional_parameters = {}
    if config.reasoning_effort is not None:
        optional_parameters["reasoning_effort"] = config.reasoning_effort

    llm = ChatOpenAI(
        api_key=config.openai_api_key,
        model=config.model,
        timeout=config.timeout_seconds,
        max_completion_tokens=config.max_tokens,
        max_retries=0,
        **optional_parameters,
    )
    return llm.bind_tools(herramientas)
