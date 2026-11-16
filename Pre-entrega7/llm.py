"""Construcción del modelo OpenAI compartido por los roles del orquestador."""

from langchain_openai import ChatOpenAI

from config import Configuracion


def crear_llm(config: Configuracion) -> ChatOpenAI:
    """Crear el modelo; cada rol vinculará sus propias herramientas o esquema."""
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
    return llm
