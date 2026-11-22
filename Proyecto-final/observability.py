"""Activar las trazas nativas de LangGraph y correlacionarlas con el trabajo."""

import asyncio
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig
from langsmith import Client, tracing_context
from langsmith.address import agent_address

from config import Configuracion

CAMPOS_CONFIGURACION_PUBLICA = (
    "model", "timeout_seconds", "max_tokens", "reasoning_effort",
    "recursion_limit", "job_timeout_seconds",
)


class Observabilidad:
    def __init__(self, config: Configuracion) -> None:
        self.client = None
        self.destino = {}
        # Lista explícita: nunca serializar toda la configuración con sus secretos.
        self.configuracion_publica = {
            campo: getattr(config, campo) for campo in CAMPOS_CONFIGURACION_PUBLICA
        }
        if config.langsmith_tracing and config.langsmith_api_key is not None:
            if config.langsmith_agent_id is not None:
                self.destino = {
                    "address": agent_address(config.langsmith_agent_id, config.langsmith_agent_environment),
                }
            else:
                self.destino = {"project_name": config.langsmith_project}
            self.client = Client(api_key=config.langsmith_api_key.get_secret_value().strip())

    def contexto(self):
        """Usar un contexto por ejecución con un cliente compartido."""
        return tracing_context(
            enabled=self.client is not None, client=self.client, parent=False, **self.destino,
            metadata={"configuracion_ejecucion": self.configuracion_publica},
        )

    async def cerrar(self) -> None:
        if self.client is not None:
            await asyncio.to_thread(self.client.close, timeout=5)


def configurar_invocacion(
    job_id: UUID, recursion_limit: int, *, es_reanudacion: bool, lote_id: UUID | None = None,
) -> RunnableConfig:
    """Identificar la traza raíz y propagar los IDs a los nodos y herramientas."""
    fase = "reanudacion" if es_reanudacion else "inicio"
    identificador = str(job_id)
    tags = ["proyecto-final", fase]
    metadata = {"job_id": identificador, "thread_id": identificador, "fase": fase}
    if lote_id is not None:
        tags.append(f"lote:{lote_id}")
        metadata["lote_id"] = str(lote_id)
    return {
        "configurable": {"thread_id": identificador},
        "recursion_limit": recursion_limit,
        "run_name": f"orquestador_viajes_{fase}",
        "run_id": uuid4() if es_reanudacion else job_id,
        "tags": tags,
        "metadata": metadata,
    }
