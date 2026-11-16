"""Abrir el checkpointer asíncrono con los tipos del estado del viaje."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

TIPOS_CHECKPOINT = [
    ("schemas", nombre)
    for nombre in (
        "SolicitudParcial",
        "SolicitudViaje",
        "ResultadoHoteleria",
        "ResultadoTurismo",
        "ResultadoAnalisis",
        "DecisionSupervisor",
        "PropuestaReserva",
        "Coordenadas",
        "Periodo",
        "Observacion",
        "OpcionHotel",
        "OpcionTuristica",
        "OpcionEvaluada",
    )
]


@asynccontextmanager
async def abrir_checkpointer(redis_url: str) -> AsyncIterator[AsyncRedisSaver]:
    """Inicializar Redis y permitir la reconstrucción de los modelos del proyecto."""
    async with AsyncRedisSaver.from_conn_string(
        redis_url, connection_args={"socket_connect_timeout": 5, "socket_timeout": 5},
    ) as saver:
        # El saver usa JSON de Redis y puede recurrir a MessagePack.
        saver.serde = JsonPlusRedisSerializer(
            allowed_json_modules=TIPOS_CHECKPOINT,
            allowed_msgpack_modules=TIPOS_CHECKPOINT,
        )
        await saver.asetup()
        yield saver
