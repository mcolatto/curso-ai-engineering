"""Crear, consultar y actualizar los registros de trabajo en Redis."""

from datetime import UTC, datetime
from uuid import UUID

from redis.asyncio import Redis
from redis.exceptions import RedisError, WatchError

from schemas import DecisionHumana, EstadoTrabajo, PropuestaReserva, ResultadoTrabajo, Trabajo

PREFIJO_TRABAJO = "preentrega7:job:"
ESTADOS_ACTIVOS: tuple[EstadoTrabajo, ...] = ("PENDING", "RUNNING")


class TrabajosRedis:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def crear(self, job_id: UUID, *, lote_id: UUID | None = None) -> Trabajo:
        trabajo = Trabajo(job_id=job_id, lote_id=lote_id, status="PENDING", created_at=datetime.now(UTC))
        creado = await self.redis.set(PREFIJO_TRABAJO + str(job_id), trabajo.model_dump_json(), nx=True)
        if not creado:
            raise RedisError("El identificador de trabajo ya existe")
        return trabajo

    async def obtener(self, job_id: UUID) -> Trabajo | None:
        registro = await self.redis.get(PREFIJO_TRABAJO + str(job_id))
        return Trabajo.model_validate_json(registro) if registro is not None else None

    async def transicionar(
        self,
        job_id: UUID,
        *,
        desde: tuple[EstadoTrabajo, ...],
        hacia: EstadoTrabajo,
        resultado: ResultadoTrabajo | None = None,
        error: str | None = None,
        accion_pendiente: PropuestaReserva | None = None,
        decision_aprobacion: DecisionHumana | None = None,
    ) -> bool:
        """Guardar cambios sólo si Redis conserva uno de los estados esperados."""
        clave = PREFIJO_TRABAJO + str(job_id)
        for _ in range(3):
            async with self.redis.pipeline(transaction=True) as operacion:
                try:
                    await operacion.watch(clave)
                    registro = await operacion.get(clave)
                    if registro is None:
                        return False
                    trabajo = Trabajo.model_validate_json(registro)
                    if trabajo.status not in desde:
                        return False

                    ahora = datetime.now(UTC)
                    trabajo.status = hacia
                    trabajo.result = resultado
                    trabajo.error = error
                    trabajo.accion_pendiente = accion_pendiente
                    if decision_aprobacion is not None:
                        trabajo.decision_aprobacion = decision_aprobacion
                    if hacia == "RUNNING" and trabajo.started_at is None:
                        trabajo.started_at = ahora
                    if hacia in ("DONE", "FAILED"):
                        trabajo.finished_at = ahora

                    operacion.multi()
                    operacion.set(clave, trabajo.model_dump_json())
                    await operacion.execute()
                    return True
                except WatchError:
                    continue
        raise RedisError("No se pudo actualizar el estado del trabajo")

    async def recuperar_huerfanos(self) -> None:
        """Cerrar los trabajos activos que quedaron al salir el proceso anterior."""
        async for clave in self.redis.scan_iter(match=PREFIJO_TRABAJO + "*"):
            job_id = UUID(clave.removeprefix(PREFIJO_TRABAJO))
            await self.transicionar(
                job_id, desde=ESTADOS_ACTIVOS, hacia="FAILED",
                error="La API se reinició antes de completar el trabajo.",
            )
