"""Lanzar cinco trabajos por HTTP y reunir su evidencia de LangSmith."""

import argparse
import asyncio
import math
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4, uuid5

import httpx
from dotenv import load_dotenv
from langsmith import APIError, AsyncClient
from langsmith.utils import LangSmithConflictError, LangSmithError
from pydantic import BaseModel, Field


RAIZ = Path(__file__).resolve().parent.parent
PROMPT = (
    "Quiero viajar a Tokio del 12 al 20 de febrero de 2027. "
    "Somos cuatro personas y buscamos alojamiento, actividades turísticas y gastronomía."
)
CANTIDAD = 5
METRICA_P95 = "latencia_p95_s"


class Medicion(BaseModel):
    peticion: int
    job_id: UUID | None = None
    estado: str = "SIN_RESPUESTA"
    aceptacion_ms: float | None = None
    latencia_traza_s: float | None = None
    inicio_traza: datetime | None = None
    tokens: int | None = None
    costo_usd: Decimal | None = None
    error: str | None = None


class Reporte(BaseModel):
    lote_id: UUID
    proyecto_id: UUID
    iniciado_en: datetime
    estado: str = "INCOMPLETO"
    mediciones: list[Medicion] = Field(default_factory=list)
    latencia_p95_s: float | None = None
    feedback_publicado: bool = False


async def medir_trabajo(http: httpx.AsyncClient, lote_id: UUID, numero: int) -> Medicion:
    medicion = Medicion(peticion=numero)
    try:
        inicio = perf_counter()
        respuesta = await http.post("/tasks", json={"prompt": PROMPT}, headers={"X-Batch-ID": str(lote_id)})
        medicion.aceptacion_ms = round((perf_counter() - inicio) * 1000, 3)
        respuesta.raise_for_status()
        if respuesta.status_code != 202 or respuesta.json()["status"] != "PENDING":
            raise ValueError("Respuesta de aceptación inválida")
        medicion.job_id = UUID(respuesta.json()["job_id"])
        medicion.estado = "PENDING"

        async with asyncio.timeout(360):
            while True:
                respuesta = await http.get(f"/tasks/{medicion.job_id}")
                respuesta.raise_for_status()
                trabajo = respuesta.json()
                if trabajo["job_id"] != str(medicion.job_id) or trabajo.get("lote_id") != str(lote_id):
                    raise ValueError("El trabajo no corresponde al lote")
                medicion.estado = trabajo["status"]
                if medicion.estado == "DONE":
                    if not trabajo.get("result", {}).get("respuesta"):
                        raise ValueError("El trabajo terminó sin respuesta")
                    return medicion
                if medicion.estado not in {"PENDING", "RUNNING"}:
                    medicion.error = f"El trabajo terminó la consulta en {medicion.estado}."
                    return medicion
                await asyncio.sleep(1)
    except httpx.HTTPStatusError as error:
        medicion.error = f"HTTP {error.response.status_code}"
    except TimeoutError:
        medicion.error = "Se agotó el plazo para consultar el trabajo."
    except (httpx.RequestError, ValueError, KeyError, TypeError) as error:
        medicion.error = type(error).__name__
    return medicion


async def lanzar_lote(url: str, reporte: Reporte) -> list[Medicion]:
    async with httpx.AsyncClient(base_url=url, timeout=10) as http:
        contrato = await http.get("/openapi.json")
        contrato.raise_for_status()
        parametros = contrato.json()["paths"]["/tasks"]["post"].get("parameters", [])
        if not any(p["in"] == "header" and p["name"].lower() == "x-batch-id" for p in parametros):
            raise ValueError("Reconstruí la API con docker compose up --build --wait antes de lanzar el lote.")
        print(f"Iniciando cinco trabajos para el lote {reporte.lote_id}", flush=True)
        return await asyncio.gather(*(
            medir_trabajo(http, reporte.lote_id, numero) for numero in range(1, CANTIDAD + 1)
        ))


def calcular_p95(latencias: list[float]) -> float:
    """Percentil de rango más cercano, en segundos."""
    return sorted(latencias)[math.ceil(0.95 * len(latencias)) - 1]


async def reunir_metricas(cliente: AsyncClient, reporte: Reporte, *, espera_s: float = 90) -> None:
    if reporte.estado == "COMPLETO":
        return
    trabajos = {
        str(medicion.job_id): medicion for medicion in reporte.mediciones
        if medicion.estado == "DONE" and medicion.job_id is not None and not medicion.error
        and medicion.latencia_traza_s is None
    }
    limite = perf_counter() + espera_s
    while trabajos:
        trazas = cliente.traces.query(
            project_id=str(reporte.proyecto_id),
            trace_ids=list(trabajos),
            min_start_time=reporte.iniciado_en - timedelta(minutes=1),
            selects=["ID", "START_TIME", "END_TIME", "ERROR", "TAGS", "TOTAL_TOKENS", "TOTAL_COST"],
            page_size=CANTIDAD, timeout=5,
        )
        async for traza in trazas:
            raiz, agregados = traza.root_run, traza.trace_aggregates
            if (
                raiz is None or agregados is None or raiz.id not in trabajos or raiz.error
                or raiz.start_time is None or raiz.end_time is None
                or f"lote:{reporte.lote_id}" not in (raiz.tags or [])
                or agregados.total_tokens is None or agregados.total_cost is None
            ):
                continue
            medicion = trabajos.pop(raiz.id)
            medicion.inicio_traza = raiz.start_time
            medicion.latencia_traza_s = (raiz.end_time - raiz.start_time).total_seconds()
            medicion.tokens = agregados.total_tokens
            medicion.costo_usd = Decimal(str(agregados.total_cost))
        if not trabajos or perf_counter() >= limite:
            break
        await asyncio.sleep(min(2, max(0, limite - perf_counter())))

    ids = {m.job_id for m in reporte.mediciones if m.job_id is not None}
    latencias = [m.latencia_traza_s for m in reporte.mediciones if m.latencia_traza_s is not None]
    if len(reporte.mediciones) == len(ids) == len(latencias) == CANTIDAD and not trabajos:
        reporte.estado = "COMPLETO"
        reporte.latencia_p95_s = calcular_p95(latencias)


async def publicar_p95(cliente: AsyncClient, reporte: Reporte) -> None:
    """Adjuntar una sola medición global a una traza del lote para el dashboard."""
    referencia = reporte.mediciones[0]
    try:
        await cliente.create_feedback(
            run_id=referencia.job_id, session_id=reporte.proyecto_id,
            start_time=referencia.inicio_traza, key=METRICA_P95, score=reporte.latencia_p95_s,
            feedback_id=uuid5(reporte.lote_id, METRICA_P95),
            comment=f"P95 del lote {reporte.lote_id}, rango más cercano sobre cinco trazas, en segundos.",
            extend_trace_retention=False, stop_after_attempt=2,
        )
    except LangSmithConflictError:
        # El mismo reporte pudo haber publicado este feedback antes de reintentarlo.
        pass
    reporte.feedback_publicado = True


def guardar(reporte: Reporte, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(reporte.model_dump_json(indent=2) + "\n", encoding="utf-8")


def mostrar(reporte: Reporte, destino: Path) -> None:
    for m in reporte.mediciones:
        print(
            f"{m.peticion}. {m.job_id or 'sin ID'} {m.estado} | aceptación {m.aceptacion_ms} ms | "
            f"traza {m.latencia_traza_s} s | tokens {m.tokens} | USD {m.costo_usd}"
            + (f" | {m.error}" if m.error else ""),
        )
    print(f"Lote {reporte.estado}. P95 (s): {reporte.latencia_p95_s}. Reporte: {destino}")
    print(f"Feedback {METRICA_P95} publicado: {'sí' if reporte.feedback_publicado else 'no'}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--project-id", type=UUID, help="UUID del proyecto o entorno visible en LangSmith.")
    modos = parser.add_mutually_exclusive_group()
    modos.add_argument("--salida", type=Path, help="Ruta del JSON de la corrida nueva.")
    modos.add_argument("--recolectar", type=Path, help="Completar un JSON existente sin enviar nuevos trabajos.")
    args = parser.parse_args()
    load_dotenv(RAIZ / ".env")
    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        parser.error("Configurá LANGSMITH_API_KEY en el .env local.")
    if args.recolectar is None and os.getenv("LANGSMITH_TRACING", "").lower() not in {"true", "1"}:
        parser.error("Activá LANGSMITH_TRACING=true y reconstruí la API antes de la corrida.")

    async with AsyncClient(timeout_ms=(5000, 5000, 5000, 5000), disable_prompt_cache=True) as cliente:
        if args.recolectar is not None:
            destino = args.recolectar
            reporte = Reporte.model_validate_json(destino.read_text(encoding="utf-8"))
        else:
            if args.project_id is not None:
                proyecto = (await cliente.read_project(project_id=args.project_id)).id
            else:
                proyecto = (
                    await cliente.read_project(project_name=os.getenv("LANGSMITH_PROJECT", "preentrega7"))
                ).id
            reporte = Reporte(lote_id=uuid4(), proyecto_id=proyecto, iniciado_en=datetime.now(UTC))
            destino = args.salida or RAIZ / "evidencias" / f"lote-{reporte.lote_id}.json"
            if destino.exists():
                parser.error("La salida ya existe. Usá otra ruta o --recolectar.")
            reporte.mediciones = await lanzar_lote(args.url, reporte)
            guardar(reporte, destino)

        try:
            await reunir_metricas(cliente, reporte)
            if reporte.estado == "COMPLETO" and not reporte.feedback_publicado:
                await publicar_p95(cliente, reporte)
        except (APIError, LangSmithError) as error:
            print(f"LangSmith no completó la recolección o publicación ({type(error).__name__}).")
        finally:
            guardar(reporte, destino)
            mostrar(reporte, destino)
    return 0 if reporte.estado == "COMPLETO" and reporte.feedback_publicado else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except httpx.ConnectError:
        print("No se pudo conectar a FastAPI. Ejecutá docker compose up --build --wait antes de lanzar el lote.")
        raise SystemExit(1) from None
    except (APIError, LangSmithError, httpx.HTTPError, OSError, ValueError, KeyError) as error:
        print(f"No se pudo iniciar o leer el reporte ({type(error).__name__}).")
        if isinstance(error, ValueError) and str(error).startswith("Reconstruí"):
            print(error)
        raise SystemExit(1) from None
