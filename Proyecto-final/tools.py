"""Consultas asíncronas a las fuentes turísticas locales y simuladas."""

import asyncio
import json
import math
from pathlib import Path

from langchain_core.tools import tool

from config import BASE_DIR
from schemas import (
    Coordenadas, Observacion, OpcionHotel, OpcionTuristica, ResultadoHoteleria, ResultadoTurismo,
)

DATA_DIR = BASE_DIR / "data"
TOKYO_STATION = Coordenadas(latitud=35.6812, longitud=139.7671)


def calcular_distancia_km(origen: Coordenadas, destino: Coordenadas) -> float:
    """Calcular distancia en línea recta con Haversine, sin servicios de mapas."""
    lat1, lat2 = math.radians(origen.latitud), math.radians(destino.latitud)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(destino.longitud - origen.longitud)
    value = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    return round(6371.0 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, value)))), 3)


def leer_catalogo(path: Path) -> list[dict]:
    """Leer registros con destino explícito; no completar datos faltantes."""
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("El catálogo debe contener una lista")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Cada registro debe ser un objeto")
        destino = record.get("destino")
        if not isinstance(destino, str) or not destino.strip():
            raise ValueError("Cada registro debe indicar su destino")
    return records


def normalizar_destino(destino: str) -> str:
    """Comparar destinos sin distinguir mayúsculas ni espacios exteriores."""
    if not destino.strip():
        raise ValueError("El destino debe contener texto")
    return destino.strip().casefold()


@tool
async def buscar_hoteles(destino: str) -> dict:
    """Recuperar alojamientos del destino desde un catálogo local ficticio.
    Devuelve IDs, fuente, capacidad, coordenadas y períodos conocidos, sin evaluar
    compatibilidad con fechas, cantidad de viajeros o preferencias del usuario.
    No confirma disponibilidad comercial ni realiza reservas. Una búsqueda vacía
    es completada; un fallo de lectura o validación se informa como error.
    """
    key = normalizar_destino(destino)
    try:
        records = await asyncio.to_thread(leer_catalogo, DATA_DIR / "hoteles.json")
        # Validar el catálogo completo antes de devolver una búsqueda exitosa.
        options = [OpcionHotel.model_validate(record) for record in records]
        hotels = [option for record, option in zip(records, options)
                  if normalizar_destino(record["destino"]) == key]
        result = ResultadoHoteleria(estado_busqueda="completada", hoteles=hotels)
    except (OSError, UnicodeError, ValueError):
        result = ResultadoHoteleria(estado_busqueda="error", observaciones=[
            Observacion(tipo="limitacion", descripcion="No se pudo consultar el catálogo local de hoteles")
        ])
    return result.model_dump(mode="json")


@tool
async def buscar_turismo(destino: str) -> dict:
    """Recuperar actividades, miradores y gastronomía del destino desde datos ficticios.
    Conserva fuente, coordenadas, períodos y restricciones conocidos para que el
    Analista evalúe la compatibilidad. No filtra por fechas ni garantiza reservas.
    Devuelve opciones vacías si el destino no figura; informa error si no puede
    consultar o validar el catálogo local.
    """
    key = normalizar_destino(destino)
    try:
        records = await asyncio.to_thread(leer_catalogo, DATA_DIR / "turismo.json")
        options = [OpcionTuristica.model_validate(record) for record in records]
        tourism = [option for record, option in zip(records, options)
                   if normalizar_destino(record["destino"]) == key]
        result = ResultadoTurismo(estado_busqueda="completada", opciones=tourism)
    except (OSError, UnicodeError, ValueError):
        result = ResultadoTurismo(estado_busqueda="error", observaciones=[
            Observacion(tipo="limitacion", descripcion="No se pudo consultar el catálogo local de turismo")
        ])
    return result.model_dump(mode="json")
