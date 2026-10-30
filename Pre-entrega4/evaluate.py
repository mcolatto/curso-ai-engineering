"""Evaluación por fuente de chunk del Top-5 híbrido sobre el Golden Set."""

import json
import logging
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from langchain_core.documents import Document

from config import BASE_DIR, DATA_DIR, GOLDEN_SET_PATH, cargar_configuracion
from rag_system import RAGSystem
from schemas import CasoGolden


def cargar_golden(
    path: Path = GOLDEN_SET_PATH, data_dir: Path = DATA_DIR
) -> list[CasoGolden]:
    """Exigir cinco casos válidos y fuentes que existan en el corpus Markdown."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 5:
        raise ValueError("El Golden Set debe ser una lista de exactamente cinco casos")
    cases = [CasoGolden.model_validate(item) for item in data]
    sources = {file.name for file in data_dir.glob("*.md") if file.is_file()}
    for case in cases:
        if case.documento_id_esperado not in sources:
            raise ValueError("Una fuente esperada del Golden Set no existe en el corpus")
    return cases


def calcular_metricas(top_k: list[Document], fuente_esperada: str) -> tuple[float, float]:
    """Precision docente con denominador variable; Recall equivale aquí a Hit."""
    if not top_k:
        return 0.0, 0.0
    coincidencias = sum(doc.metadata["source"] == fuente_esperada for doc in top_k)
    precision = coincidencias / len(top_k)
    recall = 1.0 if coincidencias else 0.0
    return precision, recall


def evaluar(rag: RAGSystem, cases: list[CasoGolden]) -> list[dict]:
    """Recuperar y medir cada caso; propagar cualquier fallo operativo."""
    results = []
    for case in cases:
        top_k = rag.obtener_top_k(case.pregunta)
        precision, recall = calcular_metricas(top_k, case.documento_id_esperado)
        results.append({
            "pregunta": case.pregunta,
            "fuente_esperada": case.documento_id_esperado,
            "fuentes_recuperadas": [doc.metadata["source"] for doc in top_k],
            "precision": precision,
            "recall": recall,
        })
    return results


def imprimir_reporte(results: list[dict]) -> None:
    """Mostrar resultados completos y medias aritméticas, sin colapsar fuentes."""
    print("=== Evaluación del Golden Set: Top-5 híbrido ===")
    print("Precision@5: coincidencias / resultados recuperados; vacío = 0.")
    print("Recall@5: Hit@5, una única fuente esperada por pregunta.")
    for number, result in enumerate(results, start=1):
        print(f"\n{number}. {result['pregunta']}")
        print(f"Fuente esperada: {result['fuente_esperada']}")
        print(f"Fuentes recuperadas: {result['fuentes_recuperadas']}")
        print(f"Precision@5: {result['precision']:.4f}")
        print(f"Recall@5: {result['recall']:.4f}")
    print("\n=== Promedios ===")
    print(f"Precision@5: {mean(result['precision'] for result in results):.4f}")
    print(f"Recall@5: {mean(result['recall'] for result in results):.4f}")


def main() -> int:
    """Cargar configuración, evaluar los cinco casos y cerrar la conexión."""
    load_dotenv(BASE_DIR / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        cases = cargar_golden()
        rag = RAGSystem(cargar_configuracion())
        try:
            results = evaluar(rag, cases)
        finally:
            rag.cerrar()
        # No presentar promedios parciales si falla alguna consulta o el cierre.
        imprimir_reporte(results)
    except Exception as error:
        print(f"Evaluación incompleta ({type(error).__name__}). Revisá configuración, ingesta y conexión.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
