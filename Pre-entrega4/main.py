"""Consulta exploratoria del Top-5 híbrido, independiente del Golden Set."""

import argparse
import logging

from dotenv import load_dotenv
from langchain_core.documents import Document

from config import BASE_DIR, cargar_configuracion
from rag_system import RAGSystem


def mostrar_resultados(documents: list[Document]) -> None:
    """Mostrar el ranking y hasta 200 caracteres del texto de cada chunk."""
    if not documents:
        print("No se recuperaron resultados.")
        return
    print("=== Top-5 híbrido ===")
    for position, document in enumerate(documents, start=1):
        print(f"\n{position}. Fuente: {document.metadata['source']}")
        print(f"Categoría: {document.metadata['categoria']}")
        print(f"ID: {document.metadata['chunk_id']}")
        print(f"Extracto: {document.page_content[:200]}")


def main(argv: list[str] | None = None) -> int:
    """Leer una pregunta, recuperar sus fragmentos y cerrar la conexión."""
    parser = argparse.ArgumentParser(description="Consultar los documentos con búsqueda híbrida")
    parser.add_argument("pregunta", nargs="?", help="Pregunta exploratoria entre comillas")
    args = parser.parse_args(argv)
    load_dotenv(BASE_DIR / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        query = args.pregunta if args.pregunta is not None else input("Ingresá una pregunta: ")
        if not query.strip():
            print("La pregunta debe contener texto no vacío.")
            return 1
        rag = RAGSystem(cargar_configuracion())
        try:
            documents = rag.obtener_top_k(query)
        finally:
            rag.cerrar()
        mostrar_resultados(documents)
    except EOFError:
        print("No se recibió una pregunta.")
        return 1
    except Exception as error:
        print(f"Consulta incompleta ({type(error).__name__}). Revisá configuración, ingesta y conexión.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
