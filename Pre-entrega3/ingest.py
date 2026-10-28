"""Carga y fragmentación de los documentos utilizados como contexto RAG."""

import argparse
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import APIError

from config import (
    BASE_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, DATA_DIR,
    PERSIST_DIR, TOKEN_ENCODING, Configuracion, cargar_configuracion,
)

logger = logging.getLogger(__name__)


def cargar_documentos(data_dir: Path = DATA_DIR) -> list[Document]:
    """Leer archivos .txt y .md no vacíos, conservando su nombre como fuente."""
    if not data_dir.is_dir():
        raise FileNotFoundError("No existe el directorio de documentos indicado")

    documentos = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            contenido = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise ValueError(f"No se pudo leer el archivo {path.name} como texto UTF-8") from None
        if contenido.strip():
            documentos.append(Document(
                page_content=contenido,
                metadata={"source": path.name},
            ))

    if not documentos:
        raise ValueError("No hay archivos .txt o .md con contenido en el directorio")
    return documentos


def crear_fragmentador() -> RecursiveCharacterTextSplitter:
    """Medir fragmentos en tokens con el tokenizer y límites compartidos."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKEN_ENCODING,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def fragmentar_documentos(documentos: list[Document]) -> list[Document]:
    """Dividir documentos conservando los metadatos en cada fragmento."""
    if not documentos:
        raise ValueError("No hay documentos para fragmentar")
    return crear_fragmentador().split_documents(documentos)


def crear_embeddings(config: Configuracion) -> OpenAIEmbeddings:
    """Usar el modelo compartido sin reintentos automáticos del cliente."""
    return OpenAIEmbeddings(
        api_key=config.api_key.get_secret_value(),
        model=config.embedding_model,
        timeout=config.timeout_seconds,
        max_retries=0,
    )


def abrir_indice(config: Configuracion, persist_dir: Path = PERSIST_DIR) -> Chroma:
    """Abrir una colección poblada y compatible sin iniciar una ingesta."""
    if not persist_dir.is_dir():
        raise ValueError("No hay un índice preparado. Ejecutá ingest.py antes de consultar.")
    client = chromadb.PersistentClient(
        path=str(persist_dir), settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    except NotFoundError:
        raise ValueError("No existe la colección del proyecto. Ejecutá ingest.py.") from None
    if (collection.metadata or {}).get("embedding_model") != config.embedding_model:
        raise ValueError("El modelo de embeddings no coincide con el índice. Usá ingest.py --rebuild.")
    if collection.count() == 0:
        raise ValueError("La colección está vacía. Ejecutá ingest.py antes de consultar.")
    return Chroma(
        client=client, collection_name=COLLECTION_NAME,
        embedding_function=crear_embeddings(config),
        create_collection_if_not_exists=False,
    )


def preparar_indice(
    config: Configuracion,
    data_dir: Path = DATA_DIR,
    persist_dir: Path = PERSIST_DIR,
    rebuild: bool = False,
) -> Chroma:
    """Crear o reutilizar la colección; reconstruirla solo si se solicita."""
    client = chromadb.PersistentClient(
        path=str(persist_dir), settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    except NotFoundError:
        collection = None

    if collection is not None and not rebuild:
        saved_model = (collection.metadata or {}).get("embedding_model")
        if saved_model != config.embedding_model:
            raise ValueError("El modelo de embeddings no coincide con el índice. Usá --rebuild.")
        count = collection.count()
        if count > 0:
            logger.info("Índice existente reutilizado: %s fragmentos", count)
            return Chroma(
                client=client, collection_name=COLLECTION_NAME,
                embedding_function=crear_embeddings(config),
                create_collection_if_not_exists=False,
            )

    # Leer y fragmentar antes de borrar una colección que se quiere reconstruir.
    documents = cargar_documentos(data_dir)
    chunks = fragmentar_documentos(documents)
    logger.info("Documentos cargados: %s; fragmentos generados: %s", len(documents), len(chunks))
    if collection is not None:
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        COLLECTION_NAME, embedding_function=None,
        metadata={"embedding_model": config.embedding_model},
    )
    vectorstore = Chroma(
        client=client, collection_name=COLLECTION_NAME,
        embedding_function=crear_embeddings(config),
        create_collection_if_not_exists=False,
    )
    try:
        vectorstore.add_documents(
            chunks, ids=[f"fragmento-{index}" for index in range(len(chunks))],
        )
    except Exception:
        # Un fallo no debe dejar un índice parcial que luego parezca reutilizable.
        client.delete_collection(COLLECTION_NAME)
        raise
    logger.info("Índice %s: %s fragmentos", "reconstruido" if rebuild else "creado", len(chunks))
    return vectorstore


def main() -> None:
    """Ejecutar la ingesta local; --rebuild reemplaza la colección del proyecto."""
    parser = argparse.ArgumentParser(description="Preparar el índice local de documentos RAG")
    parser.add_argument("--rebuild", action="store_true", help="Reconstruir la colección del proyecto")
    args = parser.parse_args()
    load_dotenv(BASE_DIR / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        preparar_indice(cargar_configuracion(), rebuild=args.rebuild)
    except (ValueError, FileNotFoundError):
        print("Error: revisá .env, los archivos de data y la compatibilidad del índice. Para reconstruirlo, usá --rebuild.")
    except APIError:
        print("Error de OpenAI al generar embeddings. Revisá credenciales, saldo, conexión y modelo.")


if __name__ == "__main__":
    main()
