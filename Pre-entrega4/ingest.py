"""Carga y fragmentación del corpus compartido por ingesta y recuperación."""

import hashlib
import json
import logging
import math
import time
from pathlib import Path
from collections.abc import Callable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from config import (
    BASE_DIR, DATA_DIR, EMBEDDING_DIM, EMBEDDING_MODEL, TOKEN_ENCODING,
    Configuracion, cargar_configuracion,
)
from schemas import MetadataChunk

logger = logging.getLogger(__name__)
WAIT_TIMEOUT = 120
POLL_INTERVAL = 2
BATCH_SIZE = 100


def cargar_documentos(data_dir: Path = DATA_DIR) -> list[Document]:
    """Leer los Markdown en orden estable y conservar su procedencia."""
    if not data_dir.is_dir():
        raise ValueError(f"No existe la carpeta de documentos: {data_dir}")

    documents = []
    for path in sorted(data_dir.glob("*.md")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError(f"No se pudo leer el documento {path.name} en UTF-8") from error
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": path.name, "categoria": path.stem},
            ))

    if not documents:
        raise ValueError("El corpus no contiene documentos Markdown con texto")
    return documents


def cargar_y_fragmentar(
    data_dir: Path = DATA_DIR,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Preparar los mismos chunks y metadatos para Pinecone y BM25."""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("El tamaño debe ser positivo y el overlap menor que el tamaño")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKEN_ENCODING,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        allowed_special=set(),
        disallowed_special=(),
        strip_whitespace=False,
    )
    chunks = []
    for document in cargar_documentos(data_dir):
        for position, text in enumerate(splitter.split_text(document.page_content)):
            if not text.strip():
                continue
            # Incluir la posición distingue fragmentos repetidos de un mismo archivo.
            identity = json.dumps(
                [document.metadata["source"], position, text], ensure_ascii=False
            )
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            metadata = MetadataChunk(
                text=text,
                source=document.metadata["source"],
                categoria=document.metadata["categoria"],
                chunk_id=chunk_id,
            )
            chunks.append(Document(page_content=text, metadata=metadata.model_dump()))
    return chunks


def esperar(condicion: Callable[[], bool], operacion: str) -> None:
    """Consultar periódicamente una condición, con una espera máxima acotada."""
    deadline = time.monotonic() + WAIT_TIMEOUT
    while True:
        if condicion():
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Se agotó la espera de {operacion}")
        time.sleep(min(POLL_INTERVAL, remaining))


def preparar_indice(client: Pinecone, config: Configuracion):
    """Crear o verificar un índice Serverless compatible, sin recrearlo."""
    if not client.has_index(config.index_name):
        client.create_index(
            name=config.index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.cloud, region=config.region),
            timeout=WAIT_TIMEOUT,
        )

    def indice_listo() -> bool:
        description = client.describe_index(config.index_name).to_dict()
        serverless = description.get("spec", {}).get("serverless", {})
        if (
            description.get("dimension") != EMBEDDING_DIM
            or description.get("metric") != "cosine"
            or serverless.get("cloud") != config.cloud
            or serverless.get("region") != config.region
        ):
            raise ValueError("El índice existente no coincide con la configuración requerida")
        return description["status"]["ready"]

    esperar(indice_listo, "creación del índice")
    host = client.describe_index(config.index_name).host
    return client.Index(host=host)


def obtener_ids_namespace(index, namespace: str) -> set[str]:
    """Reunir los IDs paginados exclusivamente del namespace indicado."""
    return {chunk_id for page in index.list(namespace=namespace) for chunk_id in page}


def namespace_coincide(index, namespace: str, expected_ids: set[str]) -> bool:
    """Comprobar cantidad e identidad de registros, contemplando visibilidad eventual."""
    stats = index.describe_index_stats()
    count = stats["namespaces"].get(namespace, {}).get("vector_count", 0)
    return count == len(expected_ids) and obtener_ids_namespace(index, namespace) == expected_ids


def preparar_registros(chunks: list[Document], embeddings: OpenAIEmbeddings) -> list[dict]:
    """Validar todo el corpus y sus vectores antes de permitir el borrado."""
    if not chunks:
        raise ValueError("No hay fragmentos para ingerir")
    metadata = [MetadataChunk(**chunk.metadata).model_dump() for chunk in chunks]
    ids = [item["chunk_id"] for item in metadata]
    if len(set(ids)) != len(ids):
        raise ValueError("Los IDs de los fragmentos deben ser únicos")
    if any(chunk.page_content != item["text"] for chunk, item in zip(chunks, metadata)):
        raise ValueError("El texto del fragmento y sus metadatos deben coincidir")

    vectors = embeddings.embed_documents([item["text"] for item in metadata])
    if len(vectors) != len(chunks):
        raise ValueError("La cantidad de embeddings no coincide con los fragmentos")
    records = []
    for item, vector in zip(metadata, vectors):
        if len(vector) != EMBEDDING_DIM or not all(math.isfinite(value) for value in vector):
            raise ValueError("Los embeddings deben tener dimensión 1536 y valores finitos")
        records.append({"id": item["chunk_id"], "values": vector, "metadata": item})
    return records


def ingerir(config: Configuracion) -> int:
    """Reemplazar únicamente el namespace configurado y verificar su contenido."""
    chunks = cargar_y_fragmentar(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    logger.info("Fragmentos preparados: %s", len(chunks))
    embeddings = OpenAIEmbeddings(
        api_key=config.openai_api_key,
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
        request_timeout=30,
        max_retries=0,
        # Los chunks ya están limitados por tokens; enviar su texto sin retokenizarlo.
        check_embedding_ctx_length=False,
    )
    records = preparar_registros(chunks, embeddings)
    client = Pinecone(api_key=config.pinecone_api_key.get_secret_value())
    with preparar_indice(client, config) as index:
        # No intentar borrar un namespace inexistente en un índice nuevo.
        stats = index.describe_index_stats()
        if config.namespace in stats["namespaces"]:
            index.delete(delete_all=True, namespace=config.namespace)
        esperar(
            lambda: namespace_coincide(index, config.namespace, set()),
            "vaciado del namespace",
        )
        for start in range(0, len(records), BATCH_SIZE):
            index.upsert(
                vectors=records[start:start + BATCH_SIZE], namespace=config.namespace
            )
        expected_ids = {record["id"] for record in records}
        esperar(
            lambda: namespace_coincide(index, config.namespace, expected_ids),
            "visibilidad de los registros",
        )
    logger.info("Ingesta completa: %s registros verificados", len(records))
    return len(records)


def main() -> int:
    """Cargar el entorno y ejecutar la ingesta con errores controlados."""
    load_dotenv(BASE_DIR / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        ingerir(cargar_configuracion())
    except Exception as error:
        # Las excepciones de servicios externos pueden contener datos sensibles.
        logger.error("Ingesta incompleta (%s). Revisá configuración y conexión.", type(error).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
