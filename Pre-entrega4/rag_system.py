"""Recuperación híbrida de fragmentos mediante BM25 y Pinecone."""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import EMBEDDING_DIM, EMBEDDING_MODEL, TOP_K, Configuracion
from ingest import cargar_y_fragmentar, namespace_coincide


class RAGSystem:
    """Combinar rankings sobre el mismo corpus, sin generar respuestas con un LLM."""

    def __init__(self, config: Configuracion):
        self.config = config
        self.chunks = cargar_y_fragmentar(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        self.expected_ids = {chunk.metadata["chunk_id"] for chunk in self.chunks}
        self._index = None
        self._retriever = None

    def _preparar_recuperador(self) -> None:
        """Abrir el índice existente y preparar ambas ramas en la primera consulta."""
        client = Pinecone(api_key=self.config.pinecone_api_key.get_secret_value())
        if not client.has_index(self.config.index_name):
            raise ValueError("No existe el índice. Ejecutá ingest.py antes de consultar")
        description = client.describe_index(self.config.index_name).to_dict()
        serverless = description.get("spec", {}).get("serverless", {})
        if (
            description.get("dimension") != EMBEDDING_DIM
            or description.get("metric") != "cosine"
            or serverless.get("cloud") != self.config.cloud
            or serverless.get("region") != self.config.region
            or not description["status"]["ready"]
        ):
            raise ValueError("El índice no está listo o no coincide con la configuración")
        self._index = client.Index(host=description["host"])
        self._comprobar_corpus()

        embeddings = OpenAIEmbeddings(
            api_key=self.config.openai_api_key,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIM,
            request_timeout=30,
            max_retries=0,
            check_embedding_ctx_length=False,
        )
        store = PineconeVectorStore(
            index=self._index,
            embedding=embeddings,
            namespace=self.config.namespace,
            text_key="text",
        )
        bm25 = BM25Retriever.from_documents(self.chunks, k=TOP_K)
        vectorial = store.as_retriever(search_kwargs={"k": TOP_K})
        self._retriever = EnsembleRetriever(
            retrievers=[bm25, vectorial],
            weights=self.config.weights,
            id_key="chunk_id",
        )

    def _comprobar_corpus(self) -> None:
        """Impedir consultas sobre registros distintos del corpus local fragmentado."""
        if not namespace_coincide(self._index, self.config.namespace, self.expected_ids):
            raise ValueError("El namespace no coincide con el corpus local. Ejecutá ingest.py nuevamente")

    def obtener_top_k(self, query: str) -> list[Document]:
        """Validar la consulta, fusionar rankings por ID y devolver hasta cinco chunks."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("La pregunta debe contener texto no vacío")
        if self._retriever is None:
            self._preparar_recuperador()
        else:
            self._comprobar_corpus()
        return self._retriever.invoke(query)[:TOP_K]

    def cerrar(self) -> None:
        """Liberar la conexión al índice cuando termine el uso del sistema."""
        if self._index is not None:
            self._index.close()
        self._index = None
        self._retriever = None
