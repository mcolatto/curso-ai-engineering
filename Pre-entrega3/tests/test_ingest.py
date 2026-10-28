"""Pruebas de lectura y chunking sin embeddings ni base vectorial."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, TOKEN_ENCODING, Configuracion
from ingest import cargar_documentos, crear_embeddings, crear_fragmentador, fragmentar_documentos, preparar_indice


class CargaDocumentosTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_dir = Path(self.temp_dir.name)

    def test_carga_txt_y_md_en_orden_y_conserva_contenido_y_fuentes(self):
        (self.data_dir / "b.txt").write_text("\nInformación de Redis.  ", encoding="utf-8")
        (self.data_dir / "a.md").write_text("# Recuperación semántica", encoding="utf-8")
        (self.data_dir / "ignorar.csv").write_text("dato", encoding="utf-8")
        (self.data_dir / "subcarpeta.txt").mkdir()

        docs = cargar_documentos(self.data_dir)
        self.assertEqual([doc.metadata["source"] for doc in docs], ["a.md", "b.txt"])
        self.assertEqual(docs[0].page_content, "# Recuperación semántica")
        self.assertEqual(docs[1].page_content, "\nInformación de Redis.  ")

    def test_omite_archivos_sin_contenido(self):
        (self.data_dir / "vacio.txt").write_text("", encoding="utf-8")
        (self.data_dir / "espacios.md").write_text(" \n\t", encoding="utf-8")
        (self.data_dir / "valido.txt").write_text("Texto útil", encoding="utf-8")
        docs = cargar_documentos(self.data_dir)
        self.assertEqual([doc.metadata["source"] for doc in docs], ["valido.txt"])

    def test_rechaza_directorio_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            cargar_documentos(self.data_dir / "inexistente")

    def test_rechaza_directorio_sin_archivos_utilizables(self):
        with self.assertRaisesRegex(ValueError, "No hay archivos"):
            cargar_documentos(self.data_dir)
        (self.data_dir / "vacio.txt").write_text(" \n\t", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "No hay archivos"):
            cargar_documentos(self.data_dir)

    def test_informa_archivo_con_codificacion_invalida(self):
        (self.data_dir / "invalido.txt").write_bytes(b"\xff\xfe")
        with self.assertRaisesRegex(ValueError, "invalido.txt.*UTF-8"):
            cargar_documentos(self.data_dir)

    def test_informa_error_de_lectura_sin_mostrar_contenido_de_la_excepcion(self):
        (self.data_dir / "archivo.txt").write_text("Texto", encoding="utf-8")
        with patch.object(Path, "read_text", side_effect=OSError("detalle-privado")):
            with self.assertRaises(ValueError) as raised:
                cargar_documentos(self.data_dir)
        self.assertIn("archivo.txt", str(raised.exception))
        self.assertNotIn("detalle-privado", str(raised.exception))


class FragmentacionTests(unittest.TestCase):
    def test_configura_conteo_en_tokens_y_solapamiento(self):
        splitter = crear_fragmentador()
        self.assertEqual(splitter._chunk_size, CHUNK_SIZE)
        self.assertEqual(splitter._chunk_overlap, CHUNK_OVERLAP)
        sample = "Recuperación semántica y bases vectoriales."
        expected = len(tiktoken.get_encoding(TOKEN_ENCODING).encode(sample))
        self.assertEqual(splitter._length_function(sample), expected)

    def test_documento_corto_no_se_rellena_hasta_500_tokens(self):
        doc = Document(page_content="Redis almacena datos.", metadata={"source": "redis.txt"})
        chunks = fragmentar_documentos([doc])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, doc.page_content)
        self.assertEqual(chunks[0].metadata, doc.metadata)

    def test_divide_texto_largo_sin_superar_limite_y_conserva_metadatos(self):
        text = "Los documentos aportan contexto para responder consultas semánticas.\n\n" * 150
        doc = Document(page_content=text, metadata={"source": "rag.txt", "tema": "RAG"})
        chunks = fragmentar_documentos([doc])
        encoding = tiktoken.get_encoding(TOKEN_ENCODING)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.page_content.strip())
            self.assertLessEqual(len(encoding.encode(chunk.page_content)), CHUNK_SIZE)
            self.assertEqual(chunk.metadata, doc.metadata)

    def test_no_mezcla_fuentes_de_documentos_distintos(self):
        docs = [
            Document(page_content="Texto de RAG.", metadata={"source": "rag.txt"}),
            Document(page_content="Texto de vectores.", metadata={"source": "vectordb.txt"}),
        ]
        chunks = fragmentar_documentos(docs)
        self.assertEqual([chunk.metadata["source"] for chunk in chunks], ["rag.txt", "vectordb.txt"])
        self.assertEqual([chunk.page_content for chunk in chunks], [doc.page_content for doc in docs])

    def test_rechaza_lista_de_documentos_vacia(self):
        with self.assertRaisesRegex(ValueError, "No hay documentos"):
            fragmentar_documentos([])


class EmbeddingsSimulados(Embeddings):
    """Vectores deterministas para comprobar persistencia, no calidad semántica."""

    def __init__(self):
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [[float(len(text)), float(sum(map(ord, text)) % 101), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(sum(map(ord, text)) % 101), 1.0]


class PersistenciaTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.data_dir = Path(temp_dir.name) / "data"
        self.data_dir.mkdir()
        self.persist_dir = Path(temp_dir.name) / "vectorstore"
        (self.data_dir / "rag.txt").write_text("RAG utiliza contexto recuperado.", encoding="utf-8")
        self.config = Configuracion(api_key="clave-ficticia")
        self.embeddings = EmbeddingsSimulados()
        patcher = patch("ingest.crear_embeddings", return_value=self.embeddings)
        self.embedding_factory = patcher.start()
        self.addCleanup(patcher.stop)

    def test_guarda_fragmentos_y_fuentes_en_chroma_real(self):
        with self.assertLogs("ingest", level="INFO") as logs:
            store = preparar_indice(self.config, self.data_dir, self.persist_dir)
        result = store.get()
        self.assertEqual(result["documents"], ["RAG utiliza contexto recuperado."])
        self.assertEqual(result["metadatas"], [{"source": "rag.txt"}])
        self.assertEqual(self.embeddings.document_calls, 1)
        self.assertTrue(self.persist_dir.is_dir())
        self.assertIn("Índice creado", "\n".join(logs.output))

    def test_reabre_indice_sin_releer_corpus_ni_generar_embeddings(self):
        first = preparar_indice(self.config, self.data_dir, self.persist_dir)
        original = first.get()
        (self.data_dir / "rag.txt").unlink()
        with self.assertLogs("ingest", level="INFO") as logs:
            reopened = preparar_indice(self.config, self.data_dir, self.persist_dir)
        self.assertEqual(reopened.get(), original)
        self.assertEqual(self.embeddings.document_calls, 1)
        self.assertIn("reutilizado", "\n".join(logs.output))

    def test_reconstruccion_reemplaza_solo_la_coleccion_del_proyecto(self):
        preparar_indice(self.config, self.data_dir, self.persist_dir)
        client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False),
        )
        client.create_collection("otra_coleccion", embedding_function=None)
        (self.data_dir / "rag.txt").unlink()
        (self.data_dir / "vectores.md").write_text("Una base vectorial almacena vectores.", encoding="utf-8")
        store = preparar_indice(self.config, self.data_dir, self.persist_dir, rebuild=True)
        self.assertEqual(store.get()["documents"], ["Una base vectorial almacena vectores."])
        self.assertEqual(store.get()["metadatas"], [{"source": "vectores.md"}])
        self.assertEqual(self.embeddings.document_calls, 2)
        self.assertEqual(client.get_collection("otra_coleccion", embedding_function=None).count(), 0)

    def test_rechaza_modelo_incompatible_sin_llamar_embeddings(self):
        preparar_indice(self.config, self.data_dir, self.persist_dir)
        incompatible = Configuracion(api_key="clave-ficticia", embedding_model="otro-modelo")
        self.embedding_factory.reset_mock()
        with self.assertRaisesRegex(ValueError, "modelo de embeddings.*--rebuild"):
            preparar_indice(incompatible, self.data_dir, self.persist_dir)
        self.embedding_factory.assert_not_called()

    def test_rebuild_permite_cambiar_modelo_y_registra_metadata(self):
        preparar_indice(self.config, self.data_dir, self.persist_dir)
        config = Configuracion(api_key="clave-ficticia", embedding_model="otro-modelo")
        preparar_indice(config, self.data_dir, self.persist_dir, rebuild=True)
        client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
        self.assertEqual(collection.metadata["embedding_model"], "otro-modelo")

    def test_carpeta_existente_sin_coleccion_no_impide_indexar(self):
        self.persist_dir.mkdir()
        store = preparar_indice(self.config, self.data_dir, self.persist_dir)
        self.assertEqual(len(store.get()["ids"]), 1)

    def test_coleccion_vacia_se_indexa(self):
        client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False),
        )
        client.create_collection(
            COLLECTION_NAME, embedding_function=None,
            metadata={"embedding_model": self.config.embedding_model},
        )
        store = preparar_indice(self.config, self.data_dir, self.persist_dir)
        self.assertEqual(len(store.get()["ids"]), 1)

    def test_rebuild_con_corpus_invalido_no_borra_indice_existente(self):
        store = preparar_indice(self.config, self.data_dir, self.persist_dir)
        original = store.get()
        (self.data_dir / "rag.txt").unlink()
        with self.assertRaisesRegex(ValueError, "No hay archivos"):
            preparar_indice(self.config, self.data_dir, self.persist_dir, rebuild=True)
        self.assertEqual(store.get(), original)

    def test_fallo_de_embeddings_no_deja_coleccion_parcial(self):
        with patch.object(self.embeddings, "embed_documents", side_effect=RuntimeError("fallo simulado")):
            with self.assertRaisesRegex(RuntimeError, "fallo simulado"):
                preparar_indice(self.config, self.data_dir, self.persist_dir)
        client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False),
        )
        with self.assertRaises(NotFoundError):
            client.get_collection(COLLECTION_NAME, embedding_function=None)


class ClienteEmbeddingsTests(unittest.TestCase):
    def test_transmite_modelo_compartido_timeout_y_reintentos_desactivados(self):
        config = Configuracion(api_key="clave-ficticia", timeout_seconds=15)
        with patch("ingest.OpenAIEmbeddings") as constructor:
            crear_embeddings(config)
        constructor.assert_called_once_with(
            api_key="clave-ficticia", model="text-embedding-3-small", timeout=15, max_retries=0,
        )


if __name__ == "__main__":
    unittest.main()
