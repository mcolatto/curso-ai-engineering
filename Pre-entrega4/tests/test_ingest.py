"""Pruebas locales del cargador, tokenizer e identidad de los fragmentos."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import tiktoken
from langchain_core.documents import Document

from config import EMBEDDING_DIM, TOKEN_ENCODING, Configuracion
from ingest import (
    cargar_documentos, cargar_y_fragmentar, esperar, ingerir, main,
    namespace_coincide, obtener_ids_namespace, preparar_indice, preparar_registros,
)
from schemas import MetadataChunk


class IngestTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data_dir = Path(directory.name)

    def escribir(self, name, text):
        (self.data_dir / name).write_text(text, encoding="utf-8")

    def test_carga_solo_markdown_en_orden_con_texto_original(self):
        self.escribir("b.md", "  Texto con acentos: información.\n")
        self.escribir("a.md", "# Primero\n")
        self.escribir("ignorado.txt", "No ingerir")
        self.escribir("golden_dataset.json", "[]")
        documents = cargar_documentos(self.data_dir)
        self.assertEqual([doc.metadata["source"] for doc in documents], ["a.md", "b.md"])
        self.assertEqual(documents[1].page_content, "  Texto con acentos: información.\n")
        self.assertEqual(documents[1].metadata["categoria"], "b")

    def test_chunk_corto_conserva_texto_y_metadatos_validos(self):
        text = "  # Memoria\nContenido breve.\n"
        self.escribir("memory.md", text)
        chunks = cargar_y_fragmentar(self.data_dir)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, text)
        metadata = MetadataChunk(**chunks[0].metadata)
        self.assertEqual(metadata.text, text)
        self.assertEqual((metadata.source, metadata.categoria), ("memory.md", "memory"))

    def test_ids_reproducibles_y_sensibles_a_fuente_y_contenido(self):
        self.escribir("a.md", "Texto compartido")
        self.escribir("b.md", "Texto compartido")
        first = cargar_y_fragmentar(self.data_dir)
        repeated = cargar_y_fragmentar(self.data_dir)
        ids = [chunk.metadata["chunk_id"] for chunk in first]
        self.assertEqual(ids, [chunk.metadata["chunk_id"] for chunk in repeated])
        self.assertEqual(len(set(ids)), 2)
        self.escribir("a.md", "Texto modificado")
        changed = cargar_y_fragmentar(self.data_dir)
        self.assertNotEqual(ids[0], changed[0].metadata["chunk_id"])
        self.assertEqual(ids[1], changed[1].metadata["chunk_id"])

    def test_limite_en_tokens_y_overlap_configurable(self):
        self.escribir("tokens.md", " palabra" * 60)
        encoding = tiktoken.get_encoding(TOKEN_ENCODING)
        chunks = cargar_y_fragmentar(self.data_dir, chunk_size=12, chunk_overlap=4)
        without_overlap = cargar_y_fragmentar(self.data_dir, chunk_size=12, chunk_overlap=0)
        self.assertGreater(len(chunks), len(without_overlap))
        self.assertTrue(all(len(encoding.encode(chunk.page_content)) <= 12 for chunk in chunks))
        first = encoding.encode(chunks[0].page_content)
        second = encoding.encode(chunks[1].page_content)
        self.assertEqual(first[-4:], second[:4])
        self.assertEqual(len({chunk.metadata["chunk_id"] for chunk in chunks}), len(chunks))

    def test_marcadores_especiales_se_tratan_como_texto(self):
        text = "Ejemplo: <|endoftext|> y <|fim_prefix|>."
        self.escribir("ejemplo.md", text)
        self.assertEqual(cargar_y_fragmentar(self.data_dir)[0].page_content, text)

    def test_informa_corpus_vacio_y_carpeta_inexistente(self):
        self.escribir("vacio.md", " \n\t")
        with self.assertRaisesRegex(ValueError, "no contiene"):
            cargar_y_fragmentar(self.data_dir)
        with self.assertRaisesRegex(ValueError, "No existe"):
            cargar_documentos(self.data_dir / "inexistente")

    def test_informa_utf8_invalido_y_error_de_lectura(self):
        (self.data_dir / "invalido.md").write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "invalido.md"):
            cargar_documentos(self.data_dir)
        with patch.object(Path, "read_text", side_effect=OSError("fallo simulado")):
            with self.assertRaisesRegex(ValueError, "No se pudo leer"):
                cargar_documentos(self.data_dir)

    def test_rechaza_parametros_incompatibles(self):
        for size, overlap in ((0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)):
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    cargar_y_fragmentar(self.data_dir, size, overlap)


class IngestaCloudTests(unittest.TestCase):
    def setUp(self):
        self.config = Configuracion(
            openai_api_key="openai-ficticia", pinecone_api_key="pinecone-ficticia",
            namespace="entrega-test", chunk_size=700, chunk_overlap=80,
        )
        self.chunks = [
            Document(
                page_content=f"Texto {number}",
                metadata={
                    "text": f"Texto {number}", "source": "memory.md",
                    "categoria": "memory", "chunk_id": f"id-{number}",
                },
            ) for number in range(3)
        ]
        self.embeddings = MagicMock()
        self.embeddings.embed_documents.return_value = [[0.1] * EMBEDDING_DIM for _ in self.chunks]
        self.client = MagicMock()
        self.description = {
            "dimension": EMBEDDING_DIM, "metric": "cosine",
            "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
            "status": {"ready": True},
        }
        self.client.describe_index.return_value.to_dict.side_effect = lambda: self.description
        self.client.describe_index.return_value.host = "host-ficticio"
        self.index = self.client.Index.return_value
        self.index.__enter__.return_value = self.index
        self.state = {"entrega-test": {"antiguo": {}}, "otro-namespace": {"conservar": {}}}
        self.index.describe_index_stats.side_effect = lambda: {
            "namespaces": {name: {"vector_count": len(records)} for name, records in self.state.items()}
        }
        self.index.list.side_effect = lambda namespace: [list(self.state.get(namespace, {}))]

        def delete(*, delete_all, namespace):
            self.assertTrue(delete_all)
            self.state[namespace] = {}

        def upsert(*, vectors, namespace):
            self.state.setdefault(namespace, {}).update({record["id"]: record for record in vectors})

        self.index.delete.side_effect = delete
        self.index.upsert.side_effect = upsert
        for target, replacement in (
            ("ingest.Pinecone", MagicMock(return_value=self.client)),
            ("ingest.OpenAIEmbeddings", MagicMock(return_value=self.embeddings)),
            ("ingest.cargar_y_fragmentar", MagicMock(return_value=self.chunks)),
        ):
            patcher = patch(target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_crea_indice_serverless_si_falta(self):
        self.client.has_index.return_value = False
        self.assertIs(preparar_indice(self.client, self.config), self.index)
        args = self.client.create_index.call_args.kwargs
        self.assertEqual(args["name"], self.config.index_name)
        self.assertEqual((args["dimension"], args["metric"], args["timeout"]), (1536, "cosine", 120))
        self.assertEqual(args["spec"].asdict(), {"serverless": {"cloud": "aws", "region": "us-east-1"}})
        self.client.Index.assert_called_once_with(host="host-ficticio")

    def test_reutiliza_indice_compatible(self):
        self.client.has_index.return_value = True
        preparar_indice(self.client, self.config)
        self.client.create_index.assert_not_called()

    def test_rechaza_indice_incompatible_sin_borrar(self):
        for field, value in (("dimension", 768), ("metric", "euclidean"), ("spec", {})):
            with self.subTest(field=field), patch.dict(self.description, {field: value}):
                with self.assertRaisesRegex(ValueError, "no coincide"):
                    ingerir(self.config)
                self.index.delete.assert_not_called()
                self.index.upsert.assert_not_called()

    def test_ingesta_por_lotes_con_texto_y_namespace_aislado(self):
        with patch("ingest.BATCH_SIZE", 2):
            self.assertEqual(ingerir(self.config), 3)
        self.assertEqual(self.index.upsert.call_count, 2)
        self.assertEqual(set(self.state["entrega-test"]), {"id-0", "id-1", "id-2"})
        self.assertEqual(self.state["otro-namespace"], {"conservar": {}})
        self.index.delete.assert_called_once_with(delete_all=True, namespace="entrega-test")
        for record in self.state["entrega-test"].values():
            self.assertEqual(len(record["values"]), EMBEDDING_DIM)
            metadata = MetadataChunk(**record["metadata"])
            self.assertEqual(record["id"], metadata.chunk_id)
        from ingest import OpenAIEmbeddings, cargar_y_fragmentar
        cargar_y_fragmentar.assert_called_once_with(chunk_size=700, chunk_overlap=80)
        self.assertEqual(OpenAIEmbeddings.call_args.kwargs["model"], "text-embedding-3-small")
        self.assertFalse(OpenAIEmbeddings.call_args.kwargs["check_embedding_ctx_length"])
        self.index.__exit__.assert_called_once()

    def test_reingesta_no_acumula_registros_ni_conserva_antiguos(self):
        ingerir(self.config)
        self.state["entrega-test"]["resto-ajeno"] = {}
        ingerir(self.config)
        self.assertEqual(set(self.state["entrega-test"]), {"id-0", "id-1", "id-2"})
        self.assertEqual(self.index.delete.call_count, 2)

    def test_indice_nuevo_no_intenta_borrar_namespace_inexistente(self):
        del self.state["entrega-test"]
        ingerir(self.config)
        self.index.delete.assert_not_called()
        self.assertEqual(len(self.state["entrega-test"]), 3)

    def test_error_de_embeddings_no_limpia_ni_abre_pinecone(self):
        self.embeddings.embed_documents.side_effect = RuntimeError("fallo simulado")
        with self.assertRaises(RuntimeError):
            ingerir(self.config)
        from ingest import Pinecone
        Pinecone.assert_not_called()
        self.index.delete.assert_not_called()

    def test_rechaza_vectores_invalidos_antes_del_borrado(self):
        cases = ([], [[0.1] * 768] * 3, [[float("nan")] * EMBEDDING_DIM] * 3)
        for vectors in cases:
            with self.subTest(count=len(vectors)):
                self.embeddings.embed_documents.return_value = vectors
                with self.assertRaises(ValueError):
                    ingerir(self.config)
                self.index.delete.assert_not_called()

    def test_valida_metadatos_e_ids_antes_de_generar_embeddings(self):
        for chunks in ([], [self.chunks[0], self.chunks[0]]):
            with self.subTest(count=len(chunks)):
                with self.assertRaises(ValueError):
                    preparar_registros(chunks, self.embeddings)
        self.chunks[0].metadata["text"] = "Texto diferente"
        with self.assertRaises(ValueError):
            preparar_registros(self.chunks, self.embeddings)
        del self.chunks[0].metadata["source"]
        with self.assertRaises(ValueError):
            preparar_registros(self.chunks, self.embeddings)
        self.embeddings.embed_documents.assert_not_called()

    def test_ids_paginados_y_cantidad_no_bastan_por_separado(self):
        self.index.list.side_effect = None
        self.index.list.return_value = [["uno"], ["dos"]]
        self.assertEqual(obtener_ids_namespace(self.index, "entrega-test"), {"uno", "dos"})
        self.state["entrega-test"] = {"uno": {}, "otro": {}}
        self.assertFalse(namespace_coincide(self.index, "entrega-test", {"uno", "otro"}))
        self.assertTrue(namespace_coincide(self.index, "entrega-test", {"uno", "dos"}))

    def test_error_de_subida_no_anuncia_exito(self):
        self.index.upsert.side_effect = RuntimeError("fallo simulado")
        with self.assertLogs("ingest", level="INFO") as captured:
            with self.assertRaises(RuntimeError):
                ingerir(self.config)
        self.assertFalse(any("Ingesta completa" in line for line in captured.output))

    def test_espera_espaciada_y_timeout_acotado(self):
        with patch("ingest.time.monotonic", side_effect=[0, 0, 120]), patch("ingest.time.sleep") as sleep:
            with self.assertRaises(TimeoutError):
                esperar(lambda: False, "operación simulada")
            sleep.assert_called_once_with(2)
        condition = MagicMock(side_effect=[False, True])
        with patch("ingest.time.monotonic", return_value=0), patch("ingest.time.sleep"):
            esperar(condition, "operación simulada")
        self.assertEqual(condition.call_count, 2)

    def test_timeout_de_creacion_borrado_y_visibilidad(self):
        for stage in ("creación", "borrado", "visibilidad"):
            with self.subTest(stage=stage):
                self.description["status"]["ready"] = stage != "creación"
                self.index.delete.side_effect = None if stage == "borrado" else lambda **kwargs: self.state.update({"entrega-test": {}})
                self.index.upsert.side_effect = None
                self.state["entrega-test"] = {"antiguo": {}}
                with patch("ingest.WAIT_TIMEOUT", 0):
                    with self.assertRaises(TimeoutError):
                        ingerir(self.config)

    def test_main_controla_errores_sin_mostrar_excepcion_sensible(self):
        with patch("ingest.load_dotenv"), patch("ingest.logging.basicConfig"), patch(
            "ingest.cargar_configuracion", return_value=self.config
        ):
            with patch("ingest.ingerir", side_effect=RuntimeError("secreto-ficticio")):
                with self.assertLogs("ingest", level="ERROR") as captured:
                    self.assertEqual(main(), 1)
                self.assertNotIn("secreto-ficticio", "".join(captured.output))
            with patch("ingest.ingerir", return_value=3):
                self.assertEqual(main(), 0)


if __name__ == "__main__":
    unittest.main()
