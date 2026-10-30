"""Fusión real de rankings con fronteras de servicios simuladas."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from config import Configuracion
from rag_system import RAGSystem


class RAGSystemTests(unittest.TestCase):
    def setUp(self):
        self.config = Configuracion(
            openai_api_key="openai-ficticia", pinecone_api_key="pinecone-ficticia"
        )
        self.chunks = [Document(
            page_content=f"Fragmento {number}",
            metadata={"source": "memory.md", "categoria": "memory", "chunk_id": f"id-{number}"},
        ) for number in range(8)]
        self.bm25_call = MagicMock(return_value=self.chunks[:5])
        # El ID 4 está en ambas ramas; los otros tres IDs amplían el ranking combinado.
        self.vector_call = MagicMock(return_value=self.chunks[4:])
        self.client = MagicMock()
        self.description = {
            "dimension": 1536, "metric": "cosine", "host": "host-ficticio",
            "status": {"ready": True},
            "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
        }
        self.client.describe_index.return_value.to_dict.return_value = self.description
        self.index = self.client.Index.return_value
        self.index.describe_index_stats.return_value = {
            "namespaces": {self.config.namespace: {"vector_count": 8}}
        }
        self.index.list.return_value = [[f"id-{n}" for n in range(8)]]
        self.store = MagicMock()
        self.store.as_retriever.return_value = RunnableLambda(self.vector_call)
        replacements = {
            "cargar_y_fragmentar": MagicMock(return_value=self.chunks),
            "Pinecone": MagicMock(return_value=self.client),
            "OpenAIEmbeddings": MagicMock(),
            "PineconeVectorStore": MagicMock(return_value=self.store),
        }
        self.patches = {}
        for name, replacement in replacements.items():
            patcher = patch(f"rag_system.{name}", replacement)
            self.patches[name] = patcher.start()
            self.addCleanup(patcher.stop)
        self.real_bm25_factory = BM25Retriever.from_documents
        patcher = patch("rag_system.BM25Retriever.from_documents", return_value=RunnableLambda(self.bm25_call))
        self.bm25_factory = patcher.start()
        self.addCleanup(patcher.stop)

    def test_fusion_por_id_y_top_cinco_despues_de_combinar(self):
        rag = RAGSystem(self.config)
        results = rag.obtener_top_k("memory")
        self.assertIsInstance(rag._retriever, EnsembleRetriever)
        combined = rag._retriever.invoke("memory")
        self.assertEqual(len(combined), 8)
        self.assertEqual(results, combined[:5])
        self.assertEqual(results[0].metadata["chunk_id"], "id-4")
        self.assertEqual(len({doc.metadata["chunk_id"] for doc in results}), 5)
        self.assertTrue(all(doc.metadata["source"] == "memory.md" for doc in results))
        self.assertTrue(all(doc.page_content and doc.metadata["categoria"] for doc in results))
        self.assertTrue(all(doc is self.chunks[int(doc.metadata["chunk_id"].split("-")[1])] for doc in results))

    def test_configura_ambas_ramas_y_embeddings_compartidos(self):
        config = self.config.model_copy(update={"chunk_size": 700, "chunk_overlap": 80, "bm25_weight": 0.7, "vector_weight": 0.3})
        rag = RAGSystem(config)
        rag.obtener_top_k("consulta")
        self.patches["cargar_y_fragmentar"].assert_called_once_with(chunk_size=700, chunk_overlap=80)
        self.bm25_factory.assert_called_once_with(self.chunks, k=5)
        self.store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
        self.patches["PineconeVectorStore"].assert_called_once_with(
            index=self.index, embedding=self.patches["OpenAIEmbeddings"].return_value,
            namespace=config.namespace, text_key="text",
        )
        args = self.patches["OpenAIEmbeddings"].call_args.kwargs
        self.assertEqual((args["model"], args["dimensions"]), ("text-embedding-3-small", 1536))
        self.assertEqual(rag._retriever.weights, [0.7, 0.3])
        self.assertEqual(rag._retriever.id_key, "chunk_id")
        self.bm25_call.assert_called_once_with("consulta")
        self.vector_call.assert_called_once_with("consulta")

    def test_consulta_vacia_no_abre_clientes_ni_consulta_ramas(self):
        rag = RAGSystem(self.config)
        for query in ("", " \t\n", None):
            with self.subTest(query=query), self.assertRaises(ValueError):
                rag.obtener_top_k(query)
        self.patches["Pinecone"].assert_not_called()
        self.patches["OpenAIEmbeddings"].assert_not_called()
        self.bm25_call.assert_not_called()
        self.vector_call.assert_not_called()

    def test_bm25_real_se_integra_con_ensemble_sin_red(self):
        self.bm25_factory.side_effect = self.real_bm25_factory
        rag = RAGSystem(self.config)
        results = rag.obtener_top_k("Fragmento 2")
        bm25 = rag._retriever.retrievers[0]
        self.assertIsInstance(bm25, BM25Retriever)
        self.assertEqual(bm25.k, 5)
        self.assertEqual(bm25.docs, self.chunks)
        self.assertEqual(len(results), 5)
        self.vector_call.assert_called_once_with("Fragmento 2")

    def test_ids_distintos_con_texto_identico_se_conservan(self):
        self.chunks[1].page_content = self.chunks[0].page_content
        results = RAGSystem(self.config).obtener_top_k("consulta")
        ids = {doc.metadata["chunk_id"] for doc in results}
        self.assertTrue({"id-0", "id-1"}.issubset(ids))

    def test_resultados_escasos_o_vacios(self):
        rag = RAGSystem(self.config)
        for documents in ([], self.chunks[:2]):
            with self.subTest(count=len(documents)):
                self.bm25_call.return_value = documents
                self.vector_call.return_value = []
                self.assertEqual(rag.obtener_top_k("consulta"), documents)

    def test_cantidad_o_ids_incorrectos_solicitan_reingesta(self):
        for count, ids in ((7, [f"id-{n}" for n in range(8)]), (8, ["otro"] * 8)):
            with self.subTest(count=count):
                self.index.describe_index_stats.return_value["namespaces"][self.config.namespace]["vector_count"] = count
                self.index.list.return_value = [ids]
                with self.assertRaisesRegex(ValueError, "ingest.py nuevamente"):
                    RAGSystem(self.config).obtener_top_k("consulta")
        self.vector_call.assert_not_called()
        self.bm25_call.assert_not_called()
        self.index.delete.assert_not_called()
        self.index.upsert.assert_not_called()

    def test_revalida_namespace_antes_de_cada_consulta(self):
        rag = RAGSystem(self.config)
        rag.obtener_top_k("primera")
        self.index.list.return_value = [["cambio-externo"]]
        with self.assertRaisesRegex(ValueError, "ingest.py nuevamente"):
            rag.obtener_top_k("segunda")
        self.vector_call.assert_called_once_with("primera")

    def test_indice_ausente_o_incompatible_no_se_crea(self):
        self.client.has_index.return_value = False
        with self.assertRaisesRegex(ValueError, "No existe"):
            RAGSystem(self.config).obtener_top_k("consulta")
        self.client.has_index.return_value = True
        for key, value in (("dimension", 768), ("metric", "euclidean"), ("status", {"ready": False}), ("spec", {})):
            with self.subTest(key=key), patch.dict(self.description, {key: value}):
                with self.assertRaises(ValueError):
                    RAGSystem(self.config).obtener_top_k("consulta")
        self.client.create_index.assert_not_called()

    def test_error_operativo_se_propaga_sin_fingir_resultado_vacio(self):
        self.vector_call.side_effect = RuntimeError("fallo simulado")
        with self.assertRaises(RuntimeError):
            RAGSystem(self.config).obtener_top_k("consulta")

    def test_cerrar_libera_indice_y_permite_reabrir(self):
        rag = RAGSystem(self.config)
        rag.cerrar()
        self.index.close.assert_not_called()
        rag.obtener_top_k("consulta")
        rag.cerrar()
        self.index.close.assert_called_once()
        self.assertIsNone(rag._retriever)
        rag.obtener_top_k("otra consulta")
        self.assertEqual(self.client.Index.call_count, 2)


if __name__ == "__main__":
    unittest.main()
