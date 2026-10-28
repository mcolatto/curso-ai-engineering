"""Pruebas de recuperación asíncrona sin embeddings pagos ni LLM."""

import asyncio
import json
import os
import runpy
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from chain import crear_cadena, crear_prompt, get_rag_response, recuperar_documentos
from config import COLLECTION_NAME, TOP_K, Configuracion
from ingest import preparar_indice
from schemas import RAGResponse


class EmbeddingsSimulados(Embeddings):
    """Vectores locales deterministas; no prueban relevancia semántica."""

    def __init__(self):
        self.document_calls = 0
        self.query_calls = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self.vector(text)

    def vector(self, text: str) -> list[float]:
        return [float(len(text)), float(sum(map(ord, text)) % 101), 1.0]


class RecuperacionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.data_dir = Path(temp_dir.name) / "data"
        self.data_dir.mkdir()
        self.persist_dir = Path(temp_dir.name) / "vectorstore"
        self.config = Configuracion(api_key="clave-ficticia")
        self.embeddings = EmbeddingsSimulados()
        patcher = patch("ingest.crear_embeddings", return_value=self.embeddings)
        self.embedding_factory = patcher.start()
        self.addCleanup(patcher.stop)

    def preparar_datos(self):
        for index in range(6):
            (self.data_dir / f"documento-{index}.txt").write_text(
                f"Contenido de prueba del documento número {index}.", encoding="utf-8",
            )
        return preparar_indice(self.config, self.data_dir, self.persist_dir)

    async def test_consulta_chroma_y_conserva_metadatos_sin_reindexar(self):
        self.preparar_datos()
        docs = await recuperar_documentos("consulta ficticia", self.config, self.persist_dir)
        self.assertEqual(len(docs), TOP_K)
        self.assertEqual(self.embeddings.query_calls, ["consulta ficticia"])
        self.assertEqual(self.embeddings.document_calls, 1)
        for doc in docs:
            self.assertIsInstance(doc, Document)
            source = doc.metadata["source"]
            self.assertEqual(doc.page_content, (self.data_dir / source).read_text(encoding="utf-8"))

    async def test_configura_similitud_y_espera_ainvoke(self):
        expected = [Document(page_content="Texto", metadata={"source": "rag.txt"})]
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(return_value=expected)
        with patch("chain.abrir_indice") as open_index:
            open_index.return_value.as_retriever.return_value = retriever
            result = await recuperar_documentos("Pregunta", self.config, self.persist_dir)
            open_index.assert_called_once_with(self.config, self.persist_dir)
            open_index.return_value.as_retriever.assert_called_once_with(
                search_type="similarity", search_kwargs={"k": TOP_K},
            )
            retriever.ainvoke.assert_awaited_once_with("Pregunta")
        self.assertIs(result, expected)

    async def test_pregunta_vacia_no_abre_indice_ni_lee_configuracion(self):
        with patch("chain.abrir_indice") as open_index, patch("chain.cargar_configuracion") as load:
            for query in ("", "   ", "\n\t", None, 123):
                with self.subTest(query=query):
                    with self.assertRaises(ValueError):
                        await recuperar_documentos(query)
            open_index.assert_not_called()
            load.assert_not_called()

    async def test_indice_ausente_no_crea_carpeta_ni_genera_embeddings(self):
        with self.assertRaisesRegex(ValueError, "ingest.py"):
            await recuperar_documentos("Pregunta", self.config, self.persist_dir)
        self.assertFalse(self.persist_dir.exists())
        self.embedding_factory.assert_not_called()

    async def test_coleccion_ausente_o_vacia_no_inicia_ingesta(self):
        client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False),
        )
        with self.assertRaisesRegex(ValueError, "No existe la colección"):
            await recuperar_documentos("Pregunta", self.config, self.persist_dir)
        client.create_collection(
            COLLECTION_NAME, embedding_function=None,
            metadata={"embedding_model": self.config.embedding_model},
        )
        with self.assertRaisesRegex(ValueError, "vacía"):
            await recuperar_documentos("Pregunta", self.config, self.persist_dir)
        self.embedding_factory.assert_not_called()
        self.assertEqual(self.embeddings.document_calls, 0)

    async def test_modelo_incompatible_no_genera_embedding_de_consulta(self):
        self.preparar_datos()
        self.embedding_factory.reset_mock()
        incompatible = Configuracion(api_key="clave-ficticia", embedding_model="otro-modelo")
        with self.assertRaisesRegex(ValueError, "modelo de embeddings.*--rebuild"):
            await recuperar_documentos("Pregunta", incompatible, self.persist_dir)
        self.embedding_factory.assert_not_called()
        self.assertEqual(self.embeddings.query_calls, [])

    async def test_error_de_recuperacion_se_propaga_sin_reintentos(self):
        with patch("chain.abrir_indice") as open_index:
            retrieve = AsyncMock(side_effect=RuntimeError("fallo simulado"))
            open_index.return_value.as_retriever.return_value.ainvoke = retrieve
            with self.assertRaisesRegex(RuntimeError, "fallo simulado"):
                await recuperar_documentos("Pregunta", self.config, self.persist_dir)
            retrieve.assert_awaited_once()

    async def test_consulta_pendiente_permite_avanzar_al_event_loop(self):
        self.preparar_datos()
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def delayed_embedding(text):
            loop.call_soon_threadsafe(started.set)
            if not release.wait(timeout=3):
                raise TimeoutError("No se liberó el embedding simulado")
            return self.embeddings.vector(text)

        with patch.object(self.embeddings, "embed_query", side_effect=delayed_embedding):
            task = asyncio.create_task(recuperar_documentos("Pregunta", self.config, self.persist_dir))
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                self.assertFalse(task.done())
                release.set()
                docs = await asyncio.wait_for(task, timeout=2)
                self.assertEqual(len(docs), TOP_K)
            finally:
                release.set()
                if not task.done():
                    await asyncio.wait_for(task, timeout=3)

    async def test_apertura_del_indice_se_ejecuta_fuera_del_hilo_del_event_loop(self):
        loop_thread = threading.get_ident()
        opening_threads = []
        store = MagicMock()
        store.as_retriever.return_value.ainvoke = AsyncMock(return_value=[])

        def open_index(*args):
            opening_threads.append(threading.get_ident())
            return store

        with patch("chain.abrir_indice", side_effect=open_index):
            self.assertEqual(await recuperar_documentos("Pregunta", self.config, self.persist_dir), [])
        self.assertEqual(len(opening_threads), 1)
        self.assertNotEqual(opening_threads[0], loop_thread)


class GeneracionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"OPENAI_API_KEY": "clave-ficticia"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        retrieve_patch = patch("chain.recuperar_documentos", new_callable=AsyncMock)
        self.retrieve = retrieve_patch.start()
        self.addCleanup(retrieve_patch.stop)
        self.docs = [
            Document(page_content="Una base vectorial almacena vectores.", metadata={"source": "vectordb.txt"}),
            Document(page_content="RAG incorpora contexto recuperado.", metadata={"source": "rag.txt"}),
            Document(page_content="Otro fragmento sobre RAG.", metadata={"source": "rag.txt"}),
        ]
        self.retrieve.return_value = self.docs

    def response(self, content):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def test_cadena_y_parser_reales_ensamblan_fuentes_del_contexto(self):
        # El campo adicional de fuentes del LLM no debe determinar las fuentes finales.
        content = json.dumps({"respuesta": "RAG incorpora contexto recuperado.", "fuentes": ["inventada.txt"]})
        with patch.object(ChatOpenAI, "_agenerate", new_callable=AsyncMock) as generate:
            generate.return_value = self.response(content)
            with self.assertLogs("chain", level="INFO") as logs:
                result = await get_rag_response("¿Qué incorpora RAG?")
        self.assertIsInstance(result, RAGResponse)
        self.assertEqual(result.respuesta, "RAG incorpora contexto recuperado.")
        self.assertEqual(result.fuentes, ["rag.txt", "vectordb.txt"])
        self.assertEqual(result.fragmentos_recuperados, 3)
        generate.assert_awaited_once()
        self.retrieve.assert_awaited_once()
        self.assertEqual(self.retrieve.call_args.args[0], "¿Qué incorpora RAG?")
        messages = generate.call_args.args[0]
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        for doc in self.docs:
            self.assertIn(doc.page_content, messages[1].content)
            self.assertIn(doc.metadata["source"], messages[1].content)
        self.assertIn("PREGUNTA:\n¿Qué incorpora RAG?", messages[1].content)
        self.assertNotIn("response_format", generate.call_args.kwargs)
        logged = "\n".join(logs.output)
        self.assertNotIn("clave-ficticia", logged)
        self.assertNotIn(self.docs[0].page_content, logged)

    async def test_no_lo_se_conserva_fuentes_consultadas_y_contador(self):
        with patch.object(ChatOpenAI, "_agenerate", new_callable=AsyncMock) as generate:
            generate.return_value = self.response('{"respuesta": "No lo sé"}')
            result = await get_rag_response("Pregunta sin respuesta en el corpus")
        self.assertEqual(result.respuesta, "No lo sé")
        self.assertEqual(result.fuentes, ["rag.txt", "vectordb.txt"])
        self.assertEqual(result.fragmentos_recuperados, 3)

    async def test_sin_documentos_devuelve_no_lo_se_sin_crear_llm(self):
        self.retrieve.return_value = []
        with patch("chain.ChatOpenAI") as constructor:
            result = await get_rag_response("Pregunta")
            constructor.assert_not_called()
        self.assertEqual(result.model_dump(), {
            "respuesta": "No lo sé", "fuentes": [], "fragmentos_recuperados": 0,
        })

    async def test_pregunta_vacia_no_lee_configuracion_ni_consulta(self):
        with patch("chain.cargar_configuracion") as load, patch("chain.ChatOpenAI") as constructor:
            for query in ("", "   ", "\n\t", None, 123):
                with self.subTest(query=query):
                    with self.assertRaises(ValueError):
                        await get_rag_response(query)
            load.assert_not_called()
            constructor.assert_not_called()
        self.retrieve.assert_not_called()

    async def test_json_invalido_o_respuesta_vacia_falla_sin_reintentar(self):
        for content in ("No es un JSON", "{}", '{"respuesta": "   "}'):
            with self.subTest(content=content), patch.object(
                ChatOpenAI, "_agenerate", new_callable=AsyncMock,
            ) as generate:
                generate.return_value = self.response(content)
                with self.assertRaises(OutputParserException):
                    await get_rag_response("Pregunta")
                generate.assert_awaited_once()

    async def test_error_de_recuperacion_impide_crear_llm(self):
        self.retrieve.side_effect = ValueError("Falta la ingesta")
        with patch("chain.ChatOpenAI") as constructor:
            with self.assertRaisesRegex(ValueError, "Falta la ingesta"):
                await get_rag_response("Pregunta")
            constructor.assert_not_called()

    async def test_llamada_llm_pendiente_permite_avanzar_al_event_loop(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed_response(*args, **kwargs):
            started.set()
            await release.wait()
            return self.response('{"respuesta": "Respuesta simulada"}')

        with patch.object(ChatOpenAI, "_agenerate", new_callable=AsyncMock) as generate:
            generate.side_effect = delayed_response
            task = asyncio.create_task(get_rag_response("Pregunta"))
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                self.assertFalse(task.done())
                release.set()
                self.assertIsInstance(await asyncio.wait_for(task, timeout=2), RAGResponse)
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def test_prompt_sustituye_variables_y_pide_no_lo_se_sin_fuentes_generadas(self):
        prompt = crear_prompt()
        self.assertEqual(set(prompt.input_variables), {"contexto", "pregunta"})
        messages = prompt.invoke({"contexto": "Texto con {llaves}", "pregunta": "¿Qué dice?"}).to_messages()
        self.assertIn("exactamente: No lo sé", messages[0].content)
        self.assertIn("exclusivamente información presente", messages[0].content)
        self.assertIn("No generes fuentes", messages[0].content)
        self.assertIn('"respuesta"', messages[0].content)
        self.assertNotIn("{formato}", messages[0].content)
        self.assertEqual(messages[1].content, "CONTEXTO:\nTexto con {llaves}\n\nPREGUNTA:\n¿Qué dice?")

    def test_configura_llm_sin_temperatura_ni_reintentos(self):
        config = Configuracion(api_key="clave-ficticia")
        with patch("chain.ChatOpenAI", wraps=ChatOpenAI) as constructor:
            crear_cadena(config)
            constructor.assert_called_once_with(
                api_key="clave-ficticia", model="gpt-6.1-sol", reasoning_effort="low",
                temperature=None, max_completion_tokens=1024, timeout=30,
                max_retries=0, use_responses_api=False,
            )

    def test_importar_chain_no_crea_clientes(self):
        with patch("langchain_openai.ChatOpenAI") as llm, patch("langchain_openai.OpenAIEmbeddings") as embeddings:
            runpy.run_module("chain", run_name="chain_import_test")
            llm.assert_not_called()
            embeddings.assert_not_called()


if __name__ == "__main__":
    unittest.main()
