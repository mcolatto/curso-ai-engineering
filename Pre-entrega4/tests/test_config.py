"""Tests locales de configuración, sin claves reales ni servicios cloud."""

import json
import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from config import (
    BASE_DIR, DATA_DIR, EMBEDDING_DIM, EMBEDDING_MODEL,
    GOLDEN_SET_PATH, TOKEN_ENCODING, TOP_K, cargar_configuracion,
)


class ConfiguracionTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            "OPENAI_API_KEY": "clave-openai-ficticia",
            "PINECONE_API_KEY": "clave-pinecone-ficticia",
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_solo_las_claves_son_obligatorias(self):
        config = cargar_configuracion()
        self.assertEqual(config.index_name, "preentrega4-rag-hibrido")
        self.assertEqual(config.namespace, "langgraph-docs")
        self.assertEqual((config.cloud, config.region), ("aws", "us-east-1"))
        self.assertEqual((config.chunk_size, config.chunk_overlap), (600, 100))
        self.assertEqual(config.weights, [0.5, 0.5])

    def test_rechaza_cada_clave_ausente_vacia_o_con_espacios(self):
        for name in ("OPENAI_API_KEY", "PINECONE_API_KEY"):
            for value in (None, "", " \n\t"):
                with self.subTest(name=name, value=value), patch.dict(os.environ):
                    if value is None:
                        del os.environ[name]
                    else:
                        os.environ[name] = value
                    with self.assertRaises(ValidationError):
                        cargar_configuracion()

    def test_parametros_personalizados_y_pesos_en_orden(self):
        with patch.dict(os.environ, {
            "INDEX_NAME": " indice-ficticio ", "NAMESPACE": " namespace-ficticio ",
            "PINECONE_REGION": "otra-region", "CHUNK_SIZE": "700",
            "CHUNK_OVERLAP": "80", "BM25_WEIGHT": "0.7", "VECTOR_WEIGHT": "0.3",
        }):
            config = cargar_configuracion()
        self.assertEqual(config.index_name, "indice-ficticio")
        self.assertEqual(config.namespace, "namespace-ficticio")
        self.assertEqual(config.region, "otra-region")
        self.assertEqual((config.chunk_size, config.chunk_overlap), (700, 80))
        self.assertEqual(config.weights, [0.7, 0.3])

    def test_rechaza_identificadores_vacios(self):
        for name in ("INDEX_NAME", "NAMESPACE", "PINECONE_CLOUD", "PINECONE_REGION"):
            with self.subTest(name=name), patch.dict(os.environ, {name: " \t"}):
                with self.assertRaises(ValidationError):
                    cargar_configuracion()

    def test_rechaza_limites_y_pesos_invalidos(self):
        cases = [
            {"CHUNK_SIZE": "0"}, {"CHUNK_SIZE": "-1"}, {"CHUNK_SIZE": "1.5"},
            {"CHUNK_OVERLAP": "-1"}, {"CHUNK_OVERLAP": "600"},
            {"CHUNK_SIZE": "80", "CHUNK_OVERLAP": "100"},
            {"BM25_WEIGHT": "-0.5"}, {"VECTOR_WEIGHT": "1.5"},
            {"BM25_WEIGHT": "nan"}, {"VECTOR_WEIGHT": "inf"},
            {"BM25_WEIGHT": "0.8"}, {"VECTOR_WEIGHT": "0.1"},
            {"BM25_WEIGHT": "0", "VECTOR_WEIGHT": "0"},
        ]
        for values in cases:
            with self.subTest(values=values), patch.dict(os.environ, values):
                with self.assertRaises(ValidationError):
                    cargar_configuracion()

    def test_acepta_solapamiento_cero_y_pesos_extremos(self):
        for bm25, vector in (("1", "0"), ("0", "1")):
            with self.subTest(bm25=bm25), patch.dict(os.environ, {
                "CHUNK_OVERLAP": "0", "BM25_WEIGHT": bm25, "VECTOR_WEIGHT": vector,
            }):
                config = cargar_configuracion()
                self.assertEqual(config.chunk_overlap, 0)
                self.assertEqual(config.weights, [float(bm25), float(vector)])

    def test_secretos_ocultos_en_representacion_json_y_error(self):
        config = cargar_configuracion()
        with patch.dict(os.environ, {"CHUNK_SIZE": "valor-sensible-ficticio"}):
            with self.assertRaises(ValidationError) as raised:
                cargar_configuracion()
        outputs = (repr(config), config.model_dump_json(), str(raised.exception))
        for output in outputs:
            self.assertNotIn("clave-openai-ficticia", output)
            self.assertNotIn("clave-pinecone-ficticia", output)
            self.assertNotIn("valor-sensible-ficticio", output)

    def test_rutas_y_constantes_compartidas(self):
        root = Path(__file__).resolve().parent.parent
        self.assertEqual(BASE_DIR, root)
        self.assertEqual(DATA_DIR, root / "data")
        self.assertEqual(GOLDEN_SET_PATH, root / "golden_dataset.json")
        self.assertNotEqual(GOLDEN_SET_PATH.parent, DATA_DIR)
        self.assertEqual((EMBEDDING_MODEL, EMBEDDING_DIM), ("text-embedding-3-small", 1536))
        self.assertEqual(TOKEN_ENCODING, "cl100k_base")
        self.assertEqual(TOP_K, 5)

    def test_datos_y_golden_conservan_nombres_y_pregunta_de_memoria(self):
        names = {p.name for p in DATA_DIR.glob("*.md")}
        self.assertEqual(names, {
            "checkpointers.md", "interrupts.md", "memory.md", "persistence.md",
            "stores.md", "subgraphs.md", "time-travel.md",
        })
        cases = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 5)
        self.assertTrue(all(case["documento_id_esperado"] in names for case in cases))
        self.assertEqual(cases[3], {
            "pregunta": "How can an agent retain context during a conversation and remember information across different sessions?",
            "documento_id_esperado": "memory.md",
        })

    def test_importar_config_no_necesita_claves(self):
        with patch.dict(os.environ, {}, clear=True):
            result = runpy.run_module("config", run_name="config_import_test")
        self.assertIn("cargar_configuracion", result)


if __name__ == "__main__":
    unittest.main()
