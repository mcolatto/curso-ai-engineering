"""Comprobaciones de configuración sin clientes ni llamadas a APIs."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from config import (
    BASE_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, DATA_DIR,
    PERSIST_DIR, TOP_K, cargar_configuracion,
)


class ConfiguracionTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"OPENAI_API_KEY": "clave-ficticia"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_solo_la_clave_es_obligatoria(self):
        config = cargar_configuracion()
        self.assertEqual(config.model, "gpt-6.1-sol")
        self.assertEqual(config.embedding_model, "text-embedding-3-small")
        self.assertEqual(config.reasoning_effort, "low")
        self.assertEqual(config.max_tokens, 1024)
        self.assertEqual(config.timeout_seconds, 30)

    def test_rechaza_clave_ausente_vacia_o_con_espacios(self):
        for value in (None, "", " \n\t"):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["OPENAI_API_KEY"] = value
                with self.assertRaises(ValidationError) as raised:
                    cargar_configuracion()
                self.assertIn("OPENAI_API_KEY", str(raised.exception))

    def test_lee_parametros_personalizados(self):
        with patch.dict(os.environ, {
            "OPENAI_MODEL": " modelo-ficticio ",
            "OPENAI_EMBEDDING_MODEL": " embedding-ficticio ",
            "OPENAI_REASONING_EFFORT": "high", "LLM_MAX_TOKENS": "512",
            "LLM_TIMEOUT_SECONDS": "15",
        }):
            config = cargar_configuracion()
        self.assertEqual(config.model, "modelo-ficticio")
        self.assertEqual(config.embedding_model, "embedding-ficticio")
        self.assertEqual(config.reasoning_effort, "high")
        self.assertEqual(config.max_tokens, 512)
        self.assertEqual(config.timeout_seconds, 15)

    def test_modelos_vacios_usan_predeterminados(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": " ", "OPENAI_EMBEDDING_MODEL": ""}):
            config = cargar_configuracion()
        self.assertEqual(config.model, "gpt-6.1-sol")
        self.assertEqual(config.embedding_model, "text-embedding-3-small")

    def test_esfuerzo_opcional_y_sin_predeterminado_para_otro_modelo(self):
        for environment in ({"OPENAI_REASONING_EFFORT": ""}, {"OPENAI_MODEL": "otro-modelo"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment):
                self.assertIsNone(cargar_configuracion().reasoning_effort)

    def test_rechaza_limites_y_esfuerzo_invalidos_sin_exponer_valores(self):
        cases = [
            ("LLM_MAX_TOKENS", "0"), ("LLM_MAX_TOKENS", "-1"),
            ("LLM_MAX_TOKENS", "1.5"), ("LLM_TIMEOUT_SECONDS", "0"),
            ("LLM_TIMEOUT_SECONDS", "nan"), ("LLM_TIMEOUT_SECONDS", "inf"),
            ("LLM_TIMEOUT_SECONDS", "-1"), ("OPENAI_REASONING_EFFORT", "loow"),
            ("LLM_MAX_TOKENS", "valor-sensible-ficticio"),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                with self.assertRaises(ValidationError) as raised:
                    cargar_configuracion()
                self.assertNotIn("clave-ficticia", str(raised.exception))
                self.assertNotIn("valor-sensible-ficticio", str(raised.exception))

    def test_clave_oculta_en_representacion_y_json(self):
        config = cargar_configuracion()
        self.assertNotIn("clave-ficticia", repr(config))
        self.assertNotIn("clave-ficticia", config.model_dump_json())
        self.assertEqual(config.api_key.get_secret_value(), "clave-ficticia")

    def test_rutas_y_parametros_compartidos(self):
        expected = Path(__file__).resolve().parent.parent
        self.assertEqual(BASE_DIR, expected)
        self.assertEqual(DATA_DIR, expected / "data")
        self.assertEqual(PERSIST_DIR, expected / "vectorstore")
        self.assertEqual((CHUNK_SIZE, CHUNK_OVERLAP, TOP_K), (500, 50, 4))
        self.assertTrue(COLLECTION_NAME)


if __name__ == "__main__":
    unittest.main()
