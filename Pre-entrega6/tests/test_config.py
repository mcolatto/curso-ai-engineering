"""Tests de configuración y archivos locales, sin claves ni APIs reales."""

import os
import runpy
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from config import BASE_DIR, CHECKPOINT_PATH, TRACE_PATH, cargar_configuracion


class ConfiguracionTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"OPENAI_API_KEY": "clave-ficticia"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_clave_obligatoria_y_valores_predeterminados(self):
        config = cargar_configuracion()
        self.assertEqual(config.model, "gpt-6.1-sol")
        self.assertEqual(config.timeout_seconds, 30)
        self.assertEqual(config.max_tokens, 4096)
        self.assertIsNone(config.reasoning_effort)
        self.assertEqual(config.recursion_limit, 10)

    def test_rechaza_clave_ausente_vacia_o_solo_espacios(self):
        for value in (None, "", " \t\n"):
            with self.subTest(value=value), patch.dict(os.environ):
                if value is None:
                    del os.environ["OPENAI_API_KEY"]
                else:
                    os.environ["OPENAI_API_KEY"] = value
                with self.assertRaises(ValidationError):
                    cargar_configuracion()

    def test_configuracion_personalizada_y_espacios_exteriores(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": " clave-ficticia ", "OPENAI_MODEL": " modelo-ficticio ",
            "OPENAI_TIMEOUT_SECONDS": "45.5", "OPENAI_MAX_TOKENS": "2048",
            "OPENAI_REASONING_EFFORT": " medium ", "RECURSION_LIMIT": "12",
        }):
            config = cargar_configuracion()
        self.assertEqual(config.openai_api_key.get_secret_value(), "clave-ficticia")
        self.assertEqual(config.model, "modelo-ficticio")
        self.assertEqual(config.timeout_seconds, 45.5)
        self.assertEqual(config.max_tokens, 2048)
        self.assertEqual(config.reasoning_effort, "medium")
        self.assertEqual(config.recursion_limit, 12)

    def test_razonamiento_opcional_y_valores_permitidos(self):
        for value in ("low", "medium", "high", "xhigh", "max", "", "  "):
            with self.subTest(value=value), patch.dict(os.environ, {"OPENAI_REASONING_EFFORT": value}):
                self.assertEqual(cargar_configuracion().reasoning_effort, value.strip() or None)

    def test_rechaza_modelo_vacio_y_limites_invalidos(self):
        cases = {
            "OPENAI_MODEL": ["", " \t"],
            "OPENAI_TIMEOUT_SECONDS": ["0", "-1", "nan", "inf", "texto"],
            "OPENAI_MAX_TOKENS": ["0", "-1", "1.5", "texto"],
            "RECURSION_LIMIT": ["0", "-1", "1.5", "texto"],
            "OPENAI_REASONING_EFFORT": ["desconocido"],
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                    with self.assertRaises(ValidationError):
                        cargar_configuracion()

    def test_secretos_ocultos_en_representacion_json_y_errores(self):
        config = cargar_configuracion()
        with patch.dict(os.environ, {"OPENAI_MAX_TOKENS": "dato-sensible-ficticio"}):
            with self.assertRaises(ValidationError) as raised:
                cargar_configuracion()
        for output in (repr(config), config.model_dump_json(), str(raised.exception)):
            self.assertNotIn("clave-ficticia", output)
            self.assertNotIn("dato-sensible-ficticio", output)

    def test_rutas_dentro_de_la_carpeta(self):
        self.assertEqual(BASE_DIR, Path(__file__).resolve().parent.parent)
        self.assertEqual(CHECKPOINT_PATH, BASE_DIR / "checkpoints.sqlite")
        self.assertEqual(TRACE_PATH, BASE_DIR / "traza_ejecucion.json")

    def test_importar_no_requiere_clave_ni_carga_dotenv(self):
        with patch.dict(os.environ, {}, clear=True):
            module = runpy.run_module("config", run_name="config_import_test")
            self.assertNotIn("OPENAI_API_KEY", os.environ)
        self.assertIn("cargar_configuracion", module)

    def test_git_ignora_secretos_y_sqlite_pero_no_la_traza(self):
        ignored = [".env", ".venv/bin/python", "__pycache__/config.pyc", "modulo.pyc",
                   "checkpoints.sqlite", "checkpoints.sqlite-wal", "checkpoints.sqlite-shm",
                   "checkpoints.sqlite-journal"]
        result = subprocess.run(
            ["git", "check-ignore", "--stdin"], input="\n".join(ignored) + "\n",
            text=True, capture_output=True, cwd=BASE_DIR, check=True,
        )
        self.assertEqual(set(result.stdout.splitlines()), set(ignored))
        result = subprocess.run(
            ["git", "check-ignore", ".env.example", "traza_ejecucion.json"],
            text=True, capture_output=True, cwd=BASE_DIR,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
