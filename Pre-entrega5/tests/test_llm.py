"""Configuración del LLM verificada con mocks, sin construir clientes reales."""

import runpy
import unittest
from unittest.mock import patch

from config import Configuracion
from llm import crear_llm
from tools import herramientas


class LLMTests(unittest.TestCase):
    def test_parametros_predeterminados_y_vinculacion_de_herramientas(self):
        config = Configuracion(openai_api_key="clave-ficticia")
        with patch("llm.ChatOpenAI") as factory:
            result = crear_llm(config)
        factory.assert_called_once_with(
            api_key=config.openai_api_key,
            model="gpt-6.1-sol",
            timeout=30,
            max_completion_tokens=1024,
            max_retries=0,
            reasoning_effort="low",
        )
        factory.return_value.bind_tools.assert_called_once_with(herramientas)
        self.assertIs(result, factory.return_value.bind_tools.return_value)

    def test_omite_razonamiento_si_no_esta_configurado(self):
        config = Configuracion(openai_api_key="clave-ficticia", reasoning_effort=None)
        with patch("llm.ChatOpenAI") as factory:
            crear_llm(config)
        self.assertNotIn("reasoning_effort", factory.call_args.kwargs)
        self.assertNotIn("temperature", factory.call_args.kwargs)
        self.assertNotIn("tool_choice", factory.return_value.bind_tools.call_args.kwargs)

    def test_parametros_personalizados(self):
        config = Configuracion(
            openai_api_key="clave-ficticia", model="modelo-ficticio",
            timeout_seconds=45, max_tokens=2048, reasoning_effort="medium",
        )
        with patch("llm.ChatOpenAI") as factory:
            crear_llm(config)
        args = factory.call_args.kwargs
        self.assertEqual(args["model"], "modelo-ficticio")
        self.assertEqual(args["timeout"], 45)
        self.assertEqual(args["max_completion_tokens"], 2048)
        self.assertEqual(args["reasoning_effort"], "medium")
        self.assertEqual(args["max_retries"], 0)

    def test_importar_no_crea_el_modelo(self):
        with patch("langchain_openai.ChatOpenAI") as factory:
            module = runpy.run_module("llm", run_name="llm_import_test")
        factory.assert_not_called()
        self.assertIn("crear_llm", module)


if __name__ == "__main__":
    unittest.main()
