"""Construcción de OpenAI simulada, sin clientes ni solicitudes reales."""

import runpy
import unittest
from unittest.mock import patch

from config import Configuracion
from llm import crear_llm


class LLMTests(unittest.TestCase):
    def test_configura_modelo_sin_vincular_herramientas_globales(self):
        config = Configuracion(openai_api_key="clave-ficticia")
        with patch("llm.ChatOpenAI") as factory:
            result = crear_llm(config)
        factory.assert_called_once_with(api_key=config.openai_api_key, model=config.model,
                                        timeout=30, max_completion_tokens=4096, max_retries=0)
        self.assertIs(result, factory.return_value)
        factory.return_value.bind_tools.assert_not_called()

    def test_parametros_personalizados_y_razonamiento_opcional(self):
        config = Configuracion(openai_api_key="clave-ficticia", model="modelo-ficticio",
                               timeout_seconds=45, max_tokens=2048, reasoning_effort="low")
        with patch("llm.ChatOpenAI") as factory:
            crear_llm(config)
        factory.assert_called_once_with(api_key=config.openai_api_key, model="modelo-ficticio",
                                        timeout=45, max_completion_tokens=2048, max_retries=0,
                                        reasoning_effort="low")

    def test_importar_no_instancia_cliente(self):
        with patch("langchain_openai.ChatOpenAI") as factory:
            runpy.run_module("llm", run_name="llm_import_test")
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
