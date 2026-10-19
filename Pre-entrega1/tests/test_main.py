"""Pruebas de la demostración con entorno y clientes simulados."""

import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import main as demo
from llmclient.schemas import ModelResponse, Provider


class MainTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        for target, options in (
            ("main.load_dotenv", {}),
            ("main.AsyncLLMManager", {}),
        ):
            patcher = patch(target, **options)
            mock = patcher.start()
            self.addCleanup(patcher.stop)
            if target.endswith("load_dotenv"):
                self.load_dotenv = mock
            else:
                self.manager_constructor = mock
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.events = []
        self.manager_constructor.side_effect = self.crear_manager_simulado

    def crear_manager_simulado(self, config):
        manager = MagicMock()

        async def generate(messages):
            self.events.append((config.provider, "normal"))
            return ModelResponse(
                provider=config.provider, model=config.model, content="Respuesta completa"
            )

        async def generate_stream(messages):
            self.events.append((config.provider, "stream"))
            yield "Primer fragmento "
            yield "segundo fragmento"

        manager.generate = AsyncMock(side_effect=generate)
        manager.generate_stream.side_effect = generate_stream
        return manager

    async def ejecutar_demo(self, environment):
        output = io.StringIO()
        with patch.dict(os.environ, environment), redirect_stdout(output):
            await demo.main()
        return output.getvalue()

    async def test_sin_claves_omite_todos_y_carga_env_junto_al_script(self):
        output = await self.ejecutar_demo({})
        self.assertEqual(output.count("Omitido:"), 2)
        self.assertIn("No se ejecutaron solicitudes:", output)
        self.manager_constructor.assert_not_called()
        self.load_dotenv.assert_called_once_with(Path(demo.__file__).resolve().parent / ".env")

    async def test_dos_proveedores_se_ejecutan_en_orden_y_en_ambos_modos(self):
        environment = {
            "OPENAI_API_KEY": "clave-ficticia-openai", "OPENAI_MODEL": "modelo-openai",
            "ANTHROPIC_API_KEY": "clave-ficticia-anthropic", "ANTHROPIC_MODEL": "modelo-claude",
            "LLM_TEMPERATURE": "0.5", "LLM_MAX_TOKENS": "123", "LLM_TIMEOUT_SECONDS": "12",
        }
        output = await self.ejecutar_demo(environment)
        self.assertEqual(self.events, [
            (Provider.OPENAI, "normal"), (Provider.OPENAI, "stream"),
            (Provider.ANTHROPIC, "normal"), (Provider.ANTHROPIC, "stream"),
        ])
        self.assertEqual(output.count("Respuesta completa"), 2)
        self.assertEqual(output.count("Primer fragmento segundo fragmento"), 2)
        self.assertNotIn("clave-ficticia", output)
        self.assertNotIn("No se ejecutaron solicitudes:", output)
        for call in self.manager_constructor.call_args_list:
            config = call.args[0]
            self.assertEqual(config.temperature, 0.5)
            self.assertEqual(config.max_tokens, 123)
            self.assertEqual(config.timeout_seconds, 12)


    async def test_lee_esfuerzo_solo_para_openai_y_acepta_variable_vacia(self):
        for effort in ("low", ""):
            with self.subTest(effort=effort):
                self.manager_constructor.reset_mock()
                await self.ejecutar_demo({
                    "OPENAI_API_KEY": "clave-ficticia", "OPENAI_MODEL": "gpt-6.1-sol",
                    "ANTHROPIC_API_KEY": "clave-ficticia", "ANTHROPIC_MODEL": "modelo-claude",
                    "OPENAI_REASONING_EFFORT": effort,
                })
                configs = [call.args[0] for call in self.manager_constructor.call_args_list]
                self.assertEqual(configs[0].reasoning_effort, effort or None)
                self.assertIsNone(configs[1].reasoning_effort)

    async def test_clave_de_espacios_se_omite(self):
        output = await self.ejecutar_demo({"OPENAI_API_KEY": " \n\t"})
        self.assertEqual(output.count("Omitido:"), 2)
        self.manager_constructor.assert_not_called()

    async def test_modelo_faltante_no_impide_ejecutar_siguiente_proveedor(self):
        output = await self.ejecutar_demo({
            "OPENAI_API_KEY": "clave-ficticia",
            "ANTHROPIC_API_KEY": "clave-ficticia", "ANTHROPIC_MODEL": "modelo-claude",
        })
        self.assertIn("Error de configuración: falta OPENAI_MODEL.", output)
        self.assertEqual(self.events, [(Provider.ANTHROPIC, "normal"), (Provider.ANTHROPIC, "stream")])

    async def test_modelos_ausentes_o_vacios_identifican_variable_sin_hacer_llamadas(self):
        for model in (None, "", " \n\t"):
            with self.subTest(model=model):
                environment = {
                    "OPENAI_API_KEY": "clave-ficticia",
                    "ANTHROPIC_API_KEY": "clave-ficticia",
                }
                if model is not None:
                    environment.update(OPENAI_MODEL=model, ANTHROPIC_MODEL=model)
                output = await self.ejecutar_demo(environment)
                self.assertIn("falta OPENAI_MODEL.", output)
                self.assertIn("falta ANTHROPIC_MODEL.", output)
                self.assertIn("No se ejecutaron solicitudes:", output)
                self.assertNotIn("clave-ficticia", output)
                self.manager_constructor.assert_not_called()

    async def test_parametros_invalidos_no_crean_clientes_ni_exponen_valores(self):
        for variable in ("LLM_TEMPERATURE", "LLM_MAX_TOKENS", "LLM_TIMEOUT_SECONDS"):
            with self.subTest(variable=variable):
                output = await self.ejecutar_demo({
                    "OPENAI_API_KEY": "clave-ficticia", "OPENAI_MODEL": "modelo-openai",
                    variable: "valor-sensible-ficticio",
                })
                self.assertIn(f"Error de configuración. Revisá: {variable}.", output)
                self.assertIn("No se ejecutaron solicitudes:", output)
                self.assertNotIn("valor-sensible-ficticio", output)
                self.assertNotIn("clave-ficticia", output)
                self.manager_constructor.assert_not_called()

    async def test_error_al_crear_cliente_no_impide_siguiente_proveedor(self):
        def create(config):
            if config.provider == Provider.OPENAI:
                raise ValueError("clave-ficticia")
            return self.crear_manager_simulado(config)

        self.manager_constructor.side_effect = create
        output = await self.ejecutar_demo({
            "OPENAI_API_KEY": "clave-ficticia", "OPENAI_MODEL": "modelo-openai",
            "ANTHROPIC_API_KEY": "clave-ficticia", "ANTHROPIC_MODEL": "modelo-claude",
        })
        self.assertIn("Error de configuración", output)
        self.assertNotIn("clave-ficticia", output)
        self.assertEqual(self.events, [(Provider.ANTHROPIC, "normal"), (Provider.ANTHROPIC, "stream")])

    async def test_error_controlado_no_impide_stream_ni_siguiente_proveedor(self):
        def create(config):
            manager = self.crear_manager_simulado(config)
            if config.provider == Provider.OPENAI:
                manager.generate = AsyncMock(return_value=ModelResponse(
                    provider=config.provider, model=config.model,
                    content="", error="Error de conexión",
                ))

                async def failed_stream(messages):
                    yield "\n[Error durante el streaming: error simulado]"

                manager.generate_stream.side_effect = failed_stream
            return manager

        self.manager_constructor.side_effect = create
        output = await self.ejecutar_demo({
            "OPENAI_API_KEY": "clave-ficticia", "OPENAI_MODEL": "modelo-openai",
            "ANTHROPIC_API_KEY": "clave-ficticia", "ANTHROPIC_MODEL": "modelo-claude",
        })
        self.assertIn("Error: Error de conexión", output)
        self.assertIn("Error durante el streaming", output)
        self.assertIn("Respuesta completa", output)
        self.assertEqual(self.events, [(Provider.ANTHROPIC, "normal"), (Provider.ANTHROPIC, "stream")])


if __name__ == "__main__":
    unittest.main()
