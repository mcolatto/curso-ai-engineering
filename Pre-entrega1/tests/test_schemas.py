"""Pruebas de validación locales, sin red ni credenciales reales."""

import unittest

from pydantic import SecretStr, ValidationError

from llmclient.schemas import ChatMessage, LLMConfig, ModelResponse, Provider


class ChatMessageTests(unittest.TestCase):
    def test_valida_esfuerzo_de_razonamiento(self):
        for effort in (None, "low", "medium", "high", "xhigh", "max"):
            config = LLMConfig(provider=Provider.OPENAI, model="modelo", reasoning_effort=effort)
            self.assertEqual(config.reasoning_effort, effort)
        for effort in ("loow", "", "none"):
            with self.assertRaises(ValidationError):
                LLMConfig(provider=Provider.OPENAI, model="modelo", reasoning_effort=effort)

    def test_acepta_los_tres_roles(self):
        for role in ("user", "assistant", "system"):
            with self.subTest(role=role):
                message = ChatMessage(role=role, content="Hola")
                self.assertEqual(message.role, role)
                self.assertEqual(message.content, "Hola")

    def test_rechaza_rol_desconocido(self):
        with self.assertRaises(ValidationError):
            ChatMessage(role="desconocido", content="Hola")


class LLMConfigTests(unittest.TestCase):
    def test_acepta_proveedores_y_valores_iniciales(self):
        for provider in Provider:
            with self.subTest(provider=provider):
                config = LLMConfig(provider=provider.value, model="modelo-de-prueba")
                self.assertEqual(config.provider, provider)
                self.assertEqual(config.temperature, 0.7)
                self.assertEqual(config.max_tokens, 1024)
                self.assertEqual(config.timeout_seconds, 30)
                self.assertIsNone(config.openai_api_key)
                self.assertIsNone(config.anthropic_api_key)

    def test_rechaza_proveedor_desconocido(self):
        with self.assertRaises(ValidationError):
            LLMConfig(provider="desconocido", model="modelo-de-prueba")

    def test_rechaza_modelos_vacios(self):
        for model in ("", "   ", "\n\t"):
            with self.subTest(model=model):
                with self.assertRaises(ValidationError):
                    LLMConfig(provider=Provider.OPENAI, model=model)

    def test_quita_espacios_exteriores_del_modelo(self):
        config = LLMConfig(provider=Provider.OPENAI, model=" modelo-de-prueba ")
        self.assertEqual(config.model, "modelo-de-prueba")

    def test_acepta_limites_de_temperatura(self):
        for temperature in (0, 2):
            with self.subTest(temperature=temperature):
                config = LLMConfig(
                    provider=Provider.OPENAI,
                    model="modelo-de-prueba",
                    temperature=temperature,
                )
                self.assertEqual(config.temperature, temperature)

    def test_rechaza_parametros_fuera_de_rango(self):
        invalid_parameters = [
            {"temperature": -0.1},
            {"temperature": 2.1},
            {"temperature": float("nan")},
            {"max_tokens": 0},
            {"max_tokens": -1},
            {"max_tokens": 1.5},
            {"timeout_seconds": 0},
            {"timeout_seconds": -1},
            {"timeout_seconds": float("inf")},
            {"timeout_seconds": float("nan")},
        ]
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValidationError):
                    LLMConfig(
                        provider=Provider.OPENAI,
                        model="modelo-de-prueba",
                        **parameters,
                    )

    def test_oculta_claves_en_representacion_y_json(self):
        fake_key = "clave-ficticia-para-test"
        for field in ("openai_api_key", "anthropic_api_key"):
            with self.subTest(field=field):
                config = LLMConfig(
                    provider=Provider.OPENAI,
                    model="modelo-de-prueba",
                    **{field: fake_key},
                )
                self.assertIsInstance(getattr(config, field), SecretStr)
                self.assertNotIn(fake_key, repr(config))
                self.assertNotIn(fake_key, str(config))
                self.assertNotIn(fake_key, config.model_dump_json())


class ModelResponseTests(unittest.TestCase):
    def test_respuesta_exitosa(self):
        response = ModelResponse(
            provider=Provider.ANTHROPIC,
            model="modelo-de-prueba",
            content="Respuesta simulada",
        )
        self.assertEqual(response.content, "Respuesta simulada")
        self.assertIsNone(response.error)

    def test_error_controlado_con_contenido_vacio(self):
        response = ModelResponse(
            provider=Provider.ANTHROPIC,
            model="modelo-de-prueba",
            content="",
            error="Error de conexión",
        )
        self.assertEqual(response.content, "")
        self.assertEqual(response.error, "Error de conexión")

    def test_rechaza_contenido_que_no_es_texto(self):
        with self.assertRaises(ValidationError):
            ModelResponse(provider=Provider.ANTHROPIC, model="modelo-de-prueba", content=None)


if __name__ == "__main__":
    unittest.main()
