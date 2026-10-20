"""Pruebas de la demostración sin claves reales ni llamadas de red."""

import io
import json
import os
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx2
from openai import (
    APIConnectionError, APIError, APITimeoutError, AuthenticationError,
    PermissionDeniedError, RateLimitError,
)

import main as demo
from chain import SalidaInvalidaError
from schemas import EntidadesTecnicas


class MainTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for target in ("main.load_dotenv", "main.logging.basicConfig", "main.process_text"):
            patcher = patch(target)
            mock = patcher.start()
            self.addCleanup(patcher.stop)
            if target.endswith("load_dotenv"):
                self.load_dotenv = mock
            elif target.endswith("basicConfig"):
                self.logging_config = mock
            else:
                self.process = mock

    async def run_demo(self):
        output = io.StringIO()
        with redirect_stdout(output):
            await demo.main()
        return output.getvalue()

    async def test_muestra_json_y_espera_el_procesamiento_del_ejemplo(self):
        result = EntidadesTecnicas(
            tecnologias=["FastAPI", "Redis", "PostgreSQL"],
            nivel_de_criticidad="alta", resumen_tecnico="La API está caída.",
        )
        ambiguous = EntidadesTecnicas(
            tecnologias=[], nivel_de_criticidad="media",
            resumen_tecnico="Se reportan demoras sin identificar el componente.",
        )
        self.process.side_effect = [result, ambiguous]
        output = await self.run_demo()
        first, second = output.split("=== Texto ambiguo ===")
        self.assertEqual(json.loads(first.split("=== Texto completo ===")[1]), result.model_dump(mode="json"))
        self.assertEqual(json.loads(second), ambiguous.model_dump(mode="json"))
        self.assertEqual(self.process.await_count, 2)
        text = self.process.call_args_list[0].args[0]
        self.assertIn("no tenemos métricas", self.process.call_args_list[1].args[0])
        for technology in result.tecnologias:
            self.assertIn(technology, text)
        self.load_dotenv.assert_called_once_with(Path(demo.__file__).resolve().parent / ".env")
        self.logging_config.assert_called_once()

    async def test_reintentos_agotados_no_muestran_traceback_ni_detalles(self):
        self.process.side_effect = SalidaInvalidaError("texto-privado clave-ficticia")
        output = await self.run_demo()
        self.assertIn("tras los dos intentos", output)
        self.assertNotIn("texto-privado", output)
        self.assertNotIn("clave-ficticia", output)
        self.assertNotIn("Traceback", output)
        self.assertEqual(self.process.await_count, 2)

    async def test_fallo_del_primer_caso_no_impide_el_segundo(self):
        result = EntidadesTecnicas(
            tecnologias=[], nivel_de_criticidad="media", resumen_tecnico="Demoras sin diagnóstico.",
        )
        self.process.side_effect = [SalidaInvalidaError("detalle-privado"), result]
        output = await self.run_demo()
        self.assertEqual(self.process.await_count, 2)
        self.assertIn("tras los dos intentos", output)
        second = output.split("=== Texto ambiguo ===")[1]
        self.assertEqual(json.loads(second), result.model_dump(mode="json"))
        self.assertNotIn("detalle-privado", output)

    async def test_configuracion_invalida_se_comunica_sin_exponer_valores(self):
        self.process.side_effect = ValueError("valor-sensible-ficticio")
        output = await self.run_demo()
        self.assertIn(".env.example", output)
        self.assertNotIn("valor-sensible-ficticio", output)

    async def test_errores_api_se_comunican_sin_reintentar_desde_main(self):
        request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
        cases = [
            (AuthenticationError("clave-ficticia", response=httpx2.Response(401, request=request), body=None), "autenticación"),
            (PermissionDeniedError("clave-ficticia", response=httpx2.Response(403, request=request), body=None), "permisos"),
            (RateLimitError("clave-ficticia", response=httpx2.Response(429, request=request), body=None), "cuota"),
            (APITimeoutError(request=request), "tiempo de espera"),
            (APIConnectionError(request=request), "conexión"),
            (APIError("clave-ficticia", request=request, body=None), "Error de la API"),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                self.process.reset_mock()
                self.process.side_effect = error
                output = await self.run_demo()
                self.assertIn(expected, output)
                self.assertNotIn("clave-ficticia", output)
                self.assertNotIn("Traceback", output)
                self.assertEqual(self.process.await_count, 2)

    def test_importar_main_no_ejecuta_la_demostracion(self):
        with patch("chain.process_text", new_callable=AsyncMock) as process:
            runpy.run_module("main", run_name="main_import_test")
            process.assert_not_called()
            self.load_dotenv.assert_not_called()
            self.logging_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
