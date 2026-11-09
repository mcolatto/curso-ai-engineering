"""Celdas de Colab verificadas localmente sin clonar, instalar ni consumir APIs."""

import ast
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


PROJECT = Path(__file__).resolve().parent.parent
NOTEBOOK = PROJECT / "Pre_entrega6_Demostracion_Colab.ipynb"


class NotebookTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.code = ["".join(cell["source"]) for cell in self.book["cells"] if cell["cell_type"] == "code"]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_formato_y_sintaxis_sin_evidencia_prefabricada(self):
        self.assertEqual(self.book["nbformat"], 4)
        self.assertEqual(len(self.code), 6)
        for cell in self.book["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                compile("".join(cell["source"]), str(NOTEBOOK), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

    def test_clone_y_rutas_con_repositorio_simulado_y_prerrequisito_de_publicacion(self):
        repository = self.root / "repo"
        source = self.code[0].replace('/content/curso-ai-engineering', str(repository))
        namespace = {}
        with patch("subprocess.run") as run, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "todavía no está publicada"):
                exec(source, namespace)
        self.assertEqual(run.call_args.args[0], ["git", "clone", "--depth", "1", "--branch", "main",
            "https://github.com/mcolatto/curso-ai-engineering.git", str(repository)])
        project = repository / "Pre-entrega6"
        project.mkdir(parents=True)
        for name in namespace["required"]:
            self.assertTrue((PROJECT / name).is_file())
            target = project / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        previous = Path.cwd()
        try:
            with patch("subprocess.run") as run, patch("subprocess.check_output", return_value="commit-test\n"):
                with redirect_stdout(io.StringIO()):
                    exec(source, namespace)
            run.assert_not_called()
            self.assertEqual(Path.cwd(), project.resolve())
            self.assertEqual(namespace["commit"], "commit-test")
        finally:
            os.chdir(previous)
            if str(project) in sys.path:
                sys.path.remove(str(project))

    def test_instalacion_reutiliza_requirements_sin_ejecutarla(self):
        namespace = {"sys": sys}
        import subprocess
        namespace["subprocess"] = subprocess
        with patch("subprocess.run") as run:
            exec(self.code[1], namespace)
        self.assertEqual(run.call_args_list[0].args[0],
                         [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
        self.assertEqual(run.call_args_list[1].args[0],
                         [sys.executable, "-m", "pip", "install", "-q", "jedi", "langchain-core==1.7.4"])
        self.assertEqual(run.call_args_list[2].args[0], [sys.executable, "-m", "pip", "check"])
        self.assertEqual(run.call_count, 3)
        self.assertTrue(all(call.kwargs["check"] for call in run.call_args_list))

    def test_secreto_en_entorno_sin_imprimirlo_y_error_sin_detalles(self):
        colab = types.ModuleType("google.colab")
        colab.userdata = MagicMock()
        colab.userdata.get.return_value = " clave-ficticia "
        with patch.dict(sys.modules, {"google.colab": colab}), patch.dict(os.environ, {}, clear=True):
            namespace = {"os": os}
            with redirect_stdout(io.StringIO()) as output:
                exec(self.code[2], namespace)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "clave-ficticia")
            self.assertNotIn("clave-ficticia", output.getvalue())
            self.assertNotIn("clave", namespace)
            colab.userdata.get.assert_called_once_with("OPENAI_API_KEY")
            colab.userdata.get.side_effect = RuntimeError("detalle-sensible")
            with self.assertRaises(RuntimeError) as raised:
                exec(self.code[2], namespace)
            self.assertNotIn("detalle-sensible", str(raised.exception))

    async def test_ejecucion_reutiliza_main_y_no_presenta_traza_antigua_como_nueva(self):
        path = self.root / "traza.json"
        path.write_text('{"evidencia": "anterior"}')
        config = types.ModuleType("config")
        config.TRACE_PATH = path
        app = types.ModuleType("main")
        app.main = AsyncMock(return_value=1)
        namespace = {}
        with patch.dict(sys.modules, {"config": config, "main": app}):
            with redirect_stdout(io.StringIO()) as output:
                await eval(compile(self.code[3], "celda_ejecucion", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace)
            self.assertFalse(namespace["traza_generada_en_esta_ejecucion"])
            self.assertIn("No se generó una traza nueva", output.getvalue())

            async def demo():
                path.write_text(json.dumps({"prueba": {"eventos": [], "estado_final": {},
                                                       "resultado_prueba": {"satisfactoria": False}}}))
                return 1

            app.main.side_effect = demo
            with redirect_stdout(io.StringIO()):
                await eval(compile(self.code[3], "celda_ejecucion", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace)
                exec(self.code[4], namespace)
            self.assertTrue(namespace["traza_generada_en_esta_ejecucion"])
            self.assertFalse(namespace["traza"]["prueba"]["resultado_prueba"]["satisfactoria"])
        self.assertEqual(app.main.await_count, 2)


if __name__ == "__main__":
    unittest.main()
