"""Casos y métricas de recuperación comprobados sin servicios externos."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from evaluate import calcular_metricas, cargar_golden, evaluar, imprimir_reporte, main
from schemas import CasoGolden


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "golden.json"
        (self.root / "memory.md").write_text("Memoria", encoding="utf-8")
        self.data = [{"pregunta": f"Pregunta {n}", "documento_id_esperado": "memory.md"} for n in range(5)]

    def escribir_golden(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def documentos(self, sources):
        return [Document(page_content=f"Chunk {n}", metadata={"source": source}) for n, source in enumerate(sources)]

    def test_carga_cinco_casos_y_golden_final_real(self):
        self.escribir_golden(self.data)
        cases = cargar_golden(self.path, self.root)
        self.assertEqual(len(cases), 5)
        self.assertTrue(all(isinstance(case, CasoGolden) for case in cases))
        real = cargar_golden()
        self.assertEqual(len(real), 5)
        self.assertEqual(real[3].documento_id_esperado, "memory.md")
        self.assertEqual(real[3].pregunta, "How can an agent retain context during a conversation and remember information across different sessions?")

    def test_rechaza_estructura_o_cantidad_incorrecta(self):
        for data in ({}, [], self.data[:4], self.data + self.data[:1]):
            with self.subTest(data=data):
                self.escribir_golden(data)
                with self.assertRaises(ValueError):
                    cargar_golden(self.path, self.root)

    def test_rechaza_caso_invalido_o_fuente_inexistente(self):
        for invalid in ({"pregunta": " "}, {"pregunta": "Pregunta", "documento_id_esperado": "no-existe.md"}):
            self.escribir_golden([invalid] + self.data[1:])
            with self.assertRaises(ValueError):
                cargar_golden(self.path, self.root)
        self.path.write_text("{ JSON inválido", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            cargar_golden(self.path, self.root)

    def test_metricas_vacio_ausente_repetidas_y_menos_de_cinco(self):
        cases = [
            ([], (0.0, 0.0)),
            (["otro.md"] * 5, (0.0, 0.0)),
            (["memory.md"] * 5, (1.0, 1.0)),
            (["memory.md", "otro.md", "memory.md", "otro.md", "otro.md"], (0.4, 1.0)),
            (["memory.md", "otro.md"], (0.5, 1.0)),
            (["memory.md"], (1.0, 1.0)),
        ]
        for sources, expected in cases:
            with self.subTest(sources=sources):
                self.assertEqual(calcular_metricas(self.documentos(sources), "memory.md"), expected)

    def test_fuente_faltante_no_se_trata_como_fallo_de_relevancia(self):
        with self.assertRaises(KeyError):
            calcular_metricas([Document(page_content="sin metadatos")], "memory.md")

    def test_evalua_cinco_consultas_y_promedia_sin_colapsar_fuentes(self):
        cases = [CasoGolden(**data) for data in self.data]
        rag = MagicMock()
        rag.obtener_top_k.side_effect = [
            self.documentos(["memory.md"] * 5),
            self.documentos(["memory.md", "otro.md"]),
            self.documentos(["otro.md"]), [],
            self.documentos(["memory.md", "memory.md", "otro.md", "otro.md", "otro.md"]),
        ]
        results = evaluar(rag, cases)
        self.assertEqual(rag.obtener_top_k.call_count, 5)
        self.assertEqual([call.args[0] for call in rag.obtener_top_k.call_args_list], [case.pregunta for case in cases])
        self.assertEqual(results[0]["fuentes_recuperadas"], ["memory.md"] * 5)
        output = io.StringIO()
        with redirect_stdout(output):
            imprimir_reporte(results)
        report = output.getvalue()
        self.assertIn("Precision@5: 0.3800", report)
        self.assertIn("Recall@5: 0.6000", report)
        self.assertIn("Fuente esperada: memory.md", report)
        self.assertIn("Pregunta 4", report)
        self.assertIn("coincidencias / resultados recuperados", report)

    def test_main_reporta_y_cierra_en_caso_exitoso(self):
        with patch("evaluate.load_dotenv"), patch("evaluate.logging.basicConfig"), patch("evaluate.cargar_configuracion"):
            with patch("evaluate.RAGSystem") as factory, redirect_stdout(io.StringIO()) as output:
                factory.return_value.obtener_top_k.return_value = []
                self.assertEqual(main(), 0)
                factory.return_value.cerrar.assert_called_once()
                self.assertIn("=== Promedios ===", output.getvalue())

    def test_error_operativo_no_publica_reporte_parcial_y_cierra(self):
        with patch("evaluate.load_dotenv"), patch("evaluate.logging.basicConfig"), patch("evaluate.cargar_configuracion"):
            with patch("evaluate.RAGSystem") as factory, redirect_stdout(io.StringIO()) as output:
                factory.return_value.obtener_top_k.side_effect = [[], RuntimeError("secreto-ficticio")]
                self.assertEqual(main(), 1)
                factory.return_value.cerrar.assert_called_once()
                self.assertIn("Evaluación incompleta", output.getvalue())
                self.assertNotIn("Promedios", output.getvalue())
                self.assertNotIn("secreto-ficticio", output.getvalue())

    def test_configuracion_invalida_no_inicia_recuperacion(self):
        with patch("evaluate.load_dotenv"), patch("evaluate.logging.basicConfig"):
            with patch("evaluate.cargar_configuracion", side_effect=ValueError("configuración ficticia")):
                with patch("evaluate.RAGSystem") as factory, redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 1)
                    factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
