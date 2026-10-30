"""Pruebas de la consola exploratoria con RAGSystem simulado."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from langchain_core.documents import Document

from main import main, mostrar_resultados


class MainTests(unittest.TestCase):
    def setUp(self):
        for name in ("load_dotenv", "logging.basicConfig", "cargar_configuracion", "RAGSystem"):
            patcher = patch(f"main.{name}")
            mock = patcher.start()
            self.addCleanup(patcher.stop)
            if name == "RAGSystem":
                self.factory = mock
            elif name == "cargar_configuracion":
                self.config = mock
        self.rag = self.factory.return_value
        self.documents = [Document(
            page_content="Texto recuperado",
            metadata={"source": "memory.md", "categoria": "memory", "chunk_id": "id-1"},
        )]
        self.rag.obtener_top_k.return_value = self.documents

    def test_pregunta_por_argumento_reutiliza_rag_y_cierra(self):
        with patch("builtins.input") as input_mock, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["How does memory work?"]), 0)
        input_mock.assert_not_called()
        self.factory.assert_called_once_with(self.config.return_value)
        self.rag.obtener_top_k.assert_called_once_with("How does memory work?")
        self.rag.cerrar.assert_called_once()
        self.assertIn("Texto recuperado", output.getvalue())

    def test_pregunta_por_consola_sin_bucle(self):
        with patch("builtins.input", return_value="Pregunta exploratoria") as input_mock:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([]), 0)
        input_mock.assert_called_once()
        self.rag.obtener_top_k.assert_called_once_with("Pregunta exploratoria")

    def test_rechaza_preguntas_vacias_antes_de_crear_rag(self):
        for query in ("", " \n\t"):
            with self.subTest(query=query), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main([query]), 1)
                self.assertIn("texto no vacío", output.getvalue())
        self.factory.assert_not_called()
        self.config.assert_not_called()

    def test_muestra_ranking_metadatos_y_extracto_de_hasta_200(self):
        documents = [
            Document(page_content="x" * 250 + "FIN-OCULTO", metadata={
                "source": "segundo.md", "categoria": "segundo", "chunk_id": "id-2",
            }),
            self.documents[0],
        ]
        with redirect_stdout(io.StringIO()) as output:
            mostrar_resultados(documents)
        text = output.getvalue()
        self.assertLess(text.index("1. Fuente: segundo.md"), text.index("2. Fuente: memory.md"))
        self.assertIn("Categoría: segundo", text)
        self.assertIn("ID: id-2", text)
        self.assertIn("Extracto: " + "x" * 200 + "\n", text)
        self.assertNotIn("x" * 201, text)
        self.assertNotIn("FIN-OCULTO", text)

    def test_resultados_vacios_se_informan(self):
        self.rag.obtener_top_k.return_value = []
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["consulta"]), 0)
        self.assertIn("No se recuperaron resultados", output.getvalue())
        self.rag.cerrar.assert_called_once()

    def test_error_operativo_se_informa_sin_secretos_y_cierra(self):
        self.rag.obtener_top_k.side_effect = RuntimeError("secreto-ficticio")
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["consulta"]), 1)
        self.rag.cerrar.assert_called_once()
        self.assertIn("Consulta incompleta", output.getvalue())
        self.assertNotIn("secreto-ficticio", output.getvalue())
        self.assertNotIn("Top-5 híbrido", output.getvalue())

    def test_configuracion_invalida_no_crea_rag(self):
        self.config.side_effect = ValueError("secreto-ficticio")
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["consulta"]), 1)
        self.factory.assert_not_called()
        self.assertNotIn("secreto-ficticio", output.getvalue())

    def test_fin_de_entrada_sin_pregunta(self):
        with patch("builtins.input", side_effect=EOFError), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([]), 1)
        self.assertIn("No se recibió una pregunta", output.getvalue())
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
