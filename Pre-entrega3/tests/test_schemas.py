"""Pruebas de los contratos de salida RAG, sin red ni credenciales."""

import json
import unittest

from pydantic import ValidationError

from schemas import RAGResponse, RespuestaLLM


class EsquemasTests(unittest.TestCase):
    def setUp(self):
        self.rag_data = {
            "respuesta": "RAG incorpora información recuperada al contexto del modelo.",
            "fuentes": ["rag.txt", "vectordb.txt"],
            "fragmentos_recuperados": 4,
        }

    def test_acepta_respuesta_llm_y_resultado_rag(self):
        llm = RespuestaLLM(respuesta=self.rag_data["respuesta"])
        rag = RAGResponse(**self.rag_data)
        self.assertEqual(llm.respuesta, self.rag_data["respuesta"])
        self.assertEqual(rag.model_dump(), self.rag_data)
        self.assertEqual(set(llm.model_dump()), {"respuesta"})

    def test_no_lo_se_es_valido_con_o_sin_contexto_recuperado(self):
        self.assertEqual(RespuestaLLM(respuesta="No lo sé").respuesta, "No lo sé")
        for fuentes, count in (([], 0), (["rag.txt"], 3)):
            with self.subTest(fuentes=fuentes, count=count):
                result = RAGResponse(
                    respuesta="No lo sé", fuentes=fuentes, fragmentos_recuperados=count,
                )
                self.assertEqual(result.fuentes, fuentes)
                self.assertEqual(result.fragmentos_recuperados, count)

    def test_rechaza_respuesta_vacia_en_ambos_modelos(self):
        for content in ("", "   ", "\n\t "):
            for schema, data in (
                (RespuestaLLM, {"respuesta": content}),
                (RAGResponse, {**self.rag_data, "respuesta": content}),
            ):
                with self.subTest(schema=schema.__name__, content=content):
                    with self.assertRaises(ValidationError):
                        schema(**data)

    def test_rechaza_nombres_de_fuentes_vacios(self):
        for fuente in ("", "   ", "\n\t"):
            with self.subTest(fuente=fuente):
                with self.assertRaises(ValidationError):
                    RAGResponse(**{**self.rag_data, "fuentes": ["rag.txt", fuente]})

    def test_rechaza_contador_negativo_o_fraccionario(self):
        for count in (-1, 1.5):
            with self.subTest(count=count):
                with self.assertRaises(ValidationError):
                    RAGResponse(**{**self.rag_data, "fragmentos_recuperados": count})

    def test_rechaza_campos_faltantes(self):
        for schema, data in (
            (RespuestaLLM, {"respuesta": "Texto"}), (RAGResponse, self.rag_data),
        ):
            for field in data:
                with self.subTest(schema=schema.__name__, field=field):
                    incomplete = data.copy()
                    del incomplete[field]
                    with self.assertRaises(ValidationError) as raised:
                        schema(**incomplete)
                    self.assertEqual(raised.exception.errors()[0]["loc"], (field,))

    def test_rechaza_tipos_incorrectos(self):
        cases = [
            ("respuesta", None), ("respuesta", 123), ("respuesta", ["Texto"]),
            ("fuentes", "rag.txt"), ("fuentes", [123]), ("fuentes", None),
            ("fragmentos_recuperados", None), ("fragmentos_recuperados", "varios"),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    RAGResponse(**{**self.rag_data, field: value})
        for value in (None, 123, ["Texto"]):
            with self.subTest(llm_respuesta=value):
                with self.assertRaises(ValidationError):
                    RespuestaLLM(respuesta=value)

    def test_exporta_json_y_conserva_texto_valido(self):
        text = " Respuesta técnica.\n"
        llm = RespuestaLLM(respuesta=text)
        rag = RAGResponse(**{**self.rag_data, "respuesta": text})
        self.assertEqual(json.loads(llm.model_dump_json()), {"respuesta": text})
        self.assertEqual(json.loads(rag.model_dump_json()), {**self.rag_data, "respuesta": text})


if __name__ == "__main__":
    unittest.main()
