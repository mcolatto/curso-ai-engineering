"""Validación de contratos sin clientes ni llamadas a servicios externos."""

import json
import unittest

from pydantic import ValidationError

from schemas import CasoGolden, MetadataChunk


class SchemasTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            (MetadataChunk, {
                "text": "  # Memoria\n\nContenido del fragmento.\n",
                "source": "memory.md",
                "categoria": "memory",
                "chunk_id": "memory-0-identificador-ficticio",
            }),
            (CasoGolden, {
                "pregunta": "How does memory work?",
                "documento_id_esperado": "memory.md",
            }),
        ]

    def test_acepta_datos_validos_y_conserva_contenido(self):
        for model, data in self.cases:
            with self.subTest(model=model.__name__):
                self.assertEqual(model(**data).model_dump(), data)

    def test_rechaza_cada_campo_faltante(self):
        for model, data in self.cases:
            for field in data:
                incomplete = data.copy()
                del incomplete[field]
                with self.subTest(model=model.__name__, field=field):
                    with self.assertRaises(ValidationError):
                        model(**incomplete)

    def test_rechaza_textos_vacios_o_solo_espacios(self):
        for model, data in self.cases:
            for field in data:
                for value in ("", "   ", "\n\t "):
                    with self.subTest(model=model.__name__, field=field, value=value):
                        with self.assertRaises(ValidationError):
                            model(**{**data, field: value})

    def test_rechaza_tipos_incorrectos(self):
        for model, data in self.cases:
            for field in data:
                for value in (None, 123, [], {}):
                    with self.subTest(model=model.__name__, field=field, value=value):
                        with self.assertRaises(ValidationError):
                            model(**{**data, field: value})

    def test_exporta_diccionario_y_json_sin_alterar_datos(self):
        for model, data in self.cases:
            with self.subTest(model=model.__name__):
                instance = model(**data)
                exported = instance.model_dump()
                self.assertTrue(all(isinstance(value, str) for value in exported.values()))
                self.assertEqual(json.loads(instance.model_dump_json()), data)
                self.assertEqual(json.loads(json.dumps(exported)), data)


if __name__ == "__main__":
    unittest.main()
