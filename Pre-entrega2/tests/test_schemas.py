"""Pruebas del contrato de salida sin red ni credenciales."""

import json
import unittest

from pydantic import ValidationError

from schemas import EntidadesTecnicas, NivelDeCriticidad


class EntidadesTecnicasTests(unittest.TestCase):
    def setUp(self):
        self.valid_data = {
            "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
            "nivel_de_criticidad": "alta",
            "resumen_tecnico": "La API no puede conectarse a PostgreSQL.",
        }

    def test_acepta_datos_validos_y_los_tres_niveles(self):
        for nivel in NivelDeCriticidad:
            with self.subTest(nivel=nivel):
                data = {**self.valid_data, "nivel_de_criticidad": nivel.value}
                result = EntidadesTecnicas(**data)
                self.assertEqual(result.nivel_de_criticidad, nivel)
                self.assertEqual(result.tecnologias, self.valid_data["tecnologias"])
                self.assertEqual(result.resumen_tecnico, self.valid_data["resumen_tecnico"])

    def test_acepta_lista_de_tecnologias_vacia(self):
        result = EntidadesTecnicas(**{**self.valid_data, "tecnologias": []})
        self.assertEqual(result.tecnologias, [])

    def test_rechaza_cada_campo_faltante(self):
        for field in self.valid_data:
            with self.subTest(field=field):
                data = self.valid_data.copy()
                del data[field]
                with self.assertRaises(ValidationError) as raised:
                    EntidadesTecnicas(**data)
                self.assertEqual(raised.exception.errors()[0]["loc"], (field,))

    def test_rechaza_criticidad_desconocida(self):
        with self.assertRaises(ValidationError):
            EntidadesTecnicas(**{**self.valid_data, "nivel_de_criticidad": "urgente"})

    def test_rechaza_tipos_incorrectos(self):
        cases = [
            ("tecnologias", "Redis"),
            ("tecnologias", [123]),
            ("tecnologias", None),
            ("nivel_de_criticidad", 1),
            ("nivel_de_criticidad", None),
            ("resumen_tecnico", 123),
            ("resumen_tecnico", ["Resumen"]),
            ("resumen_tecnico", None),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    EntidadesTecnicas(**{**self.valid_data, field: value})

    def test_rechaza_textos_vacios_o_solo_espacios(self):
        for value in ("", "   ", "\n\t "):
            for field, content in (
                ("tecnologias", ["Redis", value]),
                ("resumen_tecnico", value),
            ):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValidationError):
                        EntidadesTecnicas(**{**self.valid_data, field: content})

    def test_normaliza_tecnologias_y_conserva_el_resumen(self):
        result = EntidadesTecnicas(**{
            **self.valid_data,
            "tecnologias": [" Redis ", "\tPostgreSQL\n"],
            "resumen_tecnico": " Resumen técnico.\n",
        })
        self.assertEqual(result.tecnologias, ["Redis", "PostgreSQL"])
        self.assertEqual(result.resumen_tecnico, " Resumen técnico.\n")

    def test_exporta_json_con_los_nombres_y_valores_esperados(self):
        result = EntidadesTecnicas(**self.valid_data)
        self.assertEqual(json.loads(result.model_dump_json()), self.valid_data)


if __name__ == "__main__":
    unittest.main()
