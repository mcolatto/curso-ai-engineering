"""Consultas sobre JSON temporal y comprobación del corpus de demostración."""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from schemas import ResultadoHoteleria, ResultadoTurismo
from tools import DATA_DIR, buscar_hoteles, buscar_turismo


class ToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.hotel = {
            "destino": "Tokio", "id": "test-hotel", "nombre": "Hotel de prueba",
            "fuente": "catálogo de prueba ficticio", "disponibilidad": "sin_datos",
        }
        self.tourism = {
            "destino": "Tokio", "id": "test-turismo", "nombre": "Museo de prueba",
            "fuente": "catálogo de prueba ficticio", "categoria": "attraction",
        }

    def escribir(self, name, records):
        (self.root / name).write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    async def test_filtra_destino_y_conserva_procedencia_y_datos_incompletos(self):
        self.escribir("hoteles.json", [self.hotel, {**self.hotel, "destino": "Osaka", "id": "osaka"}])
        with patch("tools.DATA_DIR", self.root):
            result = ResultadoHoteleria.model_validate(await buscar_hoteles.ainvoke({"destino": " TOKIO "}))
        self.assertEqual(result.estado_busqueda, "completada")
        self.assertEqual([hotel.id for hotel in result.hoteles], ["test-hotel"])
        self.assertEqual(result.hoteles[0].fuente, self.hotel["fuente"])
        self.assertIsNone(result.hoteles[0].capacidad)
        self.assertIsNone(result.hoteles[0].coordenadas)
        self.assertIsNone(result.hoteles[0].periodos_disponibles)

    async def test_conserva_capacidad_y_periodos_sin_filtrar_compatibilidad(self):
        hotel = {**self.hotel, "capacidad": 1, "disponibilidad": "disponible_en_fuente",
                 "periodos_disponibles": [{"inicio": "2027-02-01", "fin": "2027-02-10"}]}
        self.escribir("hoteles.json", [hotel])
        with patch("tools.DATA_DIR", self.root):
            result = await buscar_hoteles.ainvoke({"destino": "Tokio"})
        self.assertEqual(result["hoteles"][0]["capacidad"], 1)
        self.assertEqual(result["hoteles"][0]["periodos_disponibles"], hotel["periodos_disponibles"])

    async def test_turismo_consulta_registros_reales_y_conserva_restricciones(self):
        self.escribir("turismo.json", [{**self.tourism, "restricciones": ["Cierra los lunes"]}])
        with patch("tools.DATA_DIR", self.root):
            first = await buscar_turismo.ainvoke({"destino": "Tokio"})
            self.escribir("turismo.json", [{**self.tourism, "nombre": "Nombre actualizado"}])
            second = await buscar_turismo.ainvoke({"destino": "Tokio"})
        self.assertEqual(first["opciones"][0]["restricciones"], ["Cierra los lunes"])
        self.assertEqual(second["opciones"][0]["nombre"], "Nombre actualizado")
        self.assertEqual(first["opciones"][0]["fuente"], self.tourism["fuente"])

    async def test_destino_ausente_o_catalogo_vacio_son_busquedas_completadas(self):
        for tool, file, data, field in (
            (buscar_hoteles, "hoteles.json", self.hotel, "hoteles"),
            (buscar_turismo, "turismo.json", self.tourism, "opciones"),
        ):
            for records, destino in (([data], "Osaka"), ([], "Tokio")):
                with self.subTest(tool=tool.name, destino=destino):
                    self.escribir(file, records)
                    with patch("tools.DATA_DIR", self.root):
                        result = await tool.ainvoke({"destino": destino})
                    self.assertEqual(result["estado_busqueda"], "completada")
                    self.assertEqual(result[field], [])

    async def test_archivo_ausente_ilegible_o_invalido_se_informa_como_error(self):
        for tool, file, model in (
            (buscar_hoteles, "hoteles.json", ResultadoHoteleria),
            (buscar_turismo, "turismo.json", ResultadoTurismo),
        ):
            path = self.root / file
            for content in (None, b"\xff", b"{mal-json", b"{}", b"[1]",
                            b'[{"destino":"Tokio"}]', b'[{"destino":" "}]'):
                with self.subTest(tool=tool.name, content=content):
                    if content is None:
                        path.unlink(missing_ok=True)
                    else:
                        path.write_bytes(content)
                    with patch("tools.DATA_DIR", self.root):
                        result = model.model_validate(await tool.ainvoke({"destino": "Tokio"}))
                    self.assertEqual(result.estado_busqueda, "error")
                    self.assertTrue(result.observaciones)
        with patch("tools.leer_catalogo", side_effect=PermissionError("secreto-ficticio")):
            result = await buscar_hoteles.ainvoke({"destino": "Tokio"})
        self.assertNotIn("secreto-ficticio", json.dumps(result))

    async def test_destino_invalido_se_rechaza_antes_de_leer(self):
        for tool in (buscar_hoteles, buscar_turismo):
            with patch("tools.leer_catalogo") as reader:
                for arguments in ({}, {"destino": None}, {"destino": 123}):
                    with self.subTest(tool=tool.name, arguments=arguments):
                        with self.assertRaises(ValidationError):
                            await tool.ainvoke(arguments)
                with self.assertRaises(ValueError):
                    await tool.ainvoke({"destino": "  "})
                reader.assert_not_called()

    async def test_corpus_soporta_febrero_y_marzo_sin_respuestas_prefabricadas(self):
        hotels = ResultadoHoteleria.model_validate(await buscar_hoteles.ainvoke({"destino": "Tokio"}))
        tourism = ResultadoTurismo.model_validate(await buscar_turismo.ainvoke({"destino": "Tokio"}))
        self.assertEqual(hotels.estado_busqueda, "completada")
        self.assertEqual(tourism.estado_busqueda, "completada")
        self.assertTrue(hotels.hoteles)
        self.assertEqual({option.categoria for option in tourism.opciones}, {"attraction", "viewpoint", "gastronomy"})
        for month, hotel_expected in ((2, True), (3, False)):
            start, end = date(2027, month, 12), date(2027, month, 20)
            compatible = [hotel for hotel in hotels.hoteles if hotel.capacidad >= 4 and any(
                period.inicio <= start and period.fin >= end for period in hotel.periodos_disponibles
            )]
            self.assertEqual(bool(compatible), hotel_expected)
            self.assertTrue(all(any(period.inicio <= start and period.fin >= end
                                    for period in option.periodos_disponibles) for option in tourism.opciones))
        for file in ("hoteles.json", "turismo.json"):
            records = json.loads((DATA_DIR / file).read_text(encoding="utf-8"))
            self.assertEqual(len({record["id"] for record in records}), len(records))
            self.assertTrue(all(record["fuente"] == f"data/{file}" for record in records))


if __name__ == "__main__":
    unittest.main()
