"""Comparación descriptiva de archivos históricos y nuevos, sin servicios externos."""

import json
import os
import unittest
from decimal import Decimal
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from scripts import comparar_lotes as comparacion
from scripts.prueba_concurrencia import Medicion, PROMPT, Reporte, guardar


class ComparacionTests(unittest.TestCase):
    def setUp(self):
        self.referencia = Reporte(
            lote_id=uuid4(), proyecto_id=uuid4(), iniciado_en=datetime.now(UTC),
            estado="COMPLETO", latencia_p95_s=5,
            mediciones=[
                Medicion(
                    peticion=numero, job_id=uuid4(), estado="DONE",
                    latencia_traza_s=numero, tokens=100, costo_usd=Decimal("0.01"),
                )
                for numero in range(1, 6)
            ],
        )
        self.actual = self.referencia.model_copy(deep=True)
        self.actual.lote_id = uuid4()
        self.actual.prompt = PROMPT
        for numero, medicion in enumerate(self.actual.mediciones, start=1):
            medicion.job_id = uuid4()
            medicion.respuesta = f"Recomendación de alojamiento y gastronomía {numero}."
            medicion.configuracion = {"model": "modelo-prueba"}
            medicion.latencia_traza_s *= 2
            medicion.tokens *= 2
            medicion.costo_usd *= 2
        self.actual.latencia_p95_s *= 2

    def test_referencia_incompleta_no_inventa_configuracion_ni_respuestas(self):
        self.assertEqual(self.referencia.latencia_p95_s, 5)
        self.assertIsNone(self.referencia.prompt)
        for medicion in self.referencia.mediciones:
            self.assertIsNone(medicion.configuracion)
            self.assertIsNone(medicion.respuesta)

    def test_diferencias_de_latencia_tokens_y_costos_son_descriptivas(self):
        resultado = comparacion.comparar(self.referencia, self.actual)
        historico = resultado["referencia_historica"]
        actual = resultado["corrida_actual"]
        self.assertEqual(Decimal(historico["costo_total_usd"]), Decimal("0.05"))
        self.assertEqual(Decimal(actual["costo_total_usd"]), Decimal("0.10"))
        self.assertEqual(actual["respuestas_registradas"], 5)
        self.assertEqual(historico["respuestas_registradas"], 0)
        for variacion in resultado["variaciones"].values():
            self.assertAlmostEqual(variacion["porcentaje"], 100)
        self.assertEqual(len(actual["latencias_traza_s"]), 5)
        self.assertEqual(len(resultado["avisos"]), 3)
        self.assertNotIn("aprobado", resultado)

    def test_base_cero_no_produce_division_por_cero(self):
        for medicion in self.referencia.mediciones:
            medicion.latencia_traza_s = 0
            medicion.tokens = 0
            medicion.costo_usd = Decimal(0)
        self.referencia.latencia_p95_s = 0
        resultado = comparacion.comparar(self.referencia, self.actual)
        self.assertTrue(all(variacion["porcentaje"] is None for variacion in resultado["variaciones"].values()))

    def test_rechaza_lotes_incompletos_mezclados_o_con_metricas_incoherentes(self):
        for caso in ("incompleto", "failed", "sin_costo", "id_repetido", "mismo_lote", "trabajo_compartido", "p95"):
            with self.subTest(caso=caso):
                actual = self.actual.model_copy(deep=True)
                if caso == "incompleto":
                    actual.estado = "INCOMPLETO"
                elif caso == "failed":
                    actual.mediciones[0].estado = "FAILED"
                elif caso == "sin_costo":
                    actual.mediciones[0].costo_usd = None
                elif caso == "id_repetido":
                    actual.mediciones[0].job_id = actual.mediciones[1].job_id
                elif caso == "mismo_lote":
                    actual.lote_id = self.referencia.lote_id
                elif caso == "trabajo_compartido":
                    actual.mediciones[0].job_id = self.referencia.mediciones[0].job_id
                else:
                    actual.latencia_p95_s = 999
                with self.assertRaises(ValueError):
                    comparacion.comparar(self.referencia, actual)

    def test_cli_compara_sin_credenciales_y_no_sobrescribe_el_resultado(self):
        with TemporaryDirectory() as carpeta:
            nuevo = Path(carpeta) / "nuevo.json"
            referencia = Path(carpeta) / "referencia.json"
            destino = Path(carpeta) / "comparacion.json"
            guardar(self.referencia, referencia)
            guardar(self.actual, nuevo)
            with patch.dict(os.environ, {}, clear=True), patch("builtins.print"):
                argumentos = [str(nuevo), "--referencia", str(referencia), "--salida", str(destino)]
                self.assertEqual(comparacion.main(argumentos), 0)
                contenido = destino.read_text(encoding="utf-8")
                self.assertEqual(json.loads(contenido)["corrida_actual"]["lote_id"], str(self.actual.lote_id))
                self.assertEqual(comparacion.main(argumentos), 1)
            self.assertEqual(destino.read_text(encoding="utf-8"), contenido)
