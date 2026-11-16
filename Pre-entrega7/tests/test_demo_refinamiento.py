"""La inyección se limita al primer análisis y el feedback exige evidencia concreta."""

import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

from demo_refinamiento import crear_analista_con_omision, feedback_identifica_omision
from schemas import OpcionEvaluada, ResultadoAnalisis, ResultadoHoteleria, ResultadoTurismo
from state import crear_estado_inicial
from tools import buscar_hoteles, buscar_turismo


class OmisionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = crear_estado_inicial("Solicitud válida")
        self.state["resultado_hoteleria"] = ResultadoHoteleria.model_validate(await buscar_hoteles.ainvoke({"destino": "Tokio"}))
        self.state["resultado_turismo"] = ResultadoTurismo.model_validate(await buscar_turismo.ainvoke({"destino": "Tokio"}))
        self.result = ResultadoAnalisis(sintesis_analisis="Análisis original", resultados_evaluados=[
            OpcionEvaluada(opcion_id=option.id, dominio=domain, clasificacion="compatible", justificacion="Evaluación de prueba")
            for domain, items in (("hoteleria", self.state["resultado_hoteleria"].hoteles),
                                  ("turismo", self.state["resultado_turismo"].opciones)) for option in items
        ])

    async def test_solo_omite_una_evaluacion_del_primer_resultado_sin_mutar_original(self):
        original = deepcopy(self.result)
        state_before = deepcopy(self.state)
        base = AsyncMock(return_value={"resultado_analisis": self.result})
        control = {}
        with patch("demo_refinamiento.crear_agente_analista", return_value=base):
            node = crear_analista_con_omision(MagicMock(), control)
        first = await node(self.state)
        second = await node(self.state)
        self.assertEqual(len(first["resultado_analisis"].resultados_evaluados), 4)
        self.assertEqual(len(second["resultado_analisis"].resultados_evaluados), 5)
        self.assertNotIn("turismo-tokio-03", [option.opcion_id for option in first["resultado_analisis"].resultados_evaluados])
        self.assertEqual(self.result, original)
        self.assertEqual(self.state, state_before)
        self.assertTrue(control["omision_aplicada"])
        self.assertEqual(base.await_count, 2)
        self.assertNotIn("control_prueba", self.state)

    async def test_analisis_ya_incompleto_no_se_rellena_para_simular_la_precondicion(self):
        self.result.resultados_evaluados = self.result.resultados_evaluados[:-1]
        base = AsyncMock(return_value={"resultado_analisis": self.result})
        control = {}
        with patch("demo_refinamiento.crear_agente_analista", return_value=base):
            node = crear_analista_con_omision(MagicMock(), control)
        with self.assertRaisesRegex(ValueError, "Precondición"):
            await node(self.state)
        self.assertFalse(control["omision_aplicada"])
        self.assertEqual(len(self.result.resultados_evaluados), 4)

    def test_feedback_concreto_no_requiere_id_exacto(self):
        for text in (
            "Completá la evaluación gastronómica omitida.",
            "Falta evaluar el Restaurante Jardín Sakura: incluí su clasificación y justificación.",
            "Evaluá la opción turismo-tokio-03 que omitiste.",
            "Evaluaste cuatro opciones de las cinco recuperadas; completá la evaluación restante.",
        ):
            with self.subTest(text=text):
                self.assertTrue(feedback_identifica_omision(text))

    def test_feedback_generico_o_sobre_otro_problema_no_es_satisfactorio(self):
        for text in ("Revisá el análisis.", "Completá el análisis.", "Revisá la gastronomía.",
                     "Justificá la distancia del hotel omitida.", "No hay hoteles compatibles; buscá otros.",
                     "No falta evaluar el restaurante. Revisá el análisis en general."):
            with self.subTest(text=text):
                self.assertFalse(feedback_identifica_omision(text))


if __name__ == "__main__":
    unittest.main()
