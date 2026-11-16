"""Evidencia calculada y análisis estructurado con LLM simulado."""

import json
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

from schemas import (
    Coordenadas, DecisionSupervisor, OpcionEvaluada, OpcionHotel, OpcionTuristica,
    Periodo, ResultadoAnalisis, ResultadoHoteleria, ResultadoTurismo, SolicitudViaje,
)
from agents.analista import INSTRUCCIONES, comprobar_opcion, crear_agente_analista
from state import crear_estado_inicial
from tools import TOKYO_STATION, buscar_hoteles, buscar_turismo, calcular_distancia_km


class AnalistaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.request = SolicitudViaje(
            destino="Tokio", fecha_inicio="2027-02-12", fecha_fin="2027-02-20",
            cantidad_personas=4, componentes=["alojamiento", "actividades", "gastronomia"],
        )
        self.hotel = OpcionHotel(
            id="hotel-1", nombre="Hotel ficticio", fuente="fuente ficticia",
            capacidad=4, disponibilidad="disponible_en_fuente",
            periodos_disponibles=[Periodo(inicio="2027-02-01", fin="2027-02-28")],
            coordenadas=TOKYO_STATION,
        )

    def test_distancia_cero_simetria_y_valor_conocido(self):
        self.assertEqual(calcular_distancia_km(TOKYO_STATION, TOKYO_STATION), 0)
        a, b = Coordenadas(latitud=0, longitud=0), Coordenadas(latitud=0, longitud=1)
        self.assertAlmostEqual(calcular_distancia_km(a, b), 111.195, places=3)
        self.assertEqual(calcular_distancia_km(a, b), calcular_distancia_km(b, a))

    def test_capacidad_y_cobertura_temporal_conocidas(self):
        facts = comprobar_opcion(self.request, self.hotel)
        self.assertTrue(facts["capacidad_suficiente"])
        self.assertTrue(facts["compatibilidad_temporal"])
        self.hotel.capacidad = 2
        self.assertFalse(comprobar_opcion(self.request, self.hotel)["capacidad_suficiente"])
        march = self.request.model_copy(update={
            "fecha_inicio": Periodo(inicio="2027-03-12", fin="2027-03-20").inicio,
            "fecha_fin": Periodo(inicio="2027-03-12", fin="2027-03-20").fin,
        })
        self.assertFalse(comprobar_opcion(march, self.hotel)["compatibilidad_temporal"])

    def test_datos_ausentes_son_desconocidos_y_no_falsos(self):
        hotel = OpcionHotel(id="h", nombre="Hotel", fuente="ficticia", disponibilidad="sin_datos")
        self.assertEqual(comprobar_opcion(self.request, hotel), {
            "compatibilidad_temporal": None, "capacidad_suficiente": None, "distancia_km": None,
        })
        hotel.disponibilidad = "no_disponible"
        self.assertFalse(comprobar_opcion(self.request, hotel)["compatibilidad_temporal"])
        self.assertIn("pendiente_verificacion, no incompatible", INSTRUCCIONES)

    def test_cobertura_de_hotel_admite_periodos_contiguos_pero_no_huecos(self):
        self.hotel.periodos_disponibles = [
            Periodo(inicio="2027-02-12", fin="2027-02-15"),
            Periodo(inicio="2027-02-16", fin="2027-02-20"),
        ]
        self.assertTrue(comprobar_opcion(self.request, self.hotel)["compatibilidad_temporal"])
        self.hotel.periodos_disponibles[1] = Periodo(inicio="2027-02-17", fin="2027-02-20")
        self.assertFalse(comprobar_opcion(self.request, self.hotel)["compatibilidad_temporal"])

    def test_actividad_solo_necesita_coincidencia_con_algun_momento_del_viaje(self):
        option = OpcionTuristica(id="a", nombre="Actividad", fuente="ficticia", categoria="attraction",
                                periodos_disponibles=[Periodo(inicio="2027-02-15", fin="2027-02-15")])
        self.assertTrue(comprobar_opcion(self.request, option)["compatibilidad_temporal"])
        option.periodos_disponibles = []
        self.assertFalse(comprobar_opcion(self.request, option)["compatibilidad_temporal"])
        option.periodos_disponibles = None
        self.assertIsNone(comprobar_opcion(self.request, option)["compatibilidad_temporal"])

    async def preparar_estado(self):
        state = crear_estado_inicial("HISTORIAL-NO-ENVIAR")
        state["solicitud_original"] = self.request
        state["decision_supervisor"] = DecisionSupervisor(next_agent="analista", instruccion="Evaluá las opciones")
        state["resultado_hoteleria"] = ResultadoHoteleria.model_validate(await buscar_hoteles.ainvoke({"destino": "Tokio"}))
        state["resultado_turismo"] = ResultadoTurismo.model_validate(await buscar_turismo.ainvoke({"destino": "Tokio"}))
        return state

    async def test_agente_recibe_evidencia_y_preserva_investigaciones(self):
        state = await self.preparar_estado()
        original = deepcopy(state)
        option = state["resultado_hoteleria"].hoteles[0]
        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=ResultadoAnalisis(
            sintesis_analisis="Evaluación simulada",
            resultados_evaluados=[OpcionEvaluada(
                opcion_id=option.id, dominio="hoteleria", clasificacion="compatible",
                justificacion="Capacidad y fechas suficientes", distancia_km=999,
            )],
        ))
        result = await crear_agente_analista(llm)(state)
        self.assertEqual(set(result), {"resultado_analisis"})
        self.assertEqual(state, original)
        llm.with_structured_output.assert_called_once_with(ResultadoAnalisis)
        llm.bind_tools.assert_not_called()
        messages = llm.with_structured_output.return_value.ainvoke.await_args.args[0]
        payload = json.loads(messages[1].content)
        self.assertEqual(payload["tarea"], "Evaluá las opciones")
        self.assertNotIn("HISTORIAL-NO-ENVIAR", str(messages))
        self.assertEqual(len(payload["comprobaciones"]), 5)
        restaurant = next(item for item in payload["investigaciones"]["turismo"]["opciones"]
                          if item["categoria"] == "gastronomy")
        self.assertEqual(restaurant["observaciones"][0]["tipo"], "incertidumbre")
        distance = result["resultado_analisis"].resultados_evaluados[0].distancia_km
        self.assertEqual(distance, calcular_distancia_km(TOKYO_STATION, option.coordenadas))
        # Una omisión sigue siendo visible para la evaluación de suficiencia del Supervisor.
        self.assertEqual(len(result["resultado_analisis"].resultados_evaluados), 1)

    async def test_rechaza_opciones_inventadas_o_duplicadas(self):
        state = await self.preparar_estado()
        known = state["resultado_hoteleria"].hoteles[0].id
        for ids in (["inventado"], [known, known]):
            llm = MagicMock()
            llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=ResultadoAnalisis(
                sintesis_analisis="Resultado",
                resultados_evaluados=[OpcionEvaluada(opcion_id=value, dominio="hoteleria",
                    clasificacion="compatible", justificacion="Evaluación simulada") for value in ids],
            ))
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                await crear_agente_analista(llm)(state)

    async def test_criterios_de_recomendacion_general_se_envian_en_ambos_escenarios(self):
        # Verifica el contrato enviado y su conservación, no la semántica de un LLM real.
        for month in (2, 3):
            with self.subTest(month=month):
                state = await self.preparar_estado()
                state["solicitud_original"] = SolicitudViaje(
                    destino="Tokio", fecha_inicio=f"2027-{month:02}-12", fecha_fin=f"2027-{month:02}-20",
                    cantidad_personas=4, componentes=["alojamiento", "actividades", "gastronomia"],
                )
                llm = MagicMock()
                llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=ResultadoAnalisis(
                    sintesis_analisis="Opciones turísticas compatibles como recomendación general",
                    resultados_evaluados=[OpcionEvaluada(opcion_id=option.id, dominio="turismo",
                        clasificacion="compatible", justificacion="Período compatible sin restricciones conocidas")
                        for option in state["resultado_turismo"].opciones],
                ))
                result = await crear_agente_analista(llm)(state)
                messages = llm.with_structured_output.return_value.ainvoke.await_args.args[0]
                rules = messages[0].content
                self.assertIn("corresponde sólo al alojamiento", rules)
                self.assertIn("no cualquier campo ausente", rules)
                self.assertIn("no justifican incompatible ni pendiente_verificacion", rules)
                self.assertIn("observaciones informativas sin confirmar reservas", rules)
                payload = json.loads(messages[1].content)
                tourism = [item for item in payload["comprobaciones"] if item["dominio"] == "turismo"]
                self.assertTrue(all(item["compatibilidad_temporal"] is True for item in tourism))
                self.assertTrue(all(item["capacidad_suficiente"] is None for item in tourism))
                self.assertTrue(all(item.clasificacion == "compatible"
                                    for item in result["resultado_analisis"].resultados_evaluados))

    async def test_busquedas_fallidas_y_no_ejecutadas_se_conservan_en_el_contexto(self):
        state = await self.preparar_estado()
        state["resultado_hoteleria"] = ResultadoHoteleria(estado_busqueda="error")
        state["resultado_turismo"] = None
        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=ResultadoAnalisis(
            sintesis_analisis="No hay evidencia suficiente"
        ))
        await crear_agente_analista(llm)(state)
        payload = json.loads(llm.with_structured_output.return_value.ainvoke.await_args.args[0][1].content)
        self.assertEqual(payload["investigaciones"]["hoteleria"]["estado_busqueda"], "error")
        self.assertIsNone(payload["investigaciones"]["turismo"])
        self.assertEqual(payload["comprobaciones"], [])

    async def test_sin_delegacion_no_invoca_y_error_del_modelo_se_propaga(self):
        state = await self.preparar_estado()
        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("fallo simulado"))
        node = crear_agente_analista(llm)
        state["decision_supervisor"] = None
        with self.assertRaises(ValueError):
            await node(state)
        llm.with_structured_output.return_value.ainvoke.assert_not_awaited()
        state["decision_supervisor"] = DecisionSupervisor(next_agent="analista", instruccion="Analizar")
        with self.assertRaises(RuntimeError):
            await node(state)


if __name__ == "__main__":
    unittest.main()
