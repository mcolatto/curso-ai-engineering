"""Decisiones estructuradas y controles de estado con Supervisor simulado."""

import json
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from agents.supervisor import crear_supervisor, rutas_permitidas
from schemas import (
    DecisionSupervisor, ResultadoAnalisis, ResultadoHoteleria, ResultadoTurismo,
    SalidaSupervisor, SolicitudParcial, SolicitudViaje,
)
from state import crear_estado_inicial
from tools import buscar_hoteles, buscar_turismo


class SupervisorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.request = SolicitudViaje(destino="Tokio", fecha_inicio="2027-02-12", fecha_fin="2027-02-20",
                                       cantidad_personas=4, componentes=["alojamiento", "actividades"])
        self.state = crear_estado_inicial("Viaje a Tokio")
        self.llm = MagicMock()
        self.model = self.llm.with_structured_output.return_value
        self.model.ainvoke = AsyncMock()
        self.node = crear_supervisor(self.llm)

    def salida(self, route, **kwargs):
        return SalidaSupervisor(es_turistica=True, solicitud=self.request,
            decision=DecisionSupervisor(next_agent=route, instruccion="Tarea concreta", **kwargs))

    async def test_extraccion_inicial_permite_cualquiera_de_las_investigaciones(self):
        for route in ("hoteleria", "turismo"):
            self.model.ainvoke.return_value = self.salida(route)
            result = await self.node(self.state)
            self.assertEqual(result["decision_supervisor"].next_agent, route)
            self.assertIsInstance(result["solicitud_original"], SolicitudViaje)
            self.assertNotIn("messages", result)
        self.llm.with_structured_output.assert_called_once_with(SalidaSupervisor)

    async def test_solicitud_incompleta_finaliza_sin_delegar_y_enumera_faltantes(self):
        self.model.ainvoke.return_value = SalidaSupervisor(es_turistica=True,
            solicitud=SolicitudParcial(destino="Tokio", componentes=["alojamiento"]),
            decision=DecisionSupervisor(next_agent="hoteleria", instruccion="Consultar"))
        result = await self.node(self.state)
        self.assertEqual(result["decision_supervisor"].next_agent, "finalizar")
        for field in ("fecha_inicio", "fecha_fin", "cantidad_personas"):
            self.assertIn(field, result["messages"][0].content)
        self.assertIsNone(result["decision_supervisor"].evaluacion_supervisor)

    async def test_consulta_ajena_finaliza_sin_investigaciones(self):
        self.model.ainvoke.return_value = SalidaSupervisor(es_turistica=False,
            decision=DecisionSupervisor(next_agent="finalizar", instruccion="Responder"),
            respuesta_final="Puedo ayudarte con solicitudes turísticas.")
        result = await self.node(self.state)
        self.assertEqual(result["decision_supervisor"].next_agent, "finalizar")
        self.assertNotIn("resultado_hoteleria", result)
        self.assertIsInstance(result["messages"][0], AIMessage)

    def test_rutas_sin_orden_fijo_errores_y_vacios_cuentan_como_ejecutados(self):
        self.assertEqual(set(rutas_permitidas(self.state, self.request)), {"hoteleria", "turismo"})
        self.state["resultado_hoteleria"] = ResultadoHoteleria(estado_busqueda="error")
        self.assertEqual(rutas_permitidas(self.state, self.request), ["turismo"])
        self.state["resultado_turismo"] = ResultadoTurismo(estado_busqueda="completada")
        self.assertEqual(rutas_permitidas(self.state, self.request), ["analista"])
        self.state["resultado_analisis"] = ResultadoAnalisis(sintesis_analisis="Resultado")
        self.assertEqual(rutas_permitidas(self.state, self.request), ["analista", "finalizar"])
        self.state["refinamientos_analista"] = 1
        self.assertEqual(rutas_permitidas(self.state, self.request), ["finalizar"])

    async def test_rechaza_cierre_prematuro_analisis_prematuro_y_repeticiones(self):
        self.state["solicitud_original"] = self.request
        for route in ("analista", "finalizar"):
            self.model.ainvoke.return_value = self.salida(route)
            with self.assertRaises(ValueError):
                await self.node(self.state)
        self.state["resultado_hoteleria"] = ResultadoHoteleria(estado_busqueda="completada")
        self.model.ainvoke.return_value = self.salida("hoteleria")
        with self.assertRaises(ValueError):
            await self.node(self.state)

    async def test_no_reescribe_la_solicitud_ya_extraida(self):
        self.state["solicitud_original"] = self.request
        changed = self.request.model_copy(update={"destino": "Osaka"})
        self.model.ainvoke.return_value = self.salida("turismo").model_copy(update={"solicitud": changed})
        result = await self.node(self.state)
        self.assertEqual(result["solicitud_original"].destino, "Tokio")
        context = json.loads(self.model.ainvoke.await_args.args[0][1].content)
        self.assertEqual(set(context["rutas_permitidas"]), {"hoteleria", "turismo"})

    async def test_cierre_exige_evaluacion_y_respuesta(self):
        self.state["solicitud_original"] = self.request
        self.state["resultado_hoteleria"] = ResultadoHoteleria(estado_busqueda="completada")
        self.state["resultado_turismo"] = ResultadoTurismo(estado_busqueda="completada")
        self.state["resultado_analisis"] = ResultadoAnalisis(sintesis_analisis="Sin opciones")
        self.model.ainvoke.return_value = self.salida("finalizar")
        with self.assertRaises(ValueError):
            await self.node(self.state)
        output = self.salida("finalizar", evaluacion_supervisor="desaprobado")
        self.model.ainvoke.return_value = output.model_copy(update={"respuesta_final": "No hay opciones en las fuentes consultadas."})
        result = await self.node(self.state)
        self.assertEqual(result["messages"][0].name, "supervisor")
        self.assertNotIn("resultado_analisis", result)

    async def test_respuesta_al_viajero_separada_de_evaluacion_y_fuentes_internas(self):
        self.state["solicitud_original"] = self.request
        self.state["resultado_hoteleria"] = ResultadoHoteleria.model_validate(
            await buscar_hoteles.ainvoke({"destino": "Tokio"}))
        self.state["resultado_turismo"] = ResultadoTurismo.model_validate(
            await buscar_turismo.ainvoke({"destino": "Tokio"}))
        self.state["resultado_analisis"] = ResultadoAnalisis(sintesis_analisis="Evaluación de prueba")
        for evaluation in ("aprobado", "aprobado_con_advertencias"):
            with self.subTest(evaluation=evaluation):
                answer = "Estas son las opciones para tu viaje, según la información consultada. No se confirman reservas."
                self.model.ainvoke.return_value = self.salida("finalizar", evaluacion_supervisor=evaluation).model_copy(
                    update={"respuesta_final": answer})
                result = await self.node(self.state)
                self.assertEqual(result["decision_supervisor"].evaluacion_supervisor, evaluation)
                self.assertEqual(result["messages"][0].content, answer)
                messages = self.model.ainvoke.await_args.args[0]
                rules = messages[0].content
                self.assertIn("no agregues requisitos ni anticipes clasificaciones", rules)
                self.assertIn("no impiden aprobado", rules)
                self.assertIn("No incluyas la calificación interna", rules)
                self.assertIn("nombres de archivos JSON", rules)
                self.assertIn("data/hoteles.json", messages[1].content)
                self.assertIn("data/turismo.json", messages[1].content)
                self.assertNotIn(".json", result["messages"][0].content)

    async def test_una_devolucion_incrementa_contador_y_conserva_feedback(self):
        self.state["solicitud_original"] = self.request
        self.state["resultado_hoteleria"] = ResultadoHoteleria(estado_busqueda="completada")
        self.state["resultado_turismo"] = ResultadoTurismo(estado_busqueda="completada")
        self.state["resultado_analisis"] = ResultadoAnalisis(sintesis_analisis="Análisis incompleto")
        original = deepcopy(self.state)
        feedback = "Justificá la clasificación de la opción turismo-tokio-03 omitida en el análisis."
        self.model.ainvoke.return_value = SalidaSupervisor(es_turistica=True,
            decision=DecisionSupervisor(next_agent="analista", instruccion=feedback))
        update = await self.node(self.state)
        self.assertEqual(update["refinamientos_analista"], 1)
        self.assertEqual(update["decision_supervisor"].instruccion, feedback)
        self.assertNotIn("resultado_analisis", update)
        self.assertEqual(self.state, original)
        self.state.update(update)
        self.assertEqual(rutas_permitidas(self.state, self.request), ["finalizar"])
        with self.assertRaises(ValueError):
            await self.node(self.state)
        context = json.loads(self.model.ainvoke.await_args.args[0][1].content)
        self.assertEqual(context["refinamientos_analista"], 1)
        self.assertEqual(context["max_refinamientos_analista"], 1)

    async def test_primera_delegacion_al_analista_no_consume_refinamiento(self):
        self.state["solicitud_original"] = self.request
        self.state["resultado_hoteleria"] = ResultadoHoteleria(estado_busqueda="completada")
        self.state["resultado_turismo"] = ResultadoTurismo(estado_busqueda="completada")
        self.model.ainvoke.return_value = self.salida("analista")
        update = await self.node(self.state)
        self.assertNotIn("refinamientos_analista", update)
        self.assertEqual(self.state["refinamientos_analista"], 0)


if __name__ == "__main__":
    unittest.main()
