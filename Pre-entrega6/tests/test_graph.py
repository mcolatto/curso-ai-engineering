"""Grafo completo y consultas locales reales con los modelos simulados."""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from graph import crear_grafo
from schemas import (
    DecisionSupervisor, OpcionEvaluada, ResultadoAnalisis, SalidaSupervisor, SolicitudViaje,
)
from state import crear_estado_inicial
from tools import leer_catalogo


class GraphTests(unittest.IsolatedAsyncioTestCase):
    def preparar(self, month=2, components=None, first="hoteleria"):
        request = SolicitudViaje(destino="Tokio", fecha_inicio=f"2027-{month:02}-12", fecha_fin=f"2027-{month:02}-20",
                                 cantidad_personas=4, componentes=components or ["alojamiento", "actividades", "gastronomia"])
        llm = MagicMock()
        supervisor = MagicMock()
        analyst = MagicMock()
        llm.with_structured_output.side_effect = lambda schema: supervisor if schema is SalidaSupervisor else analyst
        routes = (["hoteleria", "turismo"] if first == "hoteleria" else ["turismo", "hoteleria"])
        if "alojamiento" not in request.componentes:
            routes = ["turismo"]
        elif request.componentes == ["alojamiento"]:
            routes = ["hoteleria"]
        outputs = [SalidaSupervisor(es_turistica=True, solicitud=request,
            decision=DecisionSupervisor(next_agent=route, instruccion=f"Ejecutar {route}")) for route in routes + ["analista"]]
        outputs.append(SalidaSupervisor(es_turistica=True,
            decision=DecisionSupervisor(next_agent="finalizar", instruccion="Responder",
                evaluacion_supervisor="aprobado" if month == 2 else "aprobado_con_advertencias"),
            respuesta_final="Recomendaciones simuladas." if month == 2 else "Hay turismo, pero no hoteles compatibles en las fuentes consultadas."))
        supervisor.ainvoke = AsyncMock(side_effect=outputs)

        def bind(tools, **kwargs):
            model = MagicMock()
            model.ainvoke = AsyncMock(return_value=AIMessage(content="", tool_calls=[
                {"name": tools[0].name, "args": {"destino": "Tokio"}, "id": "consulta-local"}
            ]))
            return model

        llm.bind_tools.side_effect = bind

        async def analyze(messages):
            # El stub no prueba semántica del LLM: permite verificar el recorrido de la evidencia.
            payload = json.loads(messages[1].content)
            return ResultadoAnalisis(sintesis_analisis="Síntesis simulada", resultados_evaluados=[
                OpcionEvaluada(opcion_id=item["opcion_id"], dominio=item["dominio"],
                    clasificacion="incompatible" if item["compatibilidad_temporal"] is False else "compatible",
                    justificacion="Clasificación simulada a partir de evidencia recibida")
                for item in payload["comprobaciones"]
            ])

        analyst.ainvoke = AsyncMock(side_effect=analyze)
        return llm, supervisor, analyst

    async def test_flujo_completo_y_parcial_en_ambos_ordenes_de_investigacion(self):
        for month, first in ((2, "hoteleria"), (3, "turismo")):
            with self.subTest(month=month, first=first):
                llm, supervisor, analyst = self.preparar(month, first=first)
                result = await crear_grafo(llm).compile().ainvoke(
                    crear_estado_inicial("Viaje a Tokio"), config={"recursion_limit": 10}
                )
                self.assertEqual(result["decision_supervisor"].next_agent, "finalizar")
                self.assertEqual(result["decision_supervisor"].evaluacion_supervisor,
                                 "aprobado" if month == 2 else "aprobado_con_advertencias")
                self.assertEqual(len(result["resultado_hoteleria"].hoteles), 2)
                self.assertEqual(len(result["resultado_turismo"].opciones), 3)
                self.assertEqual(len(result["resultado_analisis"].resultados_evaluados), 5)
                self.assertEqual(len(result["messages"]), 2)
                self.assertEqual(result["messages"][-1].name, "supervisor")
                self.assertEqual(supervisor.ainvoke.await_count, 4)
                analyst.ainvoke.assert_awaited_once()
                final_context = json.loads(supervisor.ainvoke.await_args.args[0][1].content)
                self.assertIsNotNone(final_context["resultado_analisis"])
                self.assertEqual(final_context["rutas_permitidas"], ["analista", "finalizar"])
                self.assertEqual(result["refinamientos_analista"], 0)

    async def test_un_solo_dominio_no_ejecuta_el_otro(self):
        for components, absent in ((["alojamiento"], "resultado_turismo"), (["gastronomia"], "resultado_hoteleria")):
            llm, supervisor, analyst = self.preparar(components=components)
            result = await crear_grafo(llm).compile().ainvoke(crear_estado_inicial("Solicitud"), config={"recursion_limit": 10})
            self.assertIsNone(result[absent])
            self.assertEqual(supervisor.ainvoke.await_count, 3)
            analyst.ainvoke.assert_awaited_once()

    async def test_limite_de_recursion_no_genera_un_cierre_falso(self):
        llm, supervisor, analyst = self.preparar()
        with self.assertRaises(GraphRecursionError):
            await crear_grafo(llm).compile().ainvoke(crear_estado_inicial("Solicitud"), config={"recursion_limit": 1})
        analyst.ainvoke.assert_not_awaited()

    async def test_solicitud_incompleta_no_ejecuta_especialistas(self):
        llm, supervisor, analyst = self.preparar()
        supervisor.ainvoke.side_effect = None
        supervisor.ainvoke.return_value = SalidaSupervisor(es_turistica=True,
            decision=DecisionSupervisor(next_agent="finalizar", instruccion="Informar faltantes"))
        result = await crear_grafo(llm).compile().ainvoke(crear_estado_inicial("Quiero viajar"), config={"recursion_limit": 10})
        self.assertIsNone(result["resultado_hoteleria"])
        self.assertIsNone(result["resultado_turismo"])
        analyst.ainvoke.assert_not_awaited()
        for call in llm.bind_tools.mock_calls:
            self.assertNotIn("ainvoke", str(call))
        self.assertIn("cantidad_personas", result["messages"][-1].content)

    async def test_refinamiento_unico_conserva_ambas_versiones_sin_reinvestigar(self):
        for corrected in (True, False):
            with self.subTest(corrected=corrected):
                llm, supervisor, analyst = self.preparar()
                responses = list(supervisor.ainvoke.side_effect)
                closing = responses.pop()
                feedback = "Evaluá la opción turismo-tokio-03 que omitiste, conservando las otras evaluaciones."
                responses.append(SalidaSupervisor(es_turistica=True,
                    decision=DecisionSupervisor(next_agent="analista", instruccion=feedback)))
                if not corrected:
                    closing = closing.model_copy(update={
                        "decision": DecisionSupervisor(next_agent="finalizar", instruccion="Informar la omisión pendiente",
                            evaluacion_supervisor="aprobado_con_advertencias"),
                        "respuesta_final": "Hay opciones útiles, pero una opción quedó sin evaluar.",
                    })
                responses.append(closing)
                supervisor.ainvoke.side_effect = responses
                base_analyze = analyst.ainvoke.side_effect
                calls = 0

                async def analyze(messages):
                    nonlocal calls
                    calls += 1
                    result = await base_analyze(messages)
                    if calls == 1 or not corrected:
                        result.resultados_evaluados = result.resultados_evaluados[:-1]
                    return result

                analyst.ainvoke.side_effect = analyze
                events = []
                with patch("tools.leer_catalogo", wraps=leer_catalogo) as read:
                    app = crear_grafo(llm).compile()
                    async for event in app.astream(crear_estado_inicial("Viaje a Tokio"),
                                                  config={"recursion_limit": 10}, stream_mode="updates"):
                        events.append(event)
                    self.assertEqual(read.call_count, 2)
                self.assertEqual([next(iter(event)) for event in events], [
                    "supervisor", "hoteleria", "supervisor", "turismo", "supervisor",
                    "analista", "supervisor", "analista", "supervisor",
                ])
                first = events[5]["analista"]["resultado_analisis"]
                second = events[7]["analista"]["resultado_analisis"]
                self.assertEqual(len(first.resultados_evaluados), 4)
                self.assertEqual(len(second.resultados_evaluados), 5 if corrected else 4)
                self.assertEqual(events[6]["supervisor"]["refinamientos_analista"], 1)
                first_payload = json.loads(analyst.ainvoke.await_args_list[0].args[0][1].content)
                second_payload = json.loads(analyst.ainvoke.await_args_list[1].args[0][1].content)
                self.assertIsNone(first_payload["analisis_anterior"])
                self.assertEqual(second_payload["analisis_anterior"], first.model_dump(mode="json"))
                self.assertEqual(second_payload["tarea"], feedback)
                self.assertEqual(first_payload["investigaciones"], second_payload["investigaciones"])
                final_context = json.loads(supervisor.ainvoke.await_args.args[0][1].content)
                self.assertEqual(final_context["rutas_permitidas"], ["finalizar"])
                self.assertEqual(final_context["refinamientos_analista"], 1)
                self.assertEqual(final_context["resultado_analisis"], second.model_dump(mode="json"))
                self.assertEqual(analyst.ainvoke.await_count, 2)
                self.assertEqual(events[-1]["supervisor"]["decision_supervisor"].evaluacion_supervisor,
                                 "aprobado" if corrected else "aprobado_con_advertencias")


if __name__ == "__main__":
    unittest.main()
