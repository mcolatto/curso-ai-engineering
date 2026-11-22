"""Actualizaciones parciales sobre el estado real de LangGraph, sin LLM."""

import unittest

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from schemas import ResultadoHoteleria, ResultadoTurismo
from state import EstadoViaje, crear_estado_inicial


class StateTests(unittest.IsolatedAsyncioTestCase):
    def test_estado_inicial_y_texto_original(self):
        state = crear_estado_inicial("  Viaje a Tokio  ")
        self.assertEqual(state["messages"][0].content, "  Viaje a Tokio  ")
        for field in ("solicitud_original", "resultado_hoteleria", "resultado_turismo",
                      "resultado_analisis", "decision_supervisor"):
            self.assertIsNone(state[field])
        self.assertNotIn("tarea_finalizada", state)
        self.assertEqual(state["refinamientos_analista"], 0)

    def test_rechaza_entrada_vacia(self):
        for value in ("", " \n\t", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                crear_estado_inicial(value)

    async def test_actualizaciones_parciales_conservan_resultados_y_acumulan_mensajes(self):
        hotels = ResultadoHoteleria(estado_busqueda="completada")
        tourism = ResultadoTurismo(estado_busqueda="completada")

        async def hoteleria(state):
            return {"resultado_hoteleria": hotels, "messages": [AIMessage(content="Hotelería completada")]}

        async def turismo(state):
            self.assertEqual(state["resultado_hoteleria"], hotels)
            return {"resultado_turismo": tourism, "messages": [AIMessage(content="Turismo completado")]}

        graph = StateGraph(EstadoViaje)
        graph.add_node("hoteleria", hoteleria)
        graph.add_node("turismo", turismo)
        graph.add_edge(START, "hoteleria")
        graph.add_edge("hoteleria", "turismo")
        graph.add_edge("turismo", END)
        result = await graph.compile().ainvoke(crear_estado_inicial("Viaje a Tokio"))
        self.assertEqual(result["resultado_hoteleria"], hotels)
        self.assertEqual(result["resultado_turismo"], tourism)
        self.assertIsNone(result["resultado_analisis"])
        self.assertEqual([message.content for message in result["messages"]], [
            "Viaje a Tokio", "Hotelería completada", "Turismo completado",
        ])

    def test_estados_independientes_no_comparten_listas(self):
        first = crear_estado_inicial("Primer viaje")
        second = crear_estado_inicial("Segundo viaje")
        first["messages"].append(AIMessage(content="Resultado"))
        self.assertEqual(len(second["messages"]), 1)


if __name__ == "__main__":
    unittest.main()
