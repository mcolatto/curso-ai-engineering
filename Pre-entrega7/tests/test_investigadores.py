"""Investigadores con decisiones simuladas y consultas reales al JSON local."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from agents.hoteleria import crear_agente_hoteleria
from agents.turismo import crear_agente_turismo
from schemas import DecisionSupervisor, SolicitudParcial, SolicitudViaje
from state import crear_estado_inicial
from tools import buscar_hoteles, buscar_turismo


class InvestigadoresTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cases = [
            ("hoteleria", crear_agente_hoteleria, buscar_hoteles),
            ("turismo", crear_agente_turismo, buscar_turismo),
        ]

    def preparar(self, domain, factory, tool):
        state = crear_estado_inicial("HISTORIAL-QUE-NO-DEBE-ENVIARSE")
        state["solicitud_original"] = SolicitudViaje(
            destino="Tokio", fecha_inicio="2027-03-12", fecha_fin="2027-03-20",
            cantidad_personas=4, componentes=["alojamiento", "actividades", "gastronomia"],
        )
        state["decision_supervisor"] = DecisionSupervisor(next_agent=domain, instruccion="Consultar las fuentes")
        llm = MagicMock()
        llm.bind_tools.return_value.ainvoke = AsyncMock(return_value=AIMessage(
            content="", tool_calls=[{"name": tool.name, "args": {"destino": "Tokio"}, "id": "consulta-1"}]
        ))
        return state, llm, factory(llm)

    async def test_cada_agente_usa_solo_su_herramienta_y_conserva_los_datos(self):
        for domain, factory, tool in self.cases:
            with self.subTest(domain=domain):
                state, llm, node = self.preparar(domain, factory, tool)
                original = deepcopy(state)
                expected = await tool.ainvoke({"destino": "Tokio"})
                result = await node(state)
                self.assertEqual(set(result), {f"resultado_{domain}"})
                self.assertEqual(result[f"resultado_{domain}"].model_dump(mode="json"), expected)
                self.assertEqual(state, original)
                llm.bind_tools.assert_called_once_with([tool], tool_choice=tool.name)
                model = llm.bind_tools.return_value
                model.ainvoke.assert_awaited_once()
                messages = model.ainvoke.await_args.args[0]
                self.assertEqual(len(messages), 2)
                self.assertIn("Consultar las fuentes", messages[1].content)
                self.assertIn("2027-03-12", messages[1].content)
                self.assertNotIn("HISTORIAL-QUE-NO-DEBE-ENVIARSE", str(messages))

    async def test_destino_sin_opciones_es_completado_y_error_de_fuente_es_error(self):
        for domain, factory, tool in self.cases:
            with self.subTest(domain=domain):
                state, llm, node = self.preparar(domain, factory, tool)
                state["solicitud_original"].destino = "Osaka"
                llm.bind_tools.return_value.ainvoke.return_value = AIMessage(content="", tool_calls=[
                    {"name": tool.name, "args": {"destino": "Osaka"}, "id": "sin-opciones"}
                ])
                result = (await node(state))[f"resultado_{domain}"]
                self.assertEqual(result.estado_busqueda, "completada")
                field = "hoteles" if domain == "hoteleria" else "opciones"
                self.assertEqual(getattr(result, field), [])
                with tempfile.TemporaryDirectory() as directory, patch("tools.DATA_DIR", Path(directory)):
                    result = (await node(state))[f"resultado_{domain}"]
                self.assertEqual(result.estado_busqueda, "error")
                self.assertTrue(result.observaciones)

    async def test_solicitud_incompleta_no_invoca_modelo(self):
        for domain, factory, tool in self.cases:
            state, llm, node = self.preparar(domain, factory, tool)
            state["solicitud_original"] = SolicitudParcial(destino="Tokio")
            with self.assertRaises(ValidationError):
                await node(state)
            llm.bind_tools.return_value.ainvoke.assert_not_awaited()

    async def test_no_ejecuta_sin_delegacion_correcta(self):
        for domain, factory, tool in self.cases:
            state, llm, node = self.preparar(domain, factory, tool)
            state["decision_supervisor"] = None
            with self.assertRaises(ValueError):
                await node(state)
            state["decision_supervisor"] = DecisionSupervisor(next_agent="analista", instruccion="Analizar")
            with self.assertRaises(ValueError):
                await node(state)
            llm.bind_tools.return_value.ainvoke.assert_not_awaited()

    async def test_rechaza_llamadas_ajenas_ausentes_repetidas_o_a_otro_destino(self):
        for domain, factory, tool in self.cases:
            state, llm, node = self.preparar(domain, factory, tool)
            valid = {"name": tool.name, "args": {"destino": "Tokio"}, "id": "1"}
            for calls in ([], [{**valid, "name": "herramienta-ajena"}],
                          [valid, {**valid, "id": "2"}],
                          [{**valid, "args": {"destino": "Osaka"}}],
                          [{**valid, "args": {}}]):
                with self.subTest(domain=domain, calls=calls):
                    llm.bind_tools.return_value.ainvoke.return_value = AIMessage(content="", tool_calls=calls)
                    with self.assertRaises(ValueError):
                        await node(state)

    async def test_error_del_modelo_se_propaga_sin_fingir_busqueda_vacia(self):
        for domain, factory, tool in self.cases:
            state, llm, node = self.preparar(domain, factory, tool)
            llm.bind_tools.return_value.ainvoke.side_effect = RuntimeError("fallo simulado")
            with self.assertRaises(RuntimeError):
                await node(state)


if __name__ == "__main__":
    unittest.main()
