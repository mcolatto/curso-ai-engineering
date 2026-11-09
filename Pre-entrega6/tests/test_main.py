"""Demostración con grafo, herramientas y SQLite reales, y modelos simulados."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from openai import LengthFinishReasonError
from openai.types.chat import ChatCompletion

from config import Configuracion
from graph import crear_grafo
from main import NOMBRE_PRUEBA, TIPOS_CHECKPOINT, ejecutar_demo, guardar_traza, main, serializar
from schemas import DecisionSupervisor, OpcionEvaluada, ResultadoAnalisis, SalidaSupervisor, SolicitudViaje


class MainTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / "checkpoints.sqlite"
        self.trace_path = Path(directory.name) / "traza_ejecucion.json"
        for target, value in (("main.CHECKPOINT_PATH", self.database), ("main.TRACE_PATH", self.trace_path)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.config = Configuracion(openai_api_key="clave-ficticia")
        self.llm = MagicMock()
        self.supervisor = MagicMock()
        self.analyst = MagicMock()
        self.llm.with_structured_output.side_effect = lambda schema: self.supervisor if schema is SalidaSupervisor else self.analyst
        self.preparar_respuestas()

        def bind(tools, **kwargs):
            model = MagicMock()
            model.ainvoke = AsyncMock(return_value=AIMessage(content="", tool_calls=[
                {"name": tools[0].name, "args": {"destino": "Tokio"}, "id": "consulta-local"}
            ]))
            return model

        self.llm.bind_tools.side_effect = bind

        async def analyze(messages):
            payload = json.loads(messages[1].content)
            return ResultadoAnalisis(sintesis_analisis="Análisis simulado", resultados_evaluados=[
                OpcionEvaluada(opcion_id=item["opcion_id"], dominio=item["dominio"],
                    clasificacion="incompatible" if item["compatibilidad_temporal"] is False else "compatible",
                    justificacion="Evaluación simulada para comprobar integración")
                for item in payload["comprobaciones"]
            ])

        self.analyst.ainvoke = AsyncMock(side_effect=analyze)

    def preparar_respuestas(self, incluir_prueba=False, feedback="Completá la evaluación gastronómica omitida"):
        responses = []
        for index, month in enumerate((2, 3, 2) if incluir_prueba else (2, 3)):
            request = SolicitudViaje(destino="Tokio", fecha_inicio=f"2027-{month:02}-12",
                fecha_fin=f"2027-{month:02}-20", cantidad_personas=4,
                componentes=["alojamiento", "actividades", "gastronomia"])
            for route in ("hoteleria", "turismo", "analista"):
                responses.append(SalidaSupervisor(es_turistica=True, solicitud=request,
                    decision=DecisionSupervisor(next_agent=route, instruccion=f"Ejecutar {route}")))
            if index == 2:
                responses.append(SalidaSupervisor(es_turistica=True,
                    decision=DecisionSupervisor(next_agent="analista", instruccion=feedback)))
            responses.append(SalidaSupervisor(es_turistica=True,
                decision=DecisionSupervisor(next_agent="finalizar", instruccion="Responder",
                    evaluacion_supervisor="aprobado" if month == 2 else "aprobado_con_advertencias"),
                respuesta_final="Opciones disponibles en febrero." if month == 2 else "Turismo disponible; sin hoteles compatibles en estas fuentes."))
        self.supervisor.ainvoke = AsyncMock(side_effect=responses)

    async def test_demo_guarda_transiciones_resultados_y_threads_separados(self):
        with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()) as output:
            trace = await ejecutar_demo(self.config)
        self.assertEqual(list(trace), ["respuesta_completa", "respuesta_parcial"])
        threads = [item["thread_id"] for item in trace.values()]
        self.assertEqual(len(set(threads)), 2)
        for name, scenario in trace.items():
            self.assertEqual([event["nodo"] for event in scenario["eventos"]],
                ["supervisor", "hoteleria", "supervisor", "turismo", "supervisor", "analista", "supervisor"])
            state = scenario["estado_final"]
            self.assertEqual(state["decision_supervisor"]["next_agent"], "finalizar")
            self.assertEqual(len(state["resultado_analisis"]["resultados_evaluados"]), 5)
            self.assertEqual(len(state["messages"]), 2)
            self.assertEqual(state["messages"][0]["contenido"], scenario["solicitud_usuario"])
        self.assertEqual(trace["respuesta_parcial"]["estado_final"]["decision_supervisor"]["evaluacion_supervisor"],
                         "aprobado_con_advertencias")
        self.assertIn("Evaluación: aprobado", output.getvalue())
        self.assertIn("Distancia:", output.getvalue())
        # Reabrir la base y recuperar ambos estados, sin solicitudes nuevas.
        async with aiosqlite.connect(str(self.database)) as connection:
            saver = AsyncSqliteSaver(connection, serde=JsonPlusSerializer(allowed_msgpack_modules=TIPOS_CHECKPOINT))
            app = crear_grafo(self.llm).compile(checkpointer=saver)
            for name, scenario in trace.items():
                snapshot = await app.aget_state({"configurable": {"thread_id": scenario["thread_id"]}})
                self.assertFalse(snapshot.next)
                self.assertEqual(snapshot.values["solicitud_original"].fecha_inicio.month,
                                 2 if name == "respuesta_completa" else 3)
        self.assertEqual(self.supervisor.ainvoke.await_count, 8)

    async def test_repetir_demo_no_reutiliza_threads_ni_historial(self):
        with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()):
            first = await ejecutar_demo(self.config)
            self.preparar_respuestas()
            second = await ejecutar_demo(self.config)
        self.assertTrue({item["thread_id"] for item in first.values()}.isdisjoint(
            {item["thread_id"] for item in second.values()}))
        self.assertTrue(all(len(item["estado_final"]["messages"]) == 2 for item in second.values()))

    async def test_refinamiento_se_persiste_y_ambas_versiones_quedan_en_la_traza(self):
        responses = list(self.supervisor.ainvoke.side_effect)
        responses.insert(3, SalidaSupervisor(es_turistica=True,
            decision=DecisionSupervisor(next_agent="analista", instruccion="Evaluá la opción turismo-tokio-03 omitida")))
        self.supervisor.ainvoke.side_effect = responses
        base_analyze = self.analyst.ainvoke.side_effect
        count = 0

        async def analyze(messages):
            nonlocal count
            count += 1
            result = await base_analyze(messages)
            if count == 1:
                result.resultados_evaluados = result.resultados_evaluados[:-1]
            return result

        self.analyst.ainvoke.side_effect = analyze
        with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()):
            trace = await ejecutar_demo(self.config)
        first = trace["respuesta_completa"]
        analyses = [event["actualizacion"]["resultado_analisis"] for event in first["eventos"]
                    if event["nodo"] == "analista"]
        self.assertEqual([len(item["resultados_evaluados"]) for item in analyses], [4, 5])
        self.assertEqual(first["estado_final"]["refinamientos_analista"], 1)
        self.assertEqual(trace["respuesta_parcial"]["estado_final"]["refinamientos_analista"], 0)
        async with aiosqlite.connect(str(self.database)) as connection:
            saver = AsyncSqliteSaver(connection, serde=JsonPlusSerializer(allowed_msgpack_modules=TIPOS_CHECKPOINT))
            app = crear_grafo(self.llm).compile(checkpointer=saver)
            snapshot = await app.aget_state({"configurable": {"thread_id": first["thread_id"]}})
            self.assertEqual(snapshot.values["refinamientos_analista"], 1)
            self.assertEqual(len(snapshot.values["resultado_analisis"].resultados_evaluados), 5)
            self.assertFalse(snapshot.next)

    async def test_main_publica_json_valido_despues_de_ambos_escenarios(self):
        self.preparar_respuestas(incluir_prueba=True)
        with patch("main.load_dotenv"), patch("main.cargar_configuracion", return_value=self.config):
            with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(await main(), 0)
        trace = json.loads(self.trace_path.read_text(encoding="utf-8"))
        self.assertEqual(len(trace), 3)
        self.assertTrue(trace[NOMBRE_PRUEBA]["resultado_prueba"]["satisfactoria"])
        self.assertNotIn("clave-ficticia", self.trace_path.read_text())
        self.assertIn("Demostración finalizada", output.getvalue())
        self.assertFalse(self.trace_path.with_suffix(".json.tmp").exists())

    async def test_feedback_generico_falla_aunque_el_analisis_se_complete_y_guarda_traza(self):
        self.preparar_respuestas(incluir_prueba=True, feedback="Revisá el análisis y mejoralo")
        with patch("main.load_dotenv"), patch("main.cargar_configuracion", return_value=self.config):
            with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(await main(), 1)
        trace = json.loads(self.trace_path.read_text())
        case = trace[NOMBRE_PRUEBA]
        self.assertFalse(case["resultado_prueba"]["satisfactoria"])
        self.assertEqual(case["resultado_prueba"]["feedback_observado"], "Revisá el análisis y mejoralo")
        self.assertTrue(any("feedback" in reason for reason in case["resultado_prueba"]["motivos"]))
        self.assertEqual(len(case["estado_final"]["resultado_analisis"]["resultados_evaluados"]), 5)
        self.assertIn("NO satisfactoria", output.getvalue())
        self.assertNotIn("Demostración finalizada", output.getvalue())

    async def test_supervisor_que_no_detecta_omision_no_es_corregido_artificialmente(self):
        self.preparar_respuestas(incluir_prueba=True)
        responses = list(self.supervisor.ainvoke.side_effect)
        del responses[11]  # El modelo simulado decide cerrar en lugar de devolver.
        self.supervisor.ainvoke.side_effect = responses
        with patch("main.load_dotenv"), patch("main.cargar_configuracion", return_value=self.config):
            with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()):
                self.assertEqual(await main(), 1)
        case = json.loads(self.trace_path.read_text())[NOMBRE_PRUEBA]
        self.assertFalse(case["resultado_prueba"]["satisfactoria"])
        self.assertIsNone(case["resultado_prueba"]["feedback_observado"])
        self.assertEqual(sum(event["nodo"] == "analista" for event in case["eventos"]), 1)
        self.assertEqual(len(case["estado_final"]["resultado_analisis"]["resultados_evaluados"]), 4)

    async def test_precondicion_fallida_se_registra_sin_completar_resultados(self):
        self.preparar_respuestas(incluir_prueba=True)
        base_analyze = self.analyst.ainvoke.side_effect
        count = 0

        async def analyze(messages):
            nonlocal count
            count += 1
            result = await base_analyze(messages)
            if count == 3:
                result.resultados_evaluados = result.resultados_evaluados[:-1]
            return result

        self.analyst.ainvoke.side_effect = analyze
        with patch("main.load_dotenv"), patch("main.cargar_configuracion", return_value=self.config):
            with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()):
                self.assertEqual(await main(), 1)
        case = json.loads(self.trace_path.read_text())[NOMBRE_PRUEBA]
        self.assertFalse(case["control_prueba"]["omision_aplicada"])
        self.assertEqual(case["control_prueba"]["cantidad_antes"], 4)
        self.assertEqual(case["error_ejecucion"], "ValueError")
        self.assertFalse(case["resultado_prueba"]["satisfactoria"])

    async def test_error_operativo_controlado_conserva_eventos_y_no_expone_detalles(self):
        self.preparar_respuestas(incluir_prueba=True)
        responses = list(self.supervisor.ainvoke.side_effect)
        self.supervisor.ainvoke.side_effect = responses[:11] + [RuntimeError("secreto-ficticio")]
        with patch("main.load_dotenv"), patch("main.cargar_configuracion", return_value=self.config):
            with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()):
                self.assertEqual(await main(), 1)
        text = self.trace_path.read_text()
        case = json.loads(text)[NOMBRE_PRUEBA]
        self.assertEqual(case["error_ejecucion"], "RuntimeError")
        self.assertTrue(case["control_prueba"]["omision_aplicada"])
        self.assertEqual(len(case["eventos"][-1]["actualizacion"]["resultado_analisis"]["resultados_evaluados"]), 4)
        self.assertNotIn("secreto-ficticio", text)

    async def test_prueba_satisfactoria_aislada_y_sin_modificar_fuentes_o_supervisor(self):
        self.preparar_respuestas(incluir_prueba=True)
        root = Path(__file__).resolve().parent.parent
        protected = [root / "agents/supervisor.py", root / "data/hoteles.json", root / "data/turismo.json"]
        before = {path: path.read_bytes() for path in protected}
        with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()):
            trace = await ejecutar_demo(self.config, incluir_prueba_refinamiento=True)
        case = trace[NOMBRE_PRUEBA]
        self.assertTrue(case["resultado_prueba"]["satisfactoria"])
        self.assertEqual(len({item["thread_id"] for item in trace.values()}), 3)
        self.assertEqual([event["nodo"] for event in case["eventos"]], [
            "supervisor", "hoteleria", "supervisor", "turismo", "supervisor",
            "analista", "supervisor", "analista", "supervisor",
        ])
        for name in ("respuesta_completa", "respuesta_parcial"):
            self.assertNotIn("control_prueba", trace[name])
            self.assertEqual(len(trace[name]["estado_final"]["resultado_analisis"]["resultados_evaluados"]), 5)
        for call in self.supervisor.ainvoke.await_args_list:
            self.assertNotIn("omision_aplicada", str(call.args[0]))
            self.assertNotIn("control_prueba", str(call.args[0]))
        self.assertEqual({path: path.read_bytes() for path in protected}, before)

    async def test_fallo_del_segundo_escenario_conserva_evidencia_anterior(self):
        previous = '{"evidencia": "anterior"}'
        self.trace_path.write_text(previous)
        original = self.supervisor.ainvoke.side_effect
        # Consumir únicamente las cuatro respuestas del primer escenario.
        self.supervisor.ainvoke.side_effect = [next(original) for _ in range(4)] + [RuntimeError("secreto-ficticio")]
        with patch("main.load_dotenv"), patch("main.cargar_configuracion", return_value=self.config):
            with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(await main(), 1)
        self.assertEqual(self.trace_path.read_text(), previous)
        self.assertIn("Demostración incompleta", output.getvalue())
        self.assertNotIn("secreto-ficticio", output.getvalue())
        self.assertNotIn("Demostración finalizada", output.getvalue())

    async def test_limite_configurado_interrumpe_y_cierra_sqlite(self):
        config = Configuracion(openai_api_key="clave-ficticia", recursion_limit=1)
        with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()):
            with self.assertRaises(GraphRecursionError):
                await ejecutar_demo(config)
        async with AsyncSqliteSaver.from_conn_string(str(self.database)) as saver:
            await saver.setup()
        self.assertFalse(self.trace_path.exists())

    async def test_configuracion_invalida_no_inicia_demo(self):
        with patch("main.load_dotenv"), patch("main.cargar_configuracion", side_effect=ValueError("dato-sensible")):
            with patch("main.ejecutar_demo", new_callable=AsyncMock) as demo, redirect_stdout(io.StringIO()) as output:
                self.assertEqual(await main(), 1)
                demo.assert_not_awaited()
        self.assertNotIn("dato-sensible", output.getvalue())

    async def test_salida_truncada_informa_causa_sin_reintentar_ni_reemplazar_traza(self):
        self.trace_path.write_text("evidencia anterior")
        completion = ChatCompletion(
            id="respuesta-ficticia", created=0, model="modelo-ficticio", object="chat.completion",
            choices=[{"index": 0, "finish_reason": "length",
                      "message": {"role": "assistant", "content": "JSON incompleto"}}],
        )
        with patch("main.load_dotenv"), patch("main.cargar_configuracion", return_value=self.config):
            with patch("main.ejecutar_demo", new_callable=AsyncMock,
                       side_effect=LengthFinishReasonError(completion=completion)) as demo:
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(await main(), 1)
                demo.assert_awaited_once()
        self.assertIn("OPENAI_MAX_TOKENS", output.getvalue())
        self.assertNotIn("JSON incompleto", output.getvalue())
        self.assertEqual(self.trace_path.read_text(), "evidencia anterior")

    def test_serializacion_de_mensajes_no_incluye_metadatos_del_proveedor(self):
        message = HumanMessage(content="Pregunta", response_metadata={"secreto": "no-publicar"})
        result = serializar({"messages": [message]})
        self.assertEqual(result["messages"][0]["contenido"], "Pregunta")
        self.assertNotIn("no-publicar", json.dumps(result))

    def test_fallo_de_escritura_no_reemplaza_traza_anterior(self):
        self.trace_path.write_text("anterior")
        with patch.object(Path, "replace", side_effect=OSError("fallo simulado")):
            with self.assertRaises(OSError):
                guardar_traza({"escenario": []})
        self.assertEqual(self.trace_path.read_text(), "anterior")
        self.assertFalse(self.trace_path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
