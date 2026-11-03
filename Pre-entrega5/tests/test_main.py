"""Persistencia SQLite real y ejecución de consola con modelo simulado."""

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from config import Configuracion
from graph import crear_grafo
from main import ejecutar_demo, guardar_traza, main, serializar_traza


def llamada(name: str, args: dict, call_id: str) -> AIMessage:
    """Representar una llamada decidida por el modelo simulado."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class MainTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / "checkpoints.sqlite"
        patcher = patch("main.CHECKPOINT_PATH", self.database)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.trace_path = Path(directory.name) / "traza_ejecucion.json"
        patcher = patch("main.TRACE_PATH", self.trace_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = Configuracion(openai_api_key="clave-ficticia")
        self.llm = MagicMock()
        self.llm.ainvoke = AsyncMock()

    def respuestas_demo(self):
        return [
            llamada("buscar_cliente_por_nombre", {"nombre": "Ana García"}, "ana-id"),
            llamada("buscar_pedidos", {"cliente_id": 102}, "ana-pedidos"),
            AIMessage(content="Ana tuvo 3 pedidos por $14500."),
            llamada("buscar_cliente_por_nombre", {"nombre": "Juan Pérez"}, "juan-id"),
            llamada("buscar_pedidos", {"cliente_id": 205}, "juan-pedidos"),
            AIMessage(content="Juan tuvo 1 pedido por $3200."),
            llamada("buscar_cliente_por_nombre", {"nombre": "Roberto Sánchez"}, "roberto-id"),
            AIMessage(content="No encontré al cliente. ¿Podés confirmar el nombre?"),
        ]

    async def test_demo_persiste_contexto_y_aisla_el_caso_de_error(self):
        self.llm.ainvoke.side_effect = self.respuestas_demo()
        with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()) as output:
            results = await ejecutar_demo(self.config)
        self.assertEqual(list(results), ["turno_1_multi_paso", "turno_2_memoria", "prueba_error_y_aclaracion"])
        self.assertEqual([len(messages) for messages in results.values()], [6, 6, 4])
        calls = self.llm.ainvoke.await_args_list
        second_turn = calls[3].args[0]
        self.assertIsInstance(second_turn[0], SystemMessage)
        self.assertEqual(second_turn[1:], [*results["turno_1_multi_paso"], results["turno_2_memoria"][0]])
        self.assertEqual(second_turn[-1].content, "¿Y Juan Pérez?")
        error_turn = calls[6].args[0]
        self.assertEqual(len(error_turn), 2)
        self.assertIn("Roberto Sánchez", error_turn[-1].content)
        self.assertIn("pedidos=1, total=$3200", calls[5].args[0][-1].content)
        self.assertTrue(calls[7].args[0][-1].content.startswith("ERROR:"))
        self.assertIn("Juan tuvo 1 pedido", output.getvalue())
        with sqlite3.connect(self.database) as connection:
            threads = connection.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        self.assertEqual(len(threads), 2)

    async def test_repetir_demo_crea_threads_nuevos_sin_historial_anterior(self):
        self.llm.ainvoke.side_effect = self.respuestas_demo() + self.respuestas_demo()
        with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()):
            await ejecutar_demo(self.config)
            await ejecutar_demo(self.config)
        self.assertEqual(len(self.llm.ainvoke.await_args_list[8].args[0]), 2)
        with sqlite3.connect(self.database) as connection:
            threads = connection.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        self.assertEqual(len(threads), 4)

    async def test_recupera_historial_tras_cerrar_y_reabrir_sqlite(self):
        self.llm.ainvoke.side_effect = [AIMessage(content="Respuesta anterior"), AIMessage(content="Continuación")]
        invocation = {"configurable": {"thread_id": "persistencia-test"}, "recursion_limit": 10}
        first_question = HumanMessage(content="Consultemos los pedidos de Ana García")
        async with AsyncSqliteSaver.from_conn_string(str(self.database)) as saver:
            app = crear_grafo(self.llm).compile(checkpointer=saver)
            first = await app.ainvoke({"messages": [first_question]}, config=invocation)
        async with AsyncSqliteSaver.from_conn_string(str(self.database)) as saver:
            app = crear_grafo(self.llm).compile(checkpointer=saver)
            second = await app.ainvoke({"messages": [HumanMessage(content="¿Y Juan Pérez?")]}, config=invocation)
        self.assertEqual(second["messages"][:2], first["messages"])
        self.assertEqual(len(second["messages"]), 4)
        self.assertEqual(self.llm.ainvoke.await_args_list[1].args[0][1:3], first["messages"])

    async def test_demo_aplica_limite_de_recursion_configurado(self):
        self.llm.ainvoke.return_value = llamada("buscar_pedidos", {"cliente_id": 102}, "limite")
        config = Configuracion(openai_api_key="clave-ficticia", recursion_limit=1)
        with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()):
            with self.assertRaises(GraphRecursionError):
                await ejecutar_demo(config)
        self.llm.ainvoke.assert_awaited_once()
        # El contexto del saver se cerró incluso al interrumpirse el grafo.
        async with AsyncSqliteSaver.from_conn_string(str(self.database)) as saver:
            await saver.setup()

    async def test_main_informa_exito_solo_al_finalizar(self):
        with patch("main.load_dotenv"), patch("main.cargar_configuracion", return_value=self.config):
            with patch("main.ejecutar_demo", new_callable=AsyncMock) as demo:
                demo.return_value = {"turno_1_multi_paso": [HumanMessage(content="Pregunta")],
                                     "turno_2_memoria": [AIMessage(content="Respuesta")],
                                     "prueba_error_y_aclaracion": [AIMessage(content="Aclaración")]}
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(await main(), 0)
                demo.assert_awaited_once_with(self.config)
        self.assertIn("Demostración finalizada", output.getvalue())
        self.assertTrue(self.trace_path.is_file())

    async def test_main_controla_error_operativo_sin_secretos_ni_exito(self):
        with patch("main.load_dotenv"), patch("main.cargar_configuracion", return_value=self.config):
            with patch("main.ejecutar_demo", new_callable=AsyncMock, side_effect=RuntimeError("secreto-ficticio")):
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(await main(), 1)
        self.assertIn("Demostración incompleta", output.getvalue())
        self.assertNotIn("secreto-ficticio", output.getvalue())
        self.assertNotIn("Demostración finalizada", output.getvalue())
        self.assertFalse(self.trace_path.exists())

    async def test_configuracion_invalida_no_ejecuta_demo(self):
        with patch("main.load_dotenv"), patch("main.cargar_configuracion", side_effect=ValueError("dato-ficticio")):
            with patch("main.ejecutar_demo", new_callable=AsyncMock) as demo:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(await main(), 1)
                demo.assert_not_awaited()

    async def test_traza_integrada_con_sqlite_y_modelo_simulado(self):
        self.llm.ainvoke.side_effect = self.respuestas_demo()
        with patch("main.load_dotenv"), patch("main.cargar_configuracion", return_value=self.config):
            with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()):
                self.assertEqual(await main(), 0)
        text = self.trace_path.read_text(encoding="utf-8")
        trace = json.loads(text)
        self.assertEqual(list(trace), ["turno_1_multi_paso", "turno_2_memoria", "prueba_error_y_aclaracion"])
        self.assertEqual([len(messages) for messages in trace.values()], [6, 6, 4])
        self.assertEqual(trace["turno_2_memoria"][0]["contenido"], "¿Y Juan Pérez?")
        self.assertNotIn("Ana", json.dumps(trace["turno_2_memoria"]))
        self.assertIn("García", text)
        self.assertNotIn("clave-ficticia", text)
        for messages in trace.values():
            calls = {}
            for message in messages:
                for call in message.get("tool_calls", []):
                    calls[call["id"]] = call["nombre"]
                if message["tipo"] == "ToolMessage":
                    self.assertEqual(message["herramienta"], calls[message["tool_call_id"]])
        first = trace["turno_1_multi_paso"]
        self.assertEqual(first[1]["tool_calls"][0]["argumentos"], {"nombre": "Ana García"})
        self.assertEqual(first[3]["tool_calls"][0]["argumentos"], {"cliente_id": 102})
        self.assertEqual(first[4]["contenido"], "pedidos=3, total=$14500")
        self.assertTrue(trace["prueba_error_y_aclaracion"][2]["contenido"].startswith("ERROR:"))
        self.assertFalse(self.trace_path.with_suffix(".json.tmp").exists())

    def test_serializar_texto_y_bloques_sin_metadatos_del_proveedor(self):
        messages = [
            HumanMessage(content="¿Y María?"),
            AIMessage(content=[{"type": "text", "text": "Una respuesta"}],
                      response_metadata={"dato": "no-incluir"}),
            ToolMessage(content="ERROR: cliente inexistente", name="buscar_cliente_por_nombre",
                        tool_call_id="ejemplo"),
        ]
        trace = serializar_traza(messages)
        self.assertEqual([item["tipo"] for item in trace], ["HumanMessage", "AIMessage", "ToolMessage"])
        self.assertEqual(trace[1]["contenido"], "Una respuesta")
        self.assertNotIn("response_metadata", trace[1])
        self.assertEqual(trace[2]["tool_call_id"], "ejemplo")

    async def test_error_en_tercer_escenario_conserva_traza_anterior(self):
        self.trace_path.write_text('{"evidencia": "anterior"}', encoding="utf-8")
        self.llm.ainvoke.side_effect = self.respuestas_demo()[:6] + [RuntimeError("secreto-ficticio")]
        with patch("main.load_dotenv"), patch("main.cargar_configuracion", return_value=self.config):
            with patch("main.crear_llm", return_value=self.llm), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(await main(), 1)
        self.assertEqual(json.loads(self.trace_path.read_text()), {"evidencia": "anterior"})
        self.assertNotIn("Traza guardada", output.getvalue())

    def test_fallo_de_escritura_no_reemplaza_evidencia_anterior(self):
        original = '{"evidencia": "anterior"}'
        self.trace_path.write_text(original, encoding="utf-8")
        with patch.object(Path, "replace", side_effect=OSError("fallo simulado")):
            with self.assertRaises(OSError):
                guardar_traza({"turno_1_multi_paso": [AIMessage(content="Nuevo")]})
        self.assertEqual(self.trace_path.read_text(), original)
        self.assertFalse(self.trace_path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
