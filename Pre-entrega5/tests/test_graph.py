"""Grafo y herramientas reales con decisiones del modelo simuladas."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START

from graph import INSTRUCCIONES, crear_grafo


def llamada(name: str, args: dict, call_id: str) -> AIMessage:
    """Preparar una decisión de herramienta sin consultar un modelo real."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class GrafoTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm = MagicMock()
        self.llm.ainvoke = AsyncMock()
        self.question = HumanMessage(content="¿Cuántos pedidos tuvo Ana García y cuál fue el total?")

    async def test_dos_llamadas_dependientes_y_respuesta_sin_duplicar_historial(self):
        self.llm.ainvoke.side_effect = [
            llamada("buscar_cliente_por_nombre", {"nombre": "Ana García"}, "cliente-1"),
            llamada("buscar_pedidos", {"cliente_id": 102}, "pedidos-1"),
            AIMessage(content="Ana García tuvo 3 pedidos por un total de $14500."),
        ]
        app = crear_grafo(self.llm).compile()
        result = await app.ainvoke({"messages": [self.question]}, config={"recursion_limit": 10})
        messages = result["messages"]
        self.assertEqual([type(m) for m in messages], [
            HumanMessage, AIMessage, ToolMessage, AIMessage, ToolMessage, AIMessage,
        ])
        self.assertIn("cliente_id=102", messages[2].content)
        self.assertEqual(messages[2].tool_call_id, "cliente-1")
        self.assertEqual(messages[4].content, "pedidos=3, total=$14500")
        self.assertEqual(messages[4].tool_call_id, "pedidos-1")
        self.assertEqual(messages[-1].content, "Ana García tuvo 3 pedidos por un total de $14500.")
        self.assertEqual(self.llm.ainvoke.await_count, 3)
        calls = self.llm.ainvoke.await_args_list
        self.assertEqual([len(call.args[0]) for call in calls], [2, 4, 6])
        self.assertIsInstance(calls[0].args[0][0], SystemMessage)
        self.assertEqual(calls[0].args[0][0].content, INSTRUCCIONES)
        self.assertIn("cliente_id=102", calls[1].args[0][-1].content)
        self.assertEqual(calls[2].args[0][-1].content, "pedidos=3, total=$14500")
        self.assertFalse(any(isinstance(m, SystemMessage) for m in messages))

    async def test_respuesta_directa_termina_sin_herramientas(self):
        self.llm.ainvoke.return_value = AIMessage(content="Hola, ¿qué cliente querés consultar?")
        result = await crear_grafo(self.llm).compile().ainvoke(
            {"messages": [HumanMessage(content="Hola")]}, config={"recursion_limit": 10}
        )
        self.assertEqual(len(result["messages"]), 2)
        self.assertFalse(any(isinstance(m, ToolMessage) for m in result["messages"]))
        self.llm.ainvoke.assert_awaited_once()

    async def test_cliente_inexistente_vuelve_al_modelo_antes_de_pedir_aclaracion(self):
        self.llm.ainvoke.side_effect = [
            llamada("buscar_cliente_por_nombre", {"nombre": "Roberto Sánchez"}, "inexistente-1"),
            AIMessage(content="No encontré a Roberto Sánchez. ¿Podés confirmar el nombre completo?"),
        ]
        result = await crear_grafo(self.llm).compile().ainvoke(
            {"messages": [HumanMessage(content="Consultá los pedidos de Roberto Sánchez")]},
            config={"recursion_limit": 10},
        )
        tool_message = result["messages"][2]
        self.assertIsInstance(tool_message, ToolMessage)
        self.assertTrue(tool_message.content.startswith("ERROR:"))
        self.assertEqual(tool_message.tool_call_id, "inexistente-1")
        self.assertEqual(self.llm.ainvoke.await_args_list[1].args[0][-1], tool_message)
        self.assertIn("confirmar", result["messages"][-1].content)
        self.assertEqual(self.llm.ainvoke.await_count, 2)

    async def test_argumentos_invalidos_se_devuelven_como_error_de_herramienta(self):
        self.llm.ainvoke.side_effect = [
            llamada("buscar_pedidos", {"cliente_id": "no-es-un-id"}, "argumento-invalido"),
            AIMessage(content="Necesito identificar al cliente. ¿Cuál es su nombre completo?"),
        ]
        result = await crear_grafo(self.llm).compile().ainvoke(
            {"messages": [self.question]}, config={"recursion_limit": 10}
        )
        tool_message = result["messages"][2]
        self.assertIsInstance(tool_message, ToolMessage)
        self.assertEqual(tool_message.status, "error")
        self.assertEqual(tool_message.tool_call_id, "argumento-invalido")
        self.assertEqual(self.llm.ainvoke.await_args_list[1].args[0][-1], tool_message)

    async def test_limite_de_recursion_detiene_ciclo_sin_respuesta_final(self):
        count = 0

        async def repetir(messages):
            nonlocal count
            count += 1
            return llamada("buscar_pedidos", {"cliente_id": 102}, f"repetida-{count}")

        self.llm.ainvoke.side_effect = repetir
        with self.assertRaises(GraphRecursionError):
            await crear_grafo(self.llm).compile().ainvoke(
                {"messages": [self.question]}, config={"recursion_limit": 4}
            )
        self.assertEqual(self.llm.ainvoke.await_count, 2)

    async def test_error_del_modelo_no_se_disfraza_de_respuesta(self):
        self.llm.ainvoke.side_effect = RuntimeError("fallo simulado")
        with self.assertRaises(RuntimeError):
            await crear_grafo(self.llm).compile().ainvoke(
                {"messages": [self.question]}, config={"recursion_limit": 10}
            )

    def test_construir_no_invoca_modelo_y_conserva_aristas_del_ciclo(self):
        grafo = crear_grafo(self.llm)
        self.llm.ainvoke.assert_not_called()
        diagram = grafo.compile().get_graph()
        edges = {(edge.source, edge.target) for edge in diagram.edges}
        self.assertTrue({
            (START, "modelo"), ("modelo", "herramientas"),
            ("herramientas", "modelo"), ("modelo", END),
        }.issubset(edges))


if __name__ == "__main__":
    unittest.main()
