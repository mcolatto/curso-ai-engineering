"""Verificar el árbol de trazas usando el trazador real y un cliente simulado."""

import asyncio
import unittest
from unittest.mock import patch
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langsmith.run_helpers import get_tracing_context
from pydantic import ValidationError

from config import Configuracion
from observability import CAMPOS_CONFIGURACION_PUBLICA, Observabilidad, configurar_invocacion
from tools import buscar_hoteles


class ObservabilidadTests(unittest.IsolatedAsyncioTestCase):
    async def test_trazas_reales_anidan_nodo_herramienta_y_modelo_sin_red(self):
        modelo = FakeMessagesListChatModel(responses=[
            AIMessage(content="Respuesta local", usage_metadata={
                "input_tokens": 5, "output_tokens": 3, "total_tokens": 8,
            }),
        ])

        async def investigador(state):
            await buscar_hoteles.ainvoke({"destino": "Tokio"})
            return {"messages": [await modelo.ainvoke(state["messages"])]}

        builder = StateGraph(MessagesState)
        builder.add_node("investigador", investigador)
        builder.add_edge(START, "investigador")
        builder.add_edge("investigador", END)
        grafo = builder.compile()
        config = Configuracion(
            openai_api_key="clave-openai-ficticia", langsmith_api_key="clave-langsmith-ficticia",
            langsmith_tracing=True, langsmith_project="prueba-local",
        )

        with patch("observability.Client") as cliente:
            observabilidad = Observabilidad(config)

            async def ejecutar(job_id, es_reanudacion):
                invocacion = configurar_invocacion(job_id, 10, es_reanudacion=es_reanudacion)
                with observabilidad.contexto():
                    await grafo.ainvoke({"messages": [HumanMessage(content="Tokio")]}, config=invocacion)
                return invocacion

            job_id = uuid4()
            otro_job_id = uuid4()
            inicio, reanudacion, otro_inicio = await asyncio.gather(
                ejecutar(job_id, False), ejecutar(job_id, True), ejecutar(otro_job_id, False),
            )
            await observabilidad.cerrar()

        cliente.assert_called_once_with(api_key="clave-langsmith-ficticia")
        cliente.return_value.close.assert_called_once_with(timeout=5)
        runs = [llamada.kwargs for llamada in cliente.return_value.create_run.call_args_list]
        updates = {
            str(llamada.args[0] if llamada.args else llamada.kwargs["run_id"]): llamada.kwargs
            for llamada in cliente.return_value.update_run.call_args_list
        }
        raices = {run["id"]: run for run in runs if run.get("parent_run_id") is None}
        self.assertEqual(inicio["run_id"], job_id)
        self.assertEqual(inicio["tags"], ["proyecto-final", "inicio"])
        self.assertNotEqual(reanudacion["run_id"], job_id)
        self.assertEqual(set(raices), {inv["run_id"] for inv in (inicio, reanudacion, otro_inicio)})
        for invocacion in (inicio, reanudacion, otro_inicio):
            raiz = raices[invocacion["run_id"]]
            self.assertEqual(raiz["session_name"], "prueba-local")
            self.assertTrue(raiz["start_time"])
            self.assertTrue(updates[str(raiz["id"])]["end_time"])
            subtrazas = [run for run in runs if run["trace_id"] == raiz["id"] and run["id"] != raiz["id"]]
            nodo = next(run for run in subtrazas if run["name"] == "investigador")
            herramienta = next(run for run in subtrazas if run["run_type"] == "tool")
            llm = next(run for run in subtrazas if run["run_type"] in ("llm", "chat_model"))
            self.assertEqual(herramienta["name"], "buscar_hoteles")
            self.assertEqual(herramienta["parent_run_id"], nodo["id"])
            self.assertEqual(llm["parent_run_id"], nodo["id"])
            for run in (raiz, nodo, herramienta, llm):
                metadata = run["extra"]["metadata"]
                for clave, valor in invocacion["metadata"].items():
                    self.assertEqual(metadata[clave], valor)
                publica = metadata["configuracion_ejecucion"]
                self.assertEqual(set(publica), set(CAMPOS_CONFIGURACION_PUBLICA))
                self.assertEqual(publica["model"], config.model)
                self.assertEqual(publica["max_tokens"], config.max_tokens)
                self.assertNotIn("openai_api_key", publica)
                self.assertNotIn("langsmith_api_key", publica)
                self.assertNotIn("redis_url", publica)
            self.assertEqual(updates[str(llm["id"])]["extra"]["metadata"]["usage_metadata"]["total_tokens"], 8)

    def test_requiere_credencial_solo_si_esta_activada(self):
        with self.assertRaisesRegex(ValidationError, "LANGSMITH_API_KEY"):
            Configuracion(openai_api_key="clave-ficticia", langsmith_tracing=True)
        config = Configuracion(openai_api_key="clave-ficticia")
        with patch("observability.Client") as cliente, Observabilidad(config).contexto():
            self.assertFalse(get_tracing_context()["enabled"])
        cliente.assert_not_called()

    def test_destino_por_agente_y_credencial_oculta(self):
        config = Configuracion(
            openai_api_key="clave-openai-ficticia", langsmith_api_key="clave-langsmith-ficticia",
            langsmith_tracing=True, langsmith_agent_id="proyecto-final",
        )
        with patch("observability.Client"), Observabilidad(config).contexto():
            context = get_tracing_context()
            self.assertEqual(context["address"], {"kind": "AGENT", "id": "proyecto-final", "environment": "LOCAL"})
            self.assertIsNone(context["project_name"])
        self.assertNotIn("clave-langsmith-ficticia", config.model_dump_json())
