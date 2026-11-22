"""La inyección se limita al primer análisis y el feedback exige evidencia concreta."""

import json
import os
import unittest
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from checkpoints import TIPOS_CHECKPOINT, abrir_checkpointer
from config import Configuracion
from demo_refinamiento import crear_analista_con_omision, feedback_identifica_omision
from schemas import (
    DecisionSupervisor, OpcionEvaluada, ResultadoAnalisis, ResultadoHoteleria,
    ResultadoTurismo, SalidaSupervisor, SolicitudViaje,
)
from scripts import prueba_refinamiento as prueba
from state import crear_estado_inicial
from tools import buscar_hoteles, buscar_turismo, leer_catalogo


@asynccontextmanager
async def checkpointer_en_memoria(redis_url):
    """Simular sólo el almacenamiento en los tests que no requieren Redis."""
    yield InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=TIPOS_CHECKPOINT))


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

    def preparar_modelo_del_recorrido(self, feedback, *, corregir=True):
        """Programar decisiones sólo en tests; el script real no interviene en ellas."""
        solicitud = SolicitudViaje(
            destino="Tokio", fecha_inicio="2027-02-12", fecha_fin="2027-02-20",
            cantidad_personas=4, componentes=["alojamiento", "actividades", "gastronomia"],
        )
        decisiones = [SalidaSupervisor(
            es_turistica=True, solicitud=solicitud,
            decision=DecisionSupervisor(next_agent=ruta, instruccion=f"Ejecutar {ruta}"),
        ) for ruta in ("hoteleria", "turismo", "analista")]
        if feedback is not None:
            decisiones.append(SalidaSupervisor(
                es_turistica=True, decision=DecisionSupervisor(next_agent="analista", instruccion=feedback),
            ))
        decisiones.append(SalidaSupervisor(
            es_turistica=True,
            decision=DecisionSupervisor(next_agent="finalizar", instruccion="Responder",
                                        evaluacion_supervisor="aprobado"),
            respuesta_final="Recomendaciones para el viaje a Tokio.",
        ))
        supervisor = MagicMock()
        supervisor.ainvoke = AsyncMock(side_effect=decisiones)
        revisado = self.result.model_copy(deep=True)
        if not corregir:
            revisado.resultados_evaluados = revisado.resultados_evaluados[:-1]
        analista = MagicMock()
        analista.ainvoke = AsyncMock(side_effect=[self.result, revisado])
        modelo = MagicMock()
        modelos = {SalidaSupervisor: supervisor, ResultadoAnalisis: analista}
        modelo.with_structured_output.side_effect = modelos.__getitem__

        def vincular(herramientas, **kwargs):
            investigador = MagicMock()
            investigador.ainvoke = AsyncMock(return_value=AIMessage(content="", tool_calls=[{
                "name": herramientas[0].name, "args": {"destino": "Tokio"}, "id": "consulta-local",
            }]))
            return investigador

        modelo.bind_tools.side_effect = vincular
        return modelo, supervisor, analista

    async def test_script_conserva_exito_y_fallos_sin_corregir_decisiones(self):
        casos = (
            ("concreto", "Falta evaluar el restaurante; completá su clasificación.", True, True),
            ("sin_deteccion", None, True, False),
            ("generico", "Revisá el análisis.", True, False),
            ("sin_corregir", "Falta evaluar el restaurante; completá su clasificación.", False, False),
        )
        for nombre, feedback, corregir, satisfactoria in casos:
            with self.subTest(caso=nombre), TemporaryDirectory() as carpeta:
                modelo, supervisor, analista = self.preparar_modelo_del_recorrido(feedback, corregir=corregir)
                config = Configuracion(openai_api_key="clave-ficticia")
                destino = Path(carpeta) / "traza.json"
                with (
                    patch.object(prueba, "crear_llm", return_value=modelo),
                    patch.object(prueba, "abrir_checkpointer", side_effect=checkpointer_en_memoria),
                    patch.object(prueba, "load_dotenv"),
                    patch.object(prueba, "cargar_configuracion", return_value=config),
                    patch("tools.leer_catalogo", wraps=leer_catalogo) as leer,
                    patch("builtins.print"),
                ):
                    codigo = await prueba.main(["--salida", str(destino)])
                self.assertEqual(codigo, 0 if satisfactoria else 1)
                traza = json.loads(destino.read_text(encoding="utf-8"))
                self.assertEqual(traza["resultado_prueba"]["satisfactoria"], satisfactoria)
                self.assertEqual(traza["estado"], "SATISFACTORIA" if satisfactoria else "NO_SATISFACTORIA")
                self.assertIsNone(traza["error_ejecucion"])
                self.assertTrue(traza["control_prueba"]["omision_aplicada"])
                self.assertEqual(traza["control_prueba"]["cantidad_antes"], 5)
                self.assertEqual(traza["control_prueba"]["cantidad_despues"], 4)
                eventos = traza["eventos"]
                for nodo in ("hoteleria", "turismo"):
                    self.assertEqual(sum(evento["nodo"] == nodo for evento in eventos), 1)
                self.assertEqual(leer.call_count, 2)
                analisis = [evento for evento in eventos if evento["nodo"] == "analista"]
                self.assertEqual(len(analisis), 1 if feedback is None else 2)
                self.assertEqual(len(analisis[0]["actualizacion"]["resultado_analisis"]["resultados_evaluados"]), 4)
                self.assertEqual(analista.ainvoke.await_count, len(analisis))
                self.assertEqual(supervisor.ainvoke.await_count, 4 if feedback is None else 5)
                if feedback is not None:
                    self.assertEqual(traza["resultado_prueba"]["feedback_observado"], feedback)
                    self.assertEqual(traza["estado_final"]["refinamientos_analista"], 1)
                self.assertNotIn("clave-ficticia", destino.read_text(encoding="utf-8"))
                self.assertIn("Análisis original", destino.read_text(encoding="utf-8"))

    async def test_fallo_despues_de_la_omision_conserva_eventos_y_checkpoint_parcial(self):
        modelo, supervisor, _ = self.preparar_modelo_del_recorrido(None)
        decisiones = list(supervisor.ainvoke.side_effect)
        decisiones[-1] = RuntimeError("detalle-sensible-ficticio")
        supervisor.ainvoke.side_effect = decisiones
        with TemporaryDirectory() as carpeta:
            destino = Path(carpeta) / "parcial.json"
            with (
                patch.object(prueba, "crear_llm", return_value=modelo),
                patch.object(prueba, "abrir_checkpointer", side_effect=checkpointer_en_memoria),
                patch("builtins.print"),
            ):
                traza, _ = await prueba.ejecutar_prueba(Configuracion(openai_api_key="clave-ficticia"), destino)
            self.assertEqual(json.loads(destino.read_text(encoding="utf-8")), traza)
            self.assertEqual(traza["error_ejecucion"], "RuntimeError")
            self.assertFalse(traza["resultado_prueba"]["satisfactoria"])
            self.assertTrue(traza["control_prueba"]["omision_aplicada"])
            self.assertEqual(traza["eventos"][-1]["nodo"], "analista")
            evaluaciones = traza["estado_final"]["resultado_analisis"]["resultados_evaluados"]
            self.assertEqual(len(evaluaciones), 4)
            self.assertNotIn("detalle-sensible-ficticio", destino.read_text(encoding="utf-8"))

    async def test_error_operativo_conserva_traza_sin_exponer_el_mensaje(self):
        with TemporaryDirectory() as carpeta:
            destino = Path(carpeta) / "fallo.json"
            with (
                patch.object(prueba, "abrir_checkpointer", side_effect=checkpointer_en_memoria),
                patch.object(prueba, "crear_llm", side_effect=RuntimeError("secreto-que-no-debe-salir")),
            ):
                traza, _ = await prueba.ejecutar_prueba(Configuracion(openai_api_key="clave-ficticia"), destino)
            texto = destino.read_text(encoding="utf-8")
            self.assertEqual(json.loads(texto), traza)
            self.assertEqual(traza["error_ejecucion"], "RuntimeError")
            self.assertFalse(traza["resultado_prueba"]["satisfactoria"])
            self.assertNotIn("secreto-que-no-debe-salir", texto)

    async def test_no_sobrescribe_evidencia_ni_crea_clientes_si_la_salida_existe(self):
        with TemporaryDirectory() as carpeta:
            destino = Path(carpeta) / "anterior.json"
            destino.write_text('{"anterior": true}', encoding="utf-8")
            with patch.object(prueba, "crear_llm") as crear, patch.object(prueba, "abrir_checkpointer") as abrir:
                with self.assertRaises(FileExistsError):
                    await prueba.ejecutar_prueba(Configuracion(openai_api_key="clave-ficticia"), destino)
            crear.assert_not_called()
            abrir.assert_not_called()
            self.assertEqual(destino.read_text(encoding="utf-8"), '{"anterior": true}')

    @unittest.skipUnless(os.getenv("TEST_REDIS_URL"), "Requiere un Redis 8 descartable")
    async def test_script_con_redis_y_arbol_de_trazas_sin_servicios_externos(self):
        modelo, _, _ = self.preparar_modelo_del_recorrido("Completá la evaluación gastronómica omitida.")
        config = Configuracion(
            openai_api_key="clave-ficticia", redis_url=os.environ["TEST_REDIS_URL"],
            langsmith_tracing=True, langsmith_api_key="clave-langsmith-ficticia",
        )
        with TemporaryDirectory() as carpeta:
            with (
                patch.object(prueba, "crear_llm", return_value=modelo),
                patch("observability.Client") as cliente,
                patch("builtins.print"),
            ):
                traza, _ = await prueba.ejecutar_prueba(config, Path(carpeta) / "redis.json")
            self.assertTrue(traza["resultado_prueba"]["satisfactoria"])
            self.assertEqual(traza["job_id"], traza["thread_id"])
            async with abrir_checkpointer(os.environ["TEST_REDIS_URL"]) as saver:
                checkpoint = await saver.aget_tuple({"configurable": {"thread_id": traza["thread_id"]}})
                self.assertIsNotNone(checkpoint)
                valores = checkpoint.checkpoint["channel_values"]
                self.assertEqual(valores["refinamientos_analista"], 1)
                self.assertEqual(len(valores["resultado_analisis"].resultados_evaluados), 5)
            runs = [llamada.kwargs for llamada in cliente.return_value.create_run.call_args_list]
            raices = [run for run in runs if run.get("parent_run_id") is None]
            self.assertEqual(len(raices), 1)
            raiz = raices[0]
            self.assertEqual(raiz["id"], UUID(traza["run_id"]))
            self.assertEqual(raiz["session_name"], "proyecto-final")
            self.assertIn("refinamiento_controlado", raiz["tags"])
            for nodo, cantidad in (("analista", 2), ("supervisor", 5), ("hoteleria", 1), ("turismo", 1)):
                hijos = [run for run in runs if run["name"] == nodo and run.get("parent_run_id") == raiz["id"]]
                self.assertEqual(len(hijos), cantidad)
                for hijo in hijos:
                    self.assertEqual(hijo["extra"]["metadata"]["thread_id"], traza["thread_id"])
            cliente.return_value.close.assert_called_once_with(timeout=5)


if __name__ == "__main__":
    unittest.main()
