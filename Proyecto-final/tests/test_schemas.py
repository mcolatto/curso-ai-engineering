"""Contratos turísticos comprobados sin modelos ni fuentes externas."""

import json
import unittest

from pydantic import ValidationError

from schemas import (
    Coordenadas, DecisionSupervisor, Observacion, OpcionEvaluada, OpcionHotel,
    OpcionTuristica, Periodo, ResultadoAnalisis, ResultadoHoteleria, ResultadoTurismo,
    SolicitudParcial, SolicitudViaje,
)


class SchemasTests(unittest.TestCase):
    def setUp(self):
        self.request = {
            "destino": "Tokio", "fecha_inicio": "2027-02-12", "fecha_fin": "2027-02-20",
            "cantidad_personas": 4, "componentes": ["alojamiento", "actividades", "gastronomia"],
        }
        self.hotel = {
            "id": "hotel-1", "nombre": "Hotel ficticio", "fuente": "hoteles.json",
            "disponibilidad": "sin_datos",
        }

    def test_solicitud_completa_fechas_y_exportacion_json(self):
        request = SolicitudViaje(**self.request)
        self.assertEqual(request.datos_faltantes(), [])
        self.assertEqual(request.preferencias, [])
        self.assertEqual(json.loads(request.model_dump_json())["fecha_inicio"], "2027-02-12")

    def test_solicitud_parcial_identifica_datos_faltantes(self):
        request = SolicitudParcial(destino="Tokio")
        self.assertEqual(request.datos_faltantes(), ["fecha_inicio", "fecha_fin", "cantidad_personas"])
        for field in ("destino", "fecha_inicio", "fecha_fin", "cantidad_personas", "componentes"):
            data = self.request.copy()
            del data[field]
            with self.subTest(field=field), self.assertRaises(ValidationError):
                SolicitudViaje(**data)

    def test_rechaza_solicitudes_invalidas(self):
        for changes in (
            {"destino": " \t"}, {"destino": 123}, {"cantidad_personas": 0},
            {"cantidad_personas": -1}, {"cantidad_personas": 1.5},
            {"fecha_fin": "2027-02-12"}, {"fecha_fin": "2027-02-11"},
            {"fecha_inicio": "fecha"}, {"componentes": []},
            {"componentes": ["vuelos"]}, {"preferencias": [" "]},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                SolicitudViaje(**{**self.request, **changes})
        with self.assertRaises(ValidationError):
            SolicitudParcial(fecha_inicio="2027-02-20", fecha_fin="2027-02-12")

    def test_periodos_inclusivos_y_coordenadas(self):
        Periodo(inicio="2027-02-12", fin="2027-02-12")
        with self.assertRaises(ValidationError):
            Periodo(inicio="2027-02-20", fin="2027-02-12")
        self.assertEqual(Coordenadas(latitud=35.6812, longitud=139.7671).latitud, 35.6812)
        for data in ({"latitud": 91, "longitud": 0}, {"latitud": 0, "longitud": -181},
                     {"latitud": float("nan"), "longitud": 0}):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                Coordenadas(**data)

    def test_hotel_conserva_procedencia_y_datos_desconocidos(self):
        hotel = OpcionHotel(**self.hotel)
        self.assertEqual(hotel.fuente, "hoteles.json")
        self.assertIsNone(hotel.capacidad)
        self.assertIsNone(hotel.coordenadas)
        self.assertIsNone(hotel.periodos_disponibles)
        self.assertEqual(hotel.disponibilidad, "sin_datos")
        for status in ("disponible_en_fuente", "no_disponible", "sin_datos"):
            OpcionHotel(**{**self.hotel, "disponibilidad": status})
        for changes in ({"fuente": " "}, {"id": ""}, {"capacidad": 0},
                        {"estrellas": 6}, {"disponibilidad": "inventada"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                OpcionHotel(**{**self.hotel, **changes})

    def test_turismo_conserva_categorias_y_restricciones(self):
        for category in ("gastronomy", "attraction", "viewpoint"):
            option = OpcionTuristica(
                id="opcion-1", nombre="Opción ficticia", fuente="turismo.json",
                categoria=category, restricciones=["Cierra los lunes"],
            )
            self.assertEqual(option.restricciones, ["Cierra los lunes"])
            self.assertIsNone(option.periodos_disponibles)
        with self.assertRaises(ValidationError):
            OpcionTuristica(id="1", nombre="Opción", fuente="datos", categoria="otra")

    def test_busqueda_completada_vacia_no_equivale_a_error(self):
        for model, field in ((ResultadoHoteleria, "hoteles"), (ResultadoTurismo, "opciones")):
            completed = model(estado_busqueda="completada")
            failed = model(estado_busqueda="error", observaciones=[
                Observacion(tipo="limitacion", descripcion="Fuente no accesible")
            ])
            self.assertEqual(getattr(completed, field), [])
            self.assertEqual(getattr(failed, field), [])
            self.assertNotEqual(completed.estado_busqueda, failed.estado_busqueda)
            with self.assertRaises(ValidationError):
                model(estado_busqueda="pendiente")

    def test_analisis_referencia_opciones_y_exige_justificacion(self):
        for status in ("compatible", "pendiente_verificacion", "incompatible"):
            evaluated = OpcionEvaluada(
                opcion_id="hotel-1", dominio="hoteleria", clasificacion=status,
                justificacion="Evaluación según la fuente", distancia_km=1.2,
            )
            result = ResultadoAnalisis(sintesis_analisis="Síntesis", resultados_evaluados=[evaluated])
            self.assertEqual(result.resultados_evaluados[0].opcion_id, "hotel-1")
        for changes in ({"justificacion": " "}, {"distancia_km": -1},
                        {"distancia_km": float("inf")}, {"clasificacion": "desconocida"}):
            data = evaluated.model_dump()
            data.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                OpcionEvaluada(**data)

    def test_observaciones_y_decisiones_del_supervisor(self):
        with self.assertRaises(ValidationError):
            Observacion(tipo="incertidumbre", descripcion=" ")
        for route in ("hoteleria", "turismo", "analista", "finalizar"):
            decision = DecisionSupervisor(next_agent=route, instruccion="Tarea concreta")
            self.assertEqual(decision.next_agent, route)
        for evaluation in ("aprobado", "aprobado_con_advertencias", "desaprobado"):
            DecisionSupervisor(next_agent="finalizar", instruccion="Responder", evaluacion_supervisor=evaluation)
        with self.assertRaises(ValidationError):
            DecisionSupervisor(next_agent="otro", instruccion="Tarea")


if __name__ == "__main__":
    unittest.main()
