"""Pruebas de las herramientas mediante su interfaz asíncrona de LangChain."""

import unittest

from pydantic import ValidationError

from tools import buscar_cliente_por_nombre, buscar_pedidos, herramientas


class HerramientasTests(unittest.IsolatedAsyncioTestCase):
    async def test_busca_los_tres_clientes_conocidos(self):
        for nombre, cliente_id in (("Ana García", 102), ("Juan Pérez", 205), ("María López", 310)):
            with self.subTest(nombre=nombre):
                result = await buscar_cliente_por_nombre.ainvoke({"nombre": nombre})
                self.assertEqual(result, f"Cliente encontrado: '{nombre}' → cliente_id={cliente_id}")

    async def test_normaliza_mayusculas_y_espacios_exteriores(self):
        nombre = "  ANA GARCÍA  "
        result = await buscar_cliente_por_nombre.ainvoke({"nombre": nombre})
        self.assertEqual(result, f"Cliente encontrado: '{nombre}' → cliente_id=102")

    async def test_cliente_inexistente_o_nombre_incompleto_devuelve_error(self):
        for nombre in ("Roberto Sánchez", "Ana", "Ana Garcia", "", "   "):
            with self.subTest(nombre=nombre):
                result = await buscar_cliente_por_nombre.ainvoke({"nombre": nombre})
                self.assertEqual(result, (
                    f"ERROR: no se encontró ningún cliente con el nombre '{nombre}'. "
                    "Verificá que el nombre esté completo y bien escrito."
                ))

    async def test_consulta_pedidos_y_total_de_los_tres_clientes(self):
        for cliente_id, expected in (
            (102, "pedidos=3, total=$14500"),
            (205, "pedidos=1, total=$3200"),
            (310, "pedidos=5, total=$27800"),
        ):
            with self.subTest(cliente_id=cliente_id):
                self.assertEqual(await buscar_pedidos.ainvoke({"cliente_id": cliente_id}), expected)

    async def test_id_inexistente_devuelve_error(self):
        result = await buscar_pedidos.ainvoke({"cliente_id": 999})
        self.assertEqual(result, "ERROR: no existe ningún cliente con id=999.")

    async def test_argumentos_faltantes_o_con_tipos_incorrectos(self):
        cases = [
            (buscar_cliente_por_nombre, {}),
            (buscar_cliente_por_nombre, {"nombre": None}),
            (buscar_cliente_por_nombre, {"nombre": 102}),
            (buscar_pedidos, {}),
            (buscar_pedidos, {"cliente_id": None}),
            (buscar_pedidos, {"cliente_id": "Ana García"}),
            (buscar_pedidos, {"cliente_id": 102.5}),
        ]
        for herramienta, arguments in cases:
            with self.subTest(herramienta=herramienta.name, arguments=arguments):
                with self.assertRaises(ValidationError):
                    await herramienta.ainvoke(arguments)

    async def test_herramientas_exponen_nombres_descripciones_y_esquemas(self):
        self.assertEqual(herramientas, [buscar_cliente_por_nombre, buscar_pedidos])
        for herramienta, field, field_type in (
            (buscar_cliente_por_nombre, "nombre", "string"),
            (buscar_pedidos, "cliente_id", "integer"),
        ):
            with self.subTest(herramienta=herramienta.name):
                schema = herramienta.args_schema.model_json_schema()
                self.assertEqual(schema["required"], [field])
                self.assertEqual(schema["properties"][field]["type"], field_type)
                self.assertIn("cliente_id", herramienta.description)
                self.assertIn("ERROR", herramienta.description)
                self.assertIsNotNone(herramienta.coroutine)


if __name__ == "__main__":
    unittest.main()
