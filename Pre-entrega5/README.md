# Pre-entrega 5: agente ReAct con memoria persistente

## Objetivo

Implementar un agente asíncrono con LangGraph que decida cuándo consultar herramientas, utilice sus resultados en sucesivas iteraciones y conserve el contexto de una conversación mediante SQLite.

El agente usa OpenAI y dos herramientas de consulta sobre datos simulados. La demostración incluye una consulta de varios pasos, una pregunta de seguimiento con memoria y un cliente inexistente que requiere aclaración.

## Estructura

```text
Pre-entrega5/
├── config.py               # Configuración validada con Pydantic y rutas locales
├── tools.py                # Datos simulados y herramientas asíncronas con @tool
├── llm.py                  # ChatOpenAI y vinculación mediante bind_tools()
├── graph.py                # Estado, nodos, aristas y ciclo ReAct
├── main.py                 # Demostración, persistencia y serialización de la traza
├── tests/                  # Tests unittest sin llamadas a APIs
├── requirements.txt        # Versiones de las dependencias utilizadas
├── .env.example            # Plantilla de variables sin credenciales
├── .gitignore              # Exclusión de secretos, entorno y archivos SQLite
├── traza_ejecucion.json     # Evidencia de una ejecución real de los tres escenarios
├── checkpoints.sqlite      # Historial local generado al ejecutar; excluido de Git
└── README.md               # Instalación, uso y evidencia de funcionamiento
```

## Instalación

Requisitos: Python 3.12 y una API key de OpenAI con acceso y saldo o cuota para el modelo configurado. El entorno utilizado para los tests fue Python 3.12.15.

Desde la raíz del repositorio:

```bash
cd Pre-entrega5
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
cp .env.example .env
```

Completar `OPENAI_API_KEY` en `.env`. Los comandos siguientes se ejecutan desde `Pre-entrega5/`. No se debe versionar la clave.

## Configuración

| Variable | Predeterminado | Responsabilidad |
| --- | --- | --- |
| `OPENAI_API_KEY` | Obligatoria, sin valor | Credencial de OpenAI |
| `OPENAI_MODEL` | `gpt-6.1-sol` | Modelo con soporte para llamadas a herramientas |
| `OPENAI_TIMEOUT_SECONDS` | `30` | Timeout del cliente del modelo, en segundos |
| `OPENAI_MAX_TOKENS` | `1024` | Límite de salida enviado como `max_completion_tokens` |
| `OPENAI_REASONING_EFFORT` | `low` | Esfuerzo opcional: `low`, `medium`, `high`, `xhigh` o `max` |
| `RECURSION_LIMIT` | `10` | Límite de pasos del grafo por invocación |

Para omitir el parámetro de razonamiento:

```dotenv
OPENAI_REASONING_EFFORT=
```

Un valor vacío se convierte en `None` y no se envía al modelo; no equivale a desactivar su razonamiento. La ejecución real incluida utilizó esta opción vacía. Si se elige otro modelo, debe admitir las herramientas y los parámetros enviados.

Pydantic rechaza claves y modelos vacíos, límites no positivos, timeouts no finitos y esfuerzos desconocidos. La clave se maneja como `SecretStr` para ocultarla en representaciones. Validar su contenido localmente no verifica su autenticidad contra OpenAI.

No se envía temperatura y los reintentos automáticos del SDK están desactivados. Los módulos no crean clientes ni realizan solicitudes al importarse. Las rutas de SQLite y de la traza se resuelven respecto de esta carpeta.

## Herramientas y datos

Las herramientas están definidas con `@tool`, parámetros tipados y docstrings que explican cuándo utilizarlas:

- `buscar_cliente_por_nombre(nombre)`: obtiene el ID interno. Normaliza mayúsculas y espacios exteriores; conserva la búsqueda exacta, incluidos los acentos.
- `buscar_pedidos(cliente_id)`: obtiene cantidad de pedidos y total mediante el ID numérico.

| Cliente | ID | Pedidos | Total |
| --- | ---: | ---: | ---: |
| Ana García | 102 | 3 | 14500 |
| Juan Pérez | 205 | 1 | 3200 |
| María López | 310 | 5 | 27800 |

Los datos viven en diccionarios locales y no requieren una base de datos comercial ni servicios externos. Las herramientas son asíncronas; sus consultas en memoria no realizan operaciones bloqueantes de red.

Un nombre o ID inexistente devuelve un texto `ERROR: ...`. El modelo recibe ese resultado y puede pedir aclaración. Es distinto de una excepción técnica: `ToolNode(..., handle_tool_errors=True)` captura errores de ejecución o validación de herramientas y los devuelve al modelo como mensajes de herramienta.

## Grafo y ciclo ReAct

```text
START → modelo → tools_condition ── sin llamadas ──→ END
                      │
                 con llamadas
                      ↓
                 herramientas
                      │
                      └──────────────────────────→ modelo
```

`StateGraph(MessagesState)` conserva el historial mediante el reducer de mensajes de LangGraph. El nodo del modelo envía las instrucciones y el historial con `ainvoke()`, y devuelve únicamente el nuevo mensaje. Las instrucciones de sistema se agregan al contexto del modelo sin acumularlas en el estado persistido.

`bind_tools()` expone las herramientas al modelo. `tools_condition` sólo decide si el último mensaje contiene llamadas; no selecciona herramientas por palabras clave del usuario. Después de ejecutar una herramienta, el grafo vuelve al modelo para que decida cómo continuar.

En la primera consulta, el nombre no proporciona directamente el ID: el agente debe buscar al cliente y utilizar el ID recibido para consultar sus pedidos. La ejecución real registra ambas llamadas en iteraciones distintas antes de la respuesta final.

`recursion_limit` protege contra ciclos indefinidos. Cuenta pasos del grafo, no llamadas a OpenAI, y no establece un presupuesto monetario. Al agotarse, la demostración informa el fallo de forma controlada.

## Persistencia y escenarios

Se usa `AsyncSqliteSaver`, la variante asíncrona del checkpointer SQLite, dentro de un contexto `async with`. La demostración se ejecuta con `asyncio.run()` y guarda el estado en `checkpoints.sqlite`.

| Escenario | Entrada | Evidencia esperada |
| --- | --- | --- |
| `turno_1_multi_paso` | ¿Cuántos pedidos tuvo Ana García y cuál fue el total? | Búsqueda de ID, consulta de pedidos y respuesta |
| `turno_2_memoria` | ¿Y Juan Pérez? | Interpretación de la pregunta mediante el contexto anterior |
| `prueba_error_y_aclaracion` | ¿Cuántos pedidos tuvo el cliente Roberto Sánchez? | Cliente no encontrado y solicitud de aclaración, sin inventar datos |

Los dos primeros turnos usan el mismo `thread_id`. En el segundo se envía sólo el nuevo `HumanMessage`: el checkpointer recupera el contexto anterior. El caso de error utiliza otro thread para aislar la conversación.

Cada ejecución genera nuevos identificadores de thread, de modo que no reutiliza conversaciones antiguas al repetir la demostración. Esto no borra el historial previo de SQLite. La persistencia permite recuperarlo al volver a utilizar su mismo identificador; esa continuidad después de cerrar y reabrir la base también se comprueba en tests.

## Ejecución

```bash
.venv/bin/python main.py
```

El comando realiza llamadas reales a OpenAI y consume saldo o cuota. La cantidad depende de las decisiones del modelo. En la ejecución incluida hubo **ocho llamadas al modelo**: tres para Ana, tres para Juan y dos para el cliente inexistente. No se realizan solicitudes de embeddings.

La consola muestra cada pregunta y la respuesta final. Si se produce un error de configuración, API, persistencia, escritura o límite de recursión, el programa termina con código `1` y un mensaje controlado que no imprime el detalle de la excepción ni credenciales. El mensaje de demostración finalizada se emite sólo al completar la ejecución y guardar la traza.

## Traza de ejecución

El archivo [traza_ejecucion.json](traza_ejecucion.json) contiene tres bloques:

```text
turno_1_multi_paso
turno_2_memoria
prueba_error_y_aclaracion
```

Se guardan tipo y texto de los mensajes, nombres y argumentos de herramientas e IDs que vinculan cada llamada con su resultado. Los IDs son identificadores de llamadas, no API keys. El segundo bloque conserva únicamente los mensajes nuevos del segundo turno; el historial completo sigue disponible en SQLite.

La traza registra eventos observables, no razonamiento interno del modelo. Se serializan los campos necesarios, sin volcar metadatos completos del proveedor. Contiene nombres y pedidos ficticios del ejemplo.

El JSON se escribe después de completar los tres escenarios. Primero se prepara un archivo temporal y luego se reemplaza la evidencia anterior. Si falla la ejecución antes de guardarlo, la traza anterior permanece y no representa la corrida fallida. SQLite puede conservar checkpoints de esa corrida interrumpida.

**Se versiona la traza JSON.** `.env`, `.venv/`, cachés, `checkpoints.sqlite` y sus archivos auxiliares quedan excluidos de Git. Cada ejecución exitosa reemplaza la traza por la más reciente.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

La suite contiene **38 tests** y se ejecuta sin llamadas a APIs. Usa modelo simulado, herramientas reales, grafo real y bases SQLite temporales para comprobar:

- Configuración, protección de credenciales y exclusiones de Git.
- Resultados de herramientas, normalización y argumentos inválidos.
- Parámetros de OpenAI y vinculación de herramientas mediante mocks.
- Ciclo con dos llamadas dependientes, retorno de errores y límite de recursión.
- Contexto entre turnos, aislamiento de threads y recuperación tras reabrir SQLite.
- Serialización de la traza, correspondencia de IDs, ausencia de mensajes repetidos y conservación de la evidencia anterior ante fallos.

Los tests con modelo simulado comprueban el recorrido del programa y el contexto enviado; no demuestran por sí solos que un LLM tomará esas decisiones. La traza incluida aporta la evidencia de una ejecución real que cumplió los tres escenarios. El texto de las respuestas y los IDs pueden variar entre ejecuciones.

## Ejemplo de ejecución real

Salida obtenida con OpenAI y esfuerzo de razonamiento omitido:

```text
=== turno_1_multi_paso ===
Usuario: ¿Cuántos pedidos tuvo Ana García y cuál fue el total?
Asistente: Ana García tuvo **3 pedidos**, por un total de **$14.500**.

=== turno_2_memoria ===
Usuario: ¿Y Juan Pérez?
Asistente: Juan Pérez tuvo **1 pedido**, por un total de **$3.200**.

=== prueba_error_y_aclaracion ===
Usuario: ¿Cuántos pedidos tuvo el cliente Roberto Sánchez?
Asistente: No encontré un cliente registrado con el nombre «Roberto Sánchez». ¿Podés confirmar si está completo y bien escrito?

Traza guardada en traza_ejecucion.json.
Demostración finalizada.
```

### Ejemplo de traza: primer turno

El siguiente bloque reproduce los mensajes del primer escenario de la evidencia incluida:

```json
[
  {
    "tipo": "HumanMessage",
    "contenido": "¿Cuántos pedidos tuvo Ana García y cuál fue el total?"
  },
  {
    "tipo": "AIMessage",
    "contenido": "",
    "tool_calls": [
      {
        "id": "call_rhckcxVzqfNdZR3guFNW3YwT",
        "nombre": "buscar_cliente_por_nombre",
        "argumentos": {
          "nombre": "Ana García"
        }
      }
    ]
  },
  {
    "tipo": "ToolMessage",
    "contenido": "Cliente encontrado: 'Ana García' → cliente_id=102",
    "herramienta": "buscar_cliente_por_nombre",
    "tool_call_id": "call_rhckcxVzqfNdZR3guFNW3YwT"
  },
  {
    "tipo": "AIMessage",
    "contenido": "",
    "tool_calls": [
      {
        "id": "call_kw60X9lnARcwYZObXXy7iMOH",
        "nombre": "buscar_pedidos",
        "argumentos": {
          "cliente_id": 102
        }
      }
    ]
  },
  {
    "tipo": "ToolMessage",
    "contenido": "pedidos=3, total=$14500",
    "herramienta": "buscar_pedidos",
    "tool_call_id": "call_kw60X9lnARcwYZObXXy7iMOH"
  },
  {
    "tipo": "AIMessage",
    "contenido": "Ana García tuvo **3 pedidos**, por un total de **$14.500**."
  }
]
```
