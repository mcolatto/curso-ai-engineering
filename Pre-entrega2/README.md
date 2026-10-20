# Pre-entrega 2: Pipeline de procesamiento validado

## Objetivo

Extraer entidades técnicas de un párrafo, una descripción de arquitectura o un log de error y devolver un objeto validado con Pydantic. El resultado contiene:

- `tecnologias`: tecnologías mencionadas explícitamente; puede ser una lista vacía.
- `nivel_de_criticidad`: `baja`, `media` o `alta`.
- `resumen_tecnico`: resumen del texto, sin inventar información.

El pipeline utiliza `ChatPromptTemplate`, `ChatOpenAI` y salida estructurada mediante `with_structured_output(EntidadesTecnicas)`. Los pasos se componen con LCEL y se ejecutan con `.ainvoke()` desde la función asíncrona `process_text(text: str) -> EntidadesTecnicas`.

El prompt asigna criticidad baja cuando no hay incidencias explícitas, media ante degradación y alta ante caída del servicio, pérdida de datos o riesgo de seguridad. Ante varias señales se elige la mayor criticidad. Un texto ambiguo puede producir un objeto válido: no necesita mencionar tecnologías para ser procesado.

## Estructura

```text
Pre-entrega2/
├── README.md              # Instalación, configuración, ejecución y comprobaciones.
├── requirements.txt       # Versiones de las dependencias.
├── .gitignore             # Exclusión de credenciales, entorno y archivos locales.
├── .env.example           # Plantilla de configuración sin credenciales reales.
├── schemas.py             # Contrato de salida y validadores Pydantic.
├── chain.py               # Prompt, modelo, cadena LCEL, reintentos y logs.
├── main.py                # Demostración asíncrona con dos textos.
└── tests/
    ├── test_schemas.py    # Validación de campos y exportación JSON.
    ├── test_chain.py      # Prompt, procesamiento asíncrono y resiliencia simulada.
    └── test_main.py       # Salida de la demostración y manejo de errores.
```

`.env` y `.venv/` son locales y no se versionan.

## Instalación

Verificado con Python 3.13. Desde la raíz del repositorio, en macOS o Linux:

```bash
cd Pre-entrega2
python3.13 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install -r requirements.txt
```

Si el entorno ya existe y tiene las dependencias instaladas, basta con activarlo. Los tests utilizan `unittest` y `unittest.mock`, incluidos en Python.

## Variables de entorno

Desde `Pre-entrega2/`, crear la configuración local:

```bash
cp .env.example .env
```

Si `.env` ya existe, editarlo directamente para conservar sus valores.

| Variable | Responsabilidad | Valor del ejemplo |
| --- | --- | --- |
| `OPENAI_API_KEY` | Clave de la API oficial de OpenAI. Obligatoria. | Vacío |
| `OPENAI_MODEL` | Modelo compatible con Structured Outputs mediante JSON Schema. Opcional: si falta o está vacío se usa `gpt-6.1-sol`. La cuenta debe tener acceso al modelo. | `gpt-6.1-sol` |
| `OPENAI_REASONING_EFFORT` | Esfuerzo compatible con el modelo. Dejar vacío para omitirlo. Si la variable no existe, se usa `low` para `gpt-6.1-sol` y se omite para otros modelos. | `low` |
| `LLM_MAX_TOKENS` | Límite positivo de tokens de salida; incluye razonamiento y texto visible cuando corresponde. | `1024` |
| `LLM_TIMEOUT_SECONDS` | Timeout de red del cliente, positivo y finito, en segundos. | `30` |

La cuenta debe disponer de créditos y acceso al modelo seleccionado. Crear una API key no garantiza saldo. No se envía temperatura al modelo.

`main.py` carga el `.env` ubicado junto al script. Las variables ya definidas en la terminal tienen prioridad. Al utilizar `process_text()` desde otro programa, el llamador debe cargar o definir las variables antes de invocarla. Importar los módulos no crea clientes ni ejecuta solicitudes.

Solo `OPENAI_API_KEY` es obligatoria; las demás variables tienen valores predeterminados. Las validaciones de configuración comprueban presencia de clave y límites numéricos; la autenticidad de la clave y la compatibilidad del modelo se comprueban al realizar la solicitud. No compartir ni versionar `.env`.

## Ejecución

Desde `Pre-entrega2/`:

```bash
.venv/bin/python main.py
```

La demostración procesa secuencialmente:

1. **Texto completo:** caída de una API con FastAPI, Redis y PostgreSQL.
2. **Texto ambiguo:** demoras sin métricas ni identificación del componente involucrado.

Cada caso muestra su nombre, logs de procesamiento y el JSON validado o un error controlado. Si un caso falla de forma controlada, se continúa con el siguiente.

**Consumo:** cada texto puede realizar hasta dos solicitudes. La demostración completa realiza normalmente dos solicitudes si ambos casos funcionan en el primer intento y hasta cuatro si ambos necesitan reintentar. Las llamadas reales consumen cuota o créditos.

### Ejemplos de salida JSON

Resultados de una prueba real de ambos casos. La redacción puede variar entre ejecuciones.

Texto completo:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "La API utiliza FastAPI, Redis como caché y PostgreSQL para persistencia. El agotamiento del pool de conexiones a PostgreSQL causó la caída del servicio; todas las solicitudes devuelven error."
}
```

Texto ambiguo:

```json
{
  "tecnologias": [],
  "nivel_de_criticidad": "media",
  "resumen_tecnico": "Se reportan demoras intermitentes en el sistema, sin métricas disponibles ni identificación del componente involucrado."
}
```

Pydantic exige los tres campos, una criticidad reconocida y un resumen con contenido real. La lista de tecnologías puede estar vacía, pero ningún elemento puede ser vacío o contener solo espacios. Se eliminan los espacios exteriores de cada tecnología; el resumen válido se conserva sin recortarlo. El esquema valida estructura y restricciones; no demuestra por sí mismo que la extracción sea semánticamente correcta.

## Reintentos y logs

La cadena se construye como `(prompt | model.with_structured_output(EntidadesTecnicas)).with_retry(...)`. LangChain valida directamente la salida con Pydantic, sin un parser manual posterior. Las instrucciones de formato del prompt se generan a partir de `EntidadesTecnicas.model_json_schema()`.

`.with_retry()` permite **dos intentos totales por texto**, es decir, un reintento. Cada intento vuelve a ejecutar la generación y la validación; no se limita a validar otra vez la misma respuesta.

Se reintentan errores de interpretación o validación, campos faltantes y respuestas incompletas. El SDK detecta `finish_reason="length"` antes de interpretar el JSON y lanza `LengthFinishReasonError`, que está incluida en los errores reintentables. Los reintentos automáticos propios del cliente están desactivados para no multiplicar solicitudes.

No se reintentan errores de configuración, autenticación, permisos, conexión, timeout, cuota ni rechazos del modelo. Un reintento tampoco garantiza una respuesta válida: por ejemplo, un límite de tokens insuficiente puede afectar ambos intentos.

Los logs registran el inicio del procesamiento con el máximo de intentos permitido, la validación correcta o el fallo final. Los intentos individuales son administrados por `.with_retry()`. No incluyen claves, texto analizado ni respuestas completas; la demostración imprime el JSON final por separado.

Si la primera respuesta es válida, se observa:

```text
INFO chain: Inicio del procesamiento: máximo dos intentos por texto
INFO chain: Procesamiento finalizado: validación correcta
```

Si ambas respuestas son inválidas o incompletas, se informa el agotamiento y `main.py` muestra:

```text
Error: no se obtuvo una respuesta válida tras los dos intentos.
```

## Tests y verificación

Desde `Pre-entrega2/`:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

La suite contiene **33 tests**, comprobados en Python 3.13. No requiere credenciales reales ni realiza llamadas a APIs.

Las pruebas simuladas cubren validación de esquemas, construcción del prompt, cadena asíncrona, primer intento inválido seguido de uno válido, agotamiento, JSON mal formado, campos faltantes, truncamiento, errores no reintentables y logs sin contenido sensible. En los tests de la cadena se mantienen reales el prompt, la composición LCEL y la validación, simulando la frontera del modelo. Las esperas del reintento se desactivan únicamente durante los tests.

También se comprueba la demostración: salida JSON, carga de configuración, mensajes de error y continuación del segundo caso si falla el primero.

**Verificación real de los ejemplos:** los JSON mostrados proceden de ejecuciones contra OpenAI que respondieron correctamente para ambos textos. El ambiguo produjo `tecnologias=[]`. Estos ejemplos no demuestran por sí solos el comportamiento de reintento: los escenarios de reintento y truncamiento se comprueban mediante simulación. La suite de 33 tests verifica la implementación actual sin realizar solicitudes pagas.
