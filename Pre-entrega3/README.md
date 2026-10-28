# Pre-entrega 3: Sistema de recuperación semántica local (RAG)

## Objetivo

Implementar un flujo RAG que reciba una pregunta, recupere fragmentos de una colección persistente en ChromaDB y genere una respuesta basada exclusivamente en ese contexto. La salida se valida con Pydantic y contiene la respuesta, los archivos consultados y la cantidad de fragmentos recuperados.

Los documentos de ejemplo abordan curación de datos, generación aumentada por recuperación y bases de datos vectoriales. ChromaDB se ejecuta localmente; los embeddings y la generación utilizan las APIs de OpenAI.

## Componentes y comprobación

| Componente | Implementación | Comprobación |
| --- | --- | --- |
| Ingesta y chunking | Lectura de `.txt` y `.md`, fragmentación en tokens y conservación de metadatos. | Tests locales y corpus real de 3 archivos, dividido en 12 fragmentos. |
| Persistencia | Colección ChromaDB en `vectorstore/`, reutilizada cuando ya está poblada. | Tests con Chroma temporal y segunda ingesta real sin generar embeddings nuevos. |
| Recuperación asíncrona | Retriever de similitud con `k=4` y `.ainvoke()`. | Tests de parámetros, metadatos y avance del event loop durante una consulta pendiente. |
| Generación fundamentada | Cadena LCEL `prompt | model | parser`. | Tres preguntas respondibles y tres preguntas trampa ejecutadas contra OpenAI. |
| Salida validada | `PydanticOutputParser` para `RespuestaLLM`; ensamblado de `RAGResponse` desde metadatos reales. | Tests de validación, fuentes y salida JSON. |

## Estructura

```text
Pre-entrega3/
├── README.md              # Instalación, ejecución y ejemplos reales.
├── requirements.txt       # Versiones de las dependencias directas.
├── .env.example           # Plantilla de configuración sin claves reales.
├── .gitignore             # Exclusión del entorno, secretos y base vectorial.
├── config.py              # Parámetros validados y rutas compartidas.
├── schemas.py             # RespuestaLLM y RAGResponse.
├── ingest.py              # Carga, chunking, creación y reutilización del índice.
├── chain.py               # Recuperación asíncrona, prompt, LLM y parser.
├── main.py                # Demostración con seis preguntas.
├── data/
│   ├── datasets.txt        # Curación y gestión de datos.
│   ├── rag.txt             # Conceptos y beneficios de RAG.
│   └── vectordb.txt        # Vectores, embeddings y bases vectoriales.
└── tests/
    ├── test_config.py      # Configuración y protección de credenciales.
    ├── test_schemas.py     # Contratos de salida.
    ├── test_ingest.py      # Lectura, fragmentación y persistencia local.
    ├── test_chain.py       # Recuperación y generación simuladas.
    └── test_main.py        # Recorrido de los seis casos y errores controlados.
```

`.env`, `.venv/`, `__pycache__/` y `vectorstore/` no se versionan. El índice se construye localmente a partir de los archivos incluidos en `data/`.

## Instalación

Verificado con Python 3.13. Desde la raíz del repositorio, en macOS o Linux:

```bash
cd Pre-entrega3
python3.13 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install -r requirements.txt
python -m pip check
```

Si el entorno ya existe, activarlo sin volver a crearlo. La primera utilización de `tiktoken` puede descargar el vocabulario público `cl100k_base` y guardarlo en caché; esa descarga no utiliza la API key ni consume créditos de OpenAI.

## Variables de entorno

Desde `Pre-entrega3/`:

```bash
cp .env.example .env
```

Si `.env` ya existe, editarlo directamente para conservar su configuración.

| Variable | Uso | Predeterminado |
| --- | --- | --- |
| `OPENAI_API_KEY` | Clave de OpenAI. Es la única variable obligatoria. | Sin valor |
| `OPENAI_MODEL` | Modelo de generación accesible para la cuenta. Si falta o está vacío, se usa el predeterminado. | `gpt-6.1-sol` |
| `OPENAI_EMBEDDING_MODEL` | Mismo modelo para indexar documentos y consultar. Si falta o está vacío, se usa el predeterminado. | `text-embedding-3-small` |
| `OPENAI_REASONING_EFFORT` | Esfuerzo compatible con el modelo. Vacío omite el parámetro. Si no existe, se usa `low` para el modelo de generación predeterminado y se omite para otros. | `low` en `.env.example` |
| `LLM_MAX_TOKENS` | Límite positivo de tokens de salida del LLM, incluido el razonamiento cuando corresponde. | `1024` |
| `LLM_TIMEOUT_SECONDS` | Timeout de los clientes, positivo y finito, en segundos. | `30` |

La cuenta debe tener créditos y acceso a los modelos. Crear una clave no garantiza saldo ni permisos. No se envía temperatura al modelo de generación.

Los scripts cargan el `.env` situado junto a ellos; las variables ya definidas en la terminal tienen prioridad. Si se importan las funciones desde otro programa, el llamador debe cargar o definir las variables antes de utilizarlas. Importar los módulos no ejecuta ingesta ni consultas.

Las rutas se calculan desde `config.py`, no desde el directorio de la terminal. La colección se llama `preentrega3_documentos` y utiliza la carpeta `vectorstore/`.

## Ingesta y persistencia

Preparar el índice antes de consultar:

```bash
.venv/bin/python ingest.py
```

El script lee archivos `.txt` y `.md` no vacíos directamente de `data/`, en UTF-8 y en orden estable. Usa `RecursiveCharacterTextSplitter.from_tiktoken_encoder()` con `cl100k_base`, tamaño máximo de 500 tokens y solapamiento configurado en 50. Los fragmentos pueden ser más cortos; no se rellenan hasta alcanzar 500 tokens. El solapamiento efectivo depende de los cortes que preservan las separaciones del texto.

Cada fragmento conserva `source` con el nombre del archivo de origen. En la primera ingesta se generan embeddings y se almacenan los fragmentos en ChromaDB. En las siguientes, se comprueba la colección y su modelo de embeddings para reutilizarla sin volver a leer ni indexar el corpus.

Para reconstruir explícitamente:

```bash
.venv/bin/python ingest.py --rebuild
```

Esta opción reemplaza solamente la colección del proyecto y vuelve a generar embeddings. Utilizarla si cambian los documentos o el modelo de embeddings; los cambios del corpus no se detectan automáticamente. Otras colecciones de la misma base no se eliminan. Si el modelo configurado no coincide con el registrado, la apertura se rechaza hasta reconstruir.

## Ejecución de las consultas

Después de la ingesta:

```bash
.venv/bin/python main.py
```

La demostración ejecuta seis preguntas secuencialmente: tres con respuesta en el corpus y tres fuera de él. Muestra el nombre del caso, la pregunta y el JSON resultante. Los errores controlados se informan sin imprimir excepciones completas y permiten continuar con los casos siguientes.

La función pública es:

```python
async get_rag_response(query: str) -> RAGResponse
```

Su recorrido es:

1. Validar que la pregunta tenga contenido.
2. Abrir una colección poblada y compatible, sin iniciar ingesta automática.
3. Recuperar hasta cuatro fragmentos por similitud mediante `retriever.ainvoke()`.
4. Formatear el contexto y ejecutar `prompt | model | PydanticOutputParser` con `.ainvoke()`.
5. Construir la respuesta final con metadatos de los documentos recuperados.

La apertura síncrona del índice se ejecuta mediante `asyncio.to_thread()`. La integración instalada de Chroma deriva la búsqueda a un executor; la llamada al LLM también se espera de forma asíncrona.

### Contratos de salida y fuentes

`RespuestaLLM` contiene únicamente `respuesta`. El modelo debe producir JSON y el parser valida ese campo como texto no vacío. El prompt exige responder exclusivamente desde el contexto y devolver exactamente `"No lo sé"` si este es insuficiente.

`RAGResponse` combina esa respuesta con:

- `fuentes`: nombres únicos y ordenados de los archivos proporcionados como contexto, obtenidos por código desde sus metadatos.
- `fragmentos_recuperados`: cantidad de fragmentos recibidos del retriever, que puede ser mayor que la cantidad de archivos distintos.

**Las fuentes identifican el contexto consultado, no son citas elegidas por el LLM ni atribuyen cada afirmación a un archivo concreto.** Se conservan incluso si el modelo responde `"No lo sé"`. Si la búsqueda no devuelve fragmentos, se entrega `"No lo sé"`, `fuentes=[]` y contador `0`, sin llamar al LLM.

Pydantic valida la estructura y los valores mínimos; no verifica por sí mismo la veracidad de una respuesta. La fundamentación se comprueba contrastando el resultado con el contexto recuperado.

### Errores y consumo

Se controlan en la demostración errores de configuración o índice, autenticación, permisos, cuota, conexión, timeout y formato de salida. No hay reintentos automáticos del LLM ni del cliente de embeddings. Un JSON inválido produce un error de parseo; no se reemplaza por una respuesta inventada.

La primera ingesta consume embeddings documentales, agrupados según el cliente y el tamaño del corpus. Reutilizar el índice no genera nuevos embeddings de documentos. Cada pregunta realiza una solicitud de embedding de consulta y, si recupera contexto, una llamada al LLM: la demostración completa puede efectuar seis solicitudes de embeddings y seis de generación. Las llamadas reales consumen cuota o créditos.

## Tests

Desde `Pre-entrega3/`:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

La suite contiene **61 tests**, comprobados en Python 3.13. No necesita claves reales ni realiza llamadas a APIs de OpenAI. El tokenizer debe tener disponible su vocabulario local o descargarlo en su primer uso.

Se verifican configuración y ocultamiento de claves; esquemas; carga y chunking; persistencia y reconstrucción con Chroma real temporal y embeddings simulados; recuperación; errores; asincronía; prompt; parser Pydantic real y ensamblado de fuentes. Las respuestas del LLM se simulan para probar el recorrido sin consumo. Esas simulaciones no demuestran calidad de recuperación semántica ni grounding real.

Las comprobaciones con servicios reales se documentan a continuación: ingesta, reutilización y las seis preguntas ejecutadas contra OpenAI. Los resultados muestran el comportamiento de esa ejecución; la redacción y los fragmentos seleccionados pueden variar entre ejecuciones.

## Ejemplos de ejecución real

### Primera ingesta

```text
$ .venv/bin/python ingest.py
INFO __main__: Documentos cargados: 3; fragmentos generados: 12
INFO httpx2: HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
INFO __main__: Índice creado: 12 fragmentos
```

### Reutilización sin reindexar

```text
$ .venv/bin/python ingest.py
INFO __main__: Índice existente reutilizado: 12 fragmentos
```

### 1. Pregunta respondible: beneficios de RAG

**Pregunta:** ¿Cuáles son los beneficios de la RAG?

```json
{
  "respuesta": "Los principales beneficios de la RAG son:\n- Implementación y escalado de IA rentables: permite usar fuentes de datos internas y autorizadas sin volver a entrenar el modelo, mitigando costos y requisitos de recursos.\n- Acceso a datos actuales y específicos del dominio: conecta los modelos con información externa actualizada y relevante.\n- Menor riesgo de alucinaciones de la IA.\n- Mayor confianza de los usuarios: las citas permiten verificar la precisión de los resultados y consultar los materiales originales.\n- Casos de uso ampliados: permite integrar información de múltiples fuentes y responder consultas más complejas.\n- Control mejorado del desarrollador y mantenimiento de modelos: las fuentes de conocimiento externas pueden modificarse para adaptar el modelo a distintas tareas.\n- Mayor seguridad de los datos: mantiene separados el modelo y el conocimiento externo, y permite revocar el acceso a los datos. No obstante, las bases de datos externas también deben protegerse.",
  "fuentes": ["rag.txt"],
  "fragmentos_recuperados": 4
}
```

### 2. Pregunta respondible: curación de datos

**Pregunta:** ¿Qué es la curación de datos?

```json
{
  "respuesta": "La curación de datos es el proceso de crear y gestionar conjuntos de datos para que las personas puedan encontrarlos, acceder a ellos, usarlos y reutilizarlos. Incluye reunir activos de datos en un repositorio central, consolidar sus metadatos, enriquecerlos con información adicional y analizar y mejorar su calidad durante todo su ciclo de vida. Puede realizarse manualmente o con ayuda de software de automatización.",
  "fuentes": ["datasets.txt", "rag.txt"],
  "fragmentos_recuperados": 4
}
```

### 3. Pregunta respondible: vectores

**Pregunta:** ¿Qué son los vectores?

```json
{
  "respuesta": "Los vectores son tensores unidimensionales que contienen múltiples escalares del mismo tipo de datos; es decir, organizan números de manera estructurada. Cada componente escalar representa una característica o dimensión del vector. Por ejemplo, las temperaturas mínima, media y máxima de un día pueden representarse mediante el vector [62, 77, 85].",
  "fuentes": ["vectordb.txt"],
  "fragmentos_recuperados": 4
}
```

### 4. Pregunta trampa: Pinecone

**Pregunta:** ¿Qué es Pinecone?

```json
{
  "respuesta": "No lo sé",
  "fuentes": ["rag.txt", "vectordb.txt"],
  "fragmentos_recuperados": 4
}
```

### 5. Pregunta trampa: índices de Pinecone

**Pregunta:** ¿Qué algoritmo utiliza Pinecone para distribuir sus índices entre servidores?

```json
{
  "respuesta": "No lo sé",
  "fuentes": ["rag.txt", "vectordb.txt"],
  "fragmentos_recuperados": 4
}
```

### 6. Pregunta trampa: HNSW

**Pregunta:** ¿Qué es HNSW?

```json
{
  "respuesta": "No lo sé",
  "fuentes": ["rag.txt"],
  "fragmentos_recuperados": 4
}
```

En las seis consultas se registraron respuestas HTTP 200 para embeddings y generación. Los logs mostraron cuatro fragmentos recuperados por pregunta y una respuesta validada. Las tres preguntas trampa produjeron `"No lo sé"` pese a recibir contexto: recuperar documentos similares no implica que estos contengan la respuesta.
