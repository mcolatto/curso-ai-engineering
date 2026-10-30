# Pre-entrega 4: recuperación híbrida con Pinecone

## Objetivo

Implementar y evaluar un sistema de recuperación híbrida sobre siete documentos Markdown de LangGraph. El flujo combina búsqueda léxica BM25 y búsqueda vectorial en Pinecone Serverless, y devuelve los cinco chunks mejor posicionados después de fusionar ambos rankings mediante `EnsembleRetriever`.

El sistema recupera fragmentos y sus metadatos. No utiliza un LLM para generar respuestas.

## Estructura

```text
Pre-entrega4/
├── config.py              # Variables de entorno, validaciones y constantes compartidas
├── schemas.py             # Contratos Pydantic de metadatos y casos del Golden Set
├── ingest.py              # Carga, fragmentación, embeddings y reemplazo del namespace
├── rag_system.py          # RAGSystem: BM25, Pinecone y fusión de rankings por chunk_id
├── evaluate.py            # Validación del Golden Set, métricas y reporte en consola
├── main.py                # Una consulta exploratoria independiente del Golden Set
├── golden_dataset.json    # Cinco preguntas con una fuente esperada por pregunta
├── data/                  # Siete Markdown originales utilizados por ambas ramas
│   ├── checkpointers.md
│   ├── interrupts.md
│   ├── memory.md
│   ├── persistence.md
│   ├── stores.md
│   ├── subgraphs.md
│   └── time-travel.md
├── tests/                 # Pruebas unittest locales con servicios simulados
├── requirements.txt       # Dependencias con versiones fijadas
├── .env.example           # Plantilla de configuración sin credenciales
├── .gitignore             # Exclusión de secretos, entorno virtual y cachés
└── README.md              # Instalación, uso, evaluación y resultados
```

## Instalación

Requisitos: Python 3.12, una API key de OpenAI con acceso a embeddings y una API key de Pinecone con permisos para crear el índice y leer/escribir sus registros.

Desde la raíz del repositorio:

```bash
cd Pre-entrega4
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
cp .env.example .env
```

Completar las claves en `.env`. Este archivo y `.venv/` están excluidos de Git. Los comandos siguientes se ejecutan desde `Pre-entrega4/`.

## Configuración

| Variable | Predeterminado | Uso |
| --- | --- | --- |
| `OPENAI_API_KEY` | Sin valor; obligatoria | Generar embeddings de documentos y consultas |
| `PINECONE_API_KEY` | Sin valor; obligatoria | Acceder a Pinecone |
| `INDEX_NAME` | `preentrega4-rag-hibrido` | Nombre del índice |
| `NAMESPACE` | `langgraph-docs` | Namespace exclusivo de este corpus; no puede estar vacío |
| `PINECONE_CLOUD` | `aws` | Nube Serverless |
| `PINECONE_REGION` | `us-east-1` | Región Serverless |
| `CHUNK_SIZE` | `600` | Tamaño máximo del chunk medido en tokens |
| `CHUNK_OVERLAP` | `100` | Solapamiento objetivo entre chunks; menor que el tamaño |
| `BM25_WEIGHT` | `0.5` | Peso de la rama léxica |
| `VECTOR_WEIGHT` | `0.5` | Peso de la rama vectorial |

Los pesos deben ser finitos, no negativos y sumar 1. El tamaño debe ser positivo y el overlap no negativo. La configuración rechaza claves vacías; su autenticidad se comprueba cuando se utilizan los servicios.

Las constantes compartidas son `text-embedding-3-small`, dimensión **1536**, tokenizer `cl100k_base` y **Top-5**. El modelo y la dimensión son los mismos para documentos y preguntas. El índice utiliza métrica coseno.

## Ingesta

```bash
.venv/bin/python ingest.py
```

El script carga exclusivamente los `.md` de `data/` en UTF-8 y en orden estable. Usa `RecursiveCharacterTextSplitter.from_tiktoken_encoder()` y conserva los nombres originales. La categoría se obtiene del nombre del archivo sin extensión.

Cada vector contiene un ID determinista calculado a partir de la fuente, la posición del fragmento y su contenido. Sus metadatos, validados con Pydantic, incluyen:

| Campo | Contenido |
| --- | --- |
| `text` | Texto original del chunk |
| `source` | Nombre del archivo, por ejemplo `memory.md` |
| `categoria` | Nombre sin extensión, por ejemplo `memory` |
| `chunk_id` | Identificador determinista del chunk |

El texto se almacena dentro de los metadatos de Pinecone. `PineconeVectorStore`, configurado con `text_key="text"`, lo reconstruye sin consultar otra base de datos.

Antes de borrar registros, se preparan y validan todos los chunks y embeddings. El índice se crea si falta; si existe, se comprueban dimensión, métrica, nube y región. Un índice incompatible produce un error y no se recrea automáticamente.

**Cada ingesta reemplaza exclusivamente el namespace configurado:** elimina sus registros anteriores, espera el vaciado, sube lotes de hasta 100 registros y verifica la cantidad y el conjunto de IDs. No borra otros namespaces. Las esperas de disponibilidad tienen un límite de 120 segundos por etapa. Ante un fallo posterior al borrado puede quedar una ingesta parcial; repetir el comando permite reconstruir el namespace.

Ejemplo de ejecución real, omitiendo los logs HTTP:

```text
INFO __main__: Fragmentos preparados: 136
INFO __main__: Ingesta completa: 136 registros verificados
```

## Recuperación híbrida y consultas exploratorias

BM25 utiliza los mismos chunks locales que se ingieren en Pinecone y recupera hasta cinco candidatos. La rama vectorial recupera hasta cinco del namespace configurado. `EnsembleRetriever` combina sus posiciones mediante Reciprocal Rank Fusion con los pesos indicados.

La identidad de fusión es `chunk_id`: un mismo chunk encontrado por ambas ramas aparece una sola vez y recibe el aporte de ambas. Distintos chunks de un mismo archivo se conservan. El corte final a cinco se aplica **después** de combinar los rankings.

Antes de recuperar resultados se comprueba que la cantidad y los IDs del namespace coincidan con el corpus local fragmentado. Si difieren, se requiere ejecutar nuevamente la ingesta.

Consulta por argumento:

```bash
.venv/bin/python main.py "How can an agent retain context during a conversation and remember information across different sessions?"
```

O ingresar una pregunta por consola:

```bash
.venv/bin/python main.py
```

Cada ejecución procesa una consulta y muestra posición, fuente, categoría, ID y un extracto de hasta 200 caracteres por resultado. No carga ni modifica el Golden Set.

## Evaluación y métricas

```bash
.venv/bin/python evaluate.py
```

El Golden Set debe contener exactamente cinco casos válidos y cada fuente esperada debe existir en el corpus:

| Pregunta | Fuente esperada |
| --- | --- |
| What is Persistence in LangGraph? | `persistence.md` |
| What is the definition of StateSnapshot in checkpointers? | `checkpointers.md` |
| How can I pause a workflow to request human approval before continuing execution? | `interrupts.md` |
| How can an agent retain context during a conversation and remember information across different sessions? | `memory.md` |
| Which operations allow LangGraph to replay an earlier execution or explore a different outcome? | `time-travel.md` |

Se evalúan las fuentes de los chunks del ranking combinado, **sin agruparlos por archivo**:

- **Precision@5 docente:** cantidad de chunks cuya fuente coincide con la esperada / `len(top_k)`. Si no hay resultados, vale `0.0`. El denominador es la cantidad efectivamente recuperada: una coincidencia entre dos resultados produce `0.5`.
- **Recall@5:** vale `1.0` si aparece la fuente esperada y `0.0` si no aparece. En este Golden Set equivale a **Hit@5**, porque cada pregunta tiene una única fuente esperada. No mide la proporción de todos los chunks relevantes del corpus.
- **Promedios:** media aritmética de cada métrica sobre las cinco preguntas.

Un error operativo interrumpe la evaluación y no se transforma en una recuperación vacía ni en un promedio parcial.

### Resultados reales de la configuración inicial

Configuración: tamaño **600**, overlap **100**, pesos **[0.5, 0.5]**, Top-5 y **136 chunks** sobre siete documentos. No se aplicaron ajustes de parámetros.

| Caso | Precision@5 | Recall@5 |
| --- | ---: | ---: |
| Persistence | 0.20 | 1.00 |
| StateSnapshot | 1.00 | 1.00 |
| Aprobación humana | 1.00 | 1.00 |
| Memoria | 0.20 | 1.00 |
| Time travel | 0.20 | 1.00 |
| **Promedio** | **0.52** | **1.00** |

### Interpretación y limitaciones

Los documentos comparten conceptos. Por ejemplo, `subgraphs.md` explica cómo conservar interacciones anteriores de un subagente y `persistence.md` describe memoria de corto y largo plazo. Por eso, un chunk de esos archivos puede aportar información pertinente a la pregunta de memoria aunque no pertenezca a `memory.md`.

La métrica cuenta exclusivamente la fuente etiquetada en el Golden Set. Por lo tanto, una Precision@5 baja no demuestra por sí sola que todos los otros chunks sean irrelevantes. A su vez, Recall@5 igual a 1 indica presencia de la fuente esperada, no que se haya recuperado toda la información necesaria ni que esa fuente ocupe el primer lugar. Los cinco casos constituyen una muestra pequeña.

La fragmentación por tamaño de tokens puede cortar ejemplos de código o separar encabezados y explicaciones. Además, el extracto de 200 caracteres es una vista parcial: el contenido pertinente puede aparecer más adelante dentro del mismo chunk. La coincidencia de IDs y cantidad comprueba el corpus esperado, pero no reemplaza una auditoría del contenido remoto ni de cómo se generaron sus vectores.

Cambiar tamaño u overlap requiere repetir la ingesta para mantener ambas ramas sobre el mismo corpus. Cambiar únicamente pesos no requiere reingesta. Los resultados pueden variar si cambian corpus, dependencias, modelo o configuración; los empates de ranking también pueden influir en el orden.

## Tests y consumo

```bash
.venv/bin/python -m unittest discover -s tests -v
```

La suite contiene **65 tests** y se ejecuta sin llamadas a OpenAI ni Pinecone. Cubre configuración, esquemas, fragmentación, metadatos, ingesta simulada, aislamiento del namespace, reingesta, timeouts, fusión por ID, consultas, métricas y errores operativos. Usa implementaciones reales de LangChain para comprobar la fusión y pruebas de integración local de BM25, con las fronteras de los servicios simuladas.

Las pruebas reales realizadas corresponden a la ingesta de los 136 chunks, la evaluación de las cinco preguntas y la consulta exploratoria de memoria. Los errores y timeouts de servicios se verifican mediante mocks.

Cada ejecución de `ingest.py` vuelve a generar los embeddings documentales y opera contra Pinecone. `evaluate.py` genera un embedding y realiza una recuperación vectorial por pregunta: cinco consultas para el Golden Set completo. `main.py` realiza una consulta. También se hacen operaciones de control y verificación sobre Pinecone. Estas ejecuciones consumen la cuota o el saldo de los servicios; BM25 funciona localmente. No se generan respuestas de chat.

## Ejemplos de salida reales

Se omiten los logs HTTP. Los ejemplos siguientes corresponden a los resultados obtenidos con la configuración inicial.

### Evaluación del Golden Set

```text
=== Evaluación del Golden Set: Top-5 híbrido ===
Precision@5: coincidencias / resultados recuperados; vacío = 0.
Recall@5: Hit@5, una única fuente esperada por pregunta.

1. What is Persistence in LangGraph?
Fuente esperada: persistence.md
Fuentes recuperadas: ['persistence.md', 'checkpointers.md', 'checkpointers.md', 'subgraphs.md', 'checkpointers.md']
Precision@5: 0.2000
Recall@5: 1.0000

2. What is the definition of StateSnapshot in checkpointers?
Fuente esperada: checkpointers.md
Fuentes recuperadas: ['checkpointers.md', 'checkpointers.md', 'checkpointers.md', 'checkpointers.md', 'checkpointers.md']
Precision@5: 1.0000
Recall@5: 1.0000

3. How can I pause a workflow to request human approval before continuing execution?
Fuente esperada: interrupts.md
Fuentes recuperadas: ['interrupts.md', 'interrupts.md', 'interrupts.md', 'interrupts.md', 'interrupts.md']
Precision@5: 1.0000
Recall@5: 1.0000

4. How can an agent retain context during a conversation and remember information across different sessions?
Fuente esperada: memory.md
Fuentes recuperadas: ['subgraphs.md', 'subgraphs.md', 'persistence.md', 'memory.md', 'subgraphs.md']
Precision@5: 0.2000
Recall@5: 1.0000

5. Which operations allow LangGraph to replay an earlier execution or explore a different outcome?
Fuente esperada: time-travel.md
Fuentes recuperadas: ['time-travel.md', 'checkpointers.md', 'checkpointers.md', 'interrupts.md', 'interrupts.md']
Precision@5: 0.2000
Recall@5: 1.0000

=== Promedios ===
Precision@5: 0.5200
Recall@5: 1.0000
```

### Consulta exploratoria de memoria

Pregunta: `How can an agent retain context during a conversation and remember information across different sessions?`

Los cinco IDs son diferentes: la repetición de `subgraphs.md` corresponde a distintos chunks.

````text
=== Top-5 híbrido ===

1. Fuente: subgraphs.md
Categoría: subgraphs
ID: 0098d233ec950b2260ee57d5e28c2e69ecf9af43c8895a111a352c6da28b587a
Extracto: 

#### Per-thread

Use per-thread persistence when a subagent needs to remember previous interactions. For example, a research assistant that builds up context over several exchanges, or a coding assi

2. Fuente: subgraphs.md
Categoría: subgraphs
ID: 563fad50238c767cd03fa233259039d95fb524f5566a9af455e6e4c340fd1e21
Extracto: 

# Resume - approve the interrupt
resumed = agent.stream_events(Command(resume=True), config=config, version="v3")
final = resumed.output
```

> **[View example trace](https://smith.langchain.com/pub

3. Fuente: persistence.md
Categoría: persistence
ID: 57a5a4dfbfdb8b300cb0ed48a78cc2771d5c3e99ba8950f0b4a8bd315db64222
Extracto: > ## Documentation Index
> Fetch the complete documentation index at: https://docs.langchain.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Persistence

> La

4. Fuente: memory.md
Categoría: memory
ID: 248196626fd8810e00ce60a325bef643c34bf326d66319d6bbc81be7af387ed2
Extracto: 

## Manage short-term memory

With [short-term memory](#add-short-term-memory) enabled, a checkpointer persists graph state across turns within the same thread. The node that calls the model controls

5. Fuente: subgraphs.md
Categoría: subgraphs
ID: 6515c6963ee47fdb4cd17bdfe3c456cc49fa7db3947df59abe97fe3edc27072b
Extracto: 

# Wrap subagent as a tool for the outer agent
@tool
def ask_fruit_expert(question: str) -> str:
    """Ask the fruit expert. Use for ALL fruit questions."""
    response = fruit_agent.invoke(
      
````
