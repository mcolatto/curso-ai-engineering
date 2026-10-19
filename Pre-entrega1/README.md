# Pre-entrega 1: Cliente de LLM robusto y asíncrono

## Objetivo

> Debes entregar un repositorio de código (o un archivo Python estructurado) que contenga la implementación de un "Unified Async LLM Client".
>
> Este artefacto debe ser una clase (o conjunto de clases) en Python 3.12 que permita:
>
> 1. **Intercambiabilidad:** Poder instanciar un proveedor (OpenAI o Anthropic) bajo una interfaz común.
> 2. **Asincronía:** Todas las llamadas a modelos deben ser no bloqueantes (`async/await`).
> 3. **Streaming:** Implementar un método que devuelva un generador asíncrono de tokens.
> 4. **Validación:** Uso de Pydantic para definir la estructura de los mensajes de entrada y la configuración del modelo.

## Criterios del entregable

Los criterios conservan el texto de la consigna. La columna de evidencia indica dónde se implementan o cómo se comprueban.

| Estado | Criterio | Evidencia |
| --- | --- | --- |
| ✅ | Configura un entorno virtual con Python 3.12 e instala `openai`, `anthropic`, `pydantic` y `python-dotenv`. | Dependencias en `requirements.txt`; suite verificada en un entorno Python 3.12. |
| ✅ | Crea un esquema de Pydantic para validar los parámetros de entrada del modelo (temperatura de 0 a 2, max_tokens, etc.). | `llmclient/schemas.py` y `tests/test_schemas.py`. |
| ✅ | Implementa una clase `AsyncLLMManager` que pueda cargar tanto OpenAI como Anthropic basándose en una variable de configuración. | `llmclient/manager.py`: selección mediante `LLMConfig.provider`. |
| ✅ | El método de generación debe ser capaz de manejar streaming: usa `yield` para retornar fragmentos de texto conforme lleguen de la API. | `generate_stream()` en ambos clientes y en el manager. |
| ✅ | Asegúrate de capturar excepciones de red y de límite de tasa (rate limiting) devolviendo un error controlado en lugar de un crash. | Manejo de excepciones en ambos clientes; pruebas de conexión, timeout y cuota con mocks. |
| ✅ | Incluye un archivo `README.md` que explique cómo ejecutar el script de prueba y qué variables de entorno son necesarias. | Instrucciones y variables documentadas abajo. |

## Estructura del proyecto

```text
Pre-entrega1/
├── README.md                      # Objetivo, configuración, ejecución y ejemplos.
├── requirements.txt               # Dependencias del programa.
├── .env.example                   # Plantilla de configuración sin credenciales.
├── main.py                        # Demostración normal y streaming por proveedor.
├── llmclient/
│   ├── __init__.py                 # Define el paquete del cliente.
│   ├── schemas.py                 # Mensajes, configuración y respuestas con Pydantic.
│   ├── base.py                    # Interfaz abstracta de generación y streaming.
│   ├── manager.py                 # Selecciona el cliente y delega las solicitudes.
│   └── providers/
│       ├── __init__.py             # Define el paquete de proveedores.
│       ├── openai_client.py        # Adaptación del SDK asíncrono oficial de OpenAI.
│       └── anthropic_client.py     # Adaptación del SDK asíncrono oficial de Anthropic.
└── tests/                         # Tests de validación, clientes, manager y demostración.
```

El archivo `.env` y el entorno virtual son locales y no se versionan.

## Instalación

Usar Python 3.12. Desde la raíz del repositorio, en macOS o Linux:

```bash
cd Pre-entrega1
python3.12 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install -r requirements.txt
```

Como alternativa, con Conda y sin otro entorno virtual activo:

```bash
conda create -n preentrega312 python=3.12 pip
conda activate preentrega312
cd Pre-entrega1
python --version
python -m pip install -r requirements.txt
```

Elegir una de las dos alternativas. `python --version` debe mostrar Python 3.12.x.
Los tests usan `unittest` y `unittest.mock`, incluidos en Python.

## Variables de entorno

Desde `Pre-entrega1/`, crear la configuración local:

```bash
cp .env.example .env
```

Si `.env` ya existe, editarlo directamente para conservar su configuración.
Completar las claves y modelos de los proveedores que se utilizarán. Para ejecutar ambos proveedores se necesita, en cada plataforma, una clave válida con créditos disponibles y acceso al identificador de modelo seleccionado. Crear una clave no garantiza saldo ni acceso al modelo.

| Variable | Uso | Valor en `.env.example` |
| --- | --- | --- |
| `OPENAI_API_KEY` | Clave de la API oficial de OpenAI. | Vacío |
| `OPENAI_MODEL` | Identificador del modelo de OpenAI; por ejemplo, `gpt-6.1-sol`. | Vacío |
| `OPENAI_REASONING_EFFORT` | Esfuerzo para OpenAI: `low`, `medium`, `high`, `xhigh` o `max`, según el modelo. Vacío omite el parámetro y deja el comportamiento predeterminado del proveedor. | `low` |
| `ANTHROPIC_API_KEY` | Clave de la API oficial de Anthropic. | Vacío |
| `ANTHROPIC_MODEL` | Identificador exacto de un modelo disponible para la cuenta de Anthropic. | Vacío |
| `LLM_TEMPERATURE` | Temperatura validada entre 0 y 2. | `0.7` |
| `LLM_MAX_TOKENS` | Límite positivo de tokens de salida. En OpenAI se envía como `max_completion_tokens` e incluye razonamiento y texto visible. | `1024` |
| `LLM_TIMEOUT_SECONDS` | Timeout positivo del SDK, expresado en segundos. | `30` |

La temperatura se valida como configuración compartida. No se envía a GPT-6.1 Sol ni cuando se configura esfuerzo de razonamiento en OpenAI. Tampoco se envía al SDK actual de Anthropic. El esfuerzo configurado se aplica únicamente a OpenAI y debe ser compatible con el modelo elegido.

`main.py` carga el `.env` ubicado junto al script. Las variables ya definidas en el entorno de la terminal tienen prioridad sobre ese archivo.
Nunca incluir credenciales reales en `.env.example`, logs o archivos versionados.

## Ejecutar la aplicación

Desde `Pre-entrega1/`, con el entorno elegido activo:

```bash
python main.py
```

Si se usa el entorno Conda indicado, también puede ejecutarse explícitamente:

```bash
conda run -n preentrega312 python main.py
```

El programa recorre OpenAI y Anthropic, en ese orden. Para cada proveedor con clave configurada, pregunta «¿Qué es la entropía? Respondé en 2 líneas.» primero en modo normal y luego en streaming. Cada ejecución realiza dos solicitudes por proveedor configurado; las llamadas reales pueden consumir créditos y requieren acceso al modelo seleccionado.

Una clave ausente, vacía o con solo espacios omite el proveedor. Si hay clave pero falta el modelo, se informa `falta OPENAI_MODEL` o `falta ANTHROPIC_MODEL`. Los parámetros inválidos se identifican mediante el nombre de su variable de entorno, sin mostrar su valor. Un error de configuración se informa y permite continuar con el siguiente. Si ninguno llega a intentar la generación, se muestra un aviso final indicando que no se ejecutaron solicitudes y se remite a `.env.example`. Los clientes devuelven errores controlados: en modo normal mediante `ModelResponse.error`, y en streaming mediante un fragmento de texto con el mensaje de error. Los reintentos automáticos del SDK están desactivados.

## Happy path: respuesta normal y streaming

Ejemplo obtenido en una ejecución real. El contenido y la longitud de las respuestas pueden variar. En la terminal, el texto del modo streaming se imprime a medida que llegan los fragmentos; este bloque muestra el resultado completo.

```text
=== OPENAI ===
Modo normal:

La entropía mide cuántas configuraciones microscópicas pueden corresponder a un mismo estado macroscópico de un sistema.
En teoría de la información, mide la incertidumbre: cuanto menos predecible es un resultado, mayor es su entropía.

Modo streaming:

La entropía mide cuántas configuraciones microscópicas pueden producir un mismo estado físico; suele asociarse con la dispersión de la energía.
En teoría de la información, mide la incertidumbre: cuanto menos predecible es un resultado, mayor es su entropía.

=== ANTHROPIC ===
Modo normal:

La entropía es una medida del desorden o de la cantidad de microestados posibles de un sistema, y en termodinámica indica qué tan irreversible es un proceso. En términos simples, mide cuánta energía de un sistema ya no está disponible para realizar trabajo útil.

Modo streaming:

La entropía es una medida del grado de desorden o de la cantidad de configuraciones microscópicas posibles de un sistema. En termodinámica, indica cuánta energía no puede convertirse en trabajo útil y tiende a aumentar en los procesos naturales.
```

## Sad paths: errores controlados

### Claves vacías

Si ambas claves están vacías, `main.py` omite las llamadas:

```text
=== OPENAI ===
Omitido: no se configuró OPENAI_API_KEY.

=== ANTHROPIC ===
Omitido: no se configuró ANTHROPIC_API_KEY.

No se ejecutaron solicitudes: configurá las claves, modelos y parámetros indicados en .env.example.
```

Al construir `LLMConfig`, Pydantic exige el modelo no vacío y la clave no vacía correspondiente al proveedor; de lo contrario lanza `ValidationError`, antes de crear el cliente del SDK. Esta validación comprueba presencia, no autenticidad, saldo ni permisos de acceso. El manager conserva además una comprobación defensiva de la clave.

`ChatMessage` admite únicamente los roles `user`, `assistant` y `system`, y rechaza contenido vacío o compuesto solo por espacios en blanco. El contenido válido se conserva sin recortarlo. En Anthropic, las instrucciones de sistema se separan de la conversación y el parámetro `system` se omite cuando no hay instrucciones.

### Clave inválida

Ejemplo de salida para una clave inválida de OpenAI, con modelo configurado:

```text
=== OPENAI ===
Modo normal:
Error: Error de autenticación con OpenAI. Revisá la clave configurada.
Modo streaming:

[Error durante el streaming: Error de autenticación con OpenAI. Revisá la clave configurada.]
```

Para Anthropic, un error de autenticación se comunica de la misma manera, identificando a Anthropic. Los tests simulan este caso para ambos proveedores.

### Sin conexión de red

Salida obtenida al ejecutar sin conexión Wi-Fi:

```text
=== OPENAI ===
Modo normal:

Error: Error de conexión con OpenAI.

Modo streaming:

[Error durante el streaming: Error de conexión con OpenAI.]

=== ANTHROPIC ===
Modo normal:

Error: Error de conexión con Anthropic.

Modo streaming:

[Error durante el streaming: Error de conexión con Anthropic.]
```

También se controlan errores de timeout y límite de solicitudes o cuota. Anthropic incluye diagnóstico de estado HTTP para otros errores, como saldo insuficiente. Los mensajes no muestran el contenido original de las excepciones ni credenciales.

## Ejecutar los tests

Desde `Pre-entrega1/`, con el entorno activo:

```bash
python -m unittest discover -s tests -v
```

Alternativa explícita con Conda:

```bash
conda run -n preentrega312 python -m unittest discover -s tests -v
```

La suite contiene **56 tests**, comprobados en Python 3.12. No requiere claves ni hace llamadas reales a APIs: utiliza mocks para simular respuestas y errores.

Cubre validación y ocultamiento de claves, interfaz abstracta, selección de proveedor, delegación, generación normal, streaming, respuestas sin texto, autenticación, cuota, conexión y timeout. También verifica la propagación incremental de fragmentos desde el SDK simulado a través del cliente Anthropic y del manager antes de cerrar el stream, la conservación de texto anterior a un error, los payloads con y sin instrucciones de sistema y el manejo de un rechazo simulado para una lista vacía. Comprueba además la conversión de mensajes de sistema de Anthropic, los parámetros de razonamiento de OpenAI, la continuación de la demostración ante errores controlados y que otra tarea pueda avanzar mientras una llamada simulada está pendiente.

Si un stream falla después de emitir texto, conserva los fragmentos ya entregados y agrega el mensaje de error; no reanuda la generación.
