# Ejercicio 

# Opcional 

Título del ejercicio: "Refactorización a LCEL Asíncrono" 

## Propósito del ejercicio 

Este ejercicio tiene como objetivo que migres de una implementación imperativa basada en SDKs crudos (OpenAI/Anthropic) a una arquitectura declarativa usando **LCEL (LangChain Expression Language)** . Practicarás la composición de componentes y la gestión de la asincronía en un flujo profesional. 

## Objetivo de aprendizaje 

Al finalizar vas a tener un archivo Python funcional que define y ejecuta **una cadena LCEL asíncrona concreta** : toma una pregunta del usuario como diccionario (ej. `{"pregunta": "..."}` ), la procesa con `ChatPromptTemplate → modelo → StrOutputParser` y devuelve la respuesta como **texto plano** . 

En concreto, vas a lograr: 

1. Reemplazar la llamada manual al SDK del Módulo 1 por un modelo de LangChain 

   - ( `ChatOpenAI` o `ChatAnthropic` ). 

2. Componer `prompt | modelo | parser` con el operador `|` . 

3. Ejecutar la cadena con `await chain.ainvoke({...})` . 

**Cómo sabés que lo lograste:** tu script corre sin errores, imprime una respuesta en **texto plano** (no un objeto `AIMessage` ) y todo el flujo principal está armado con LCEL (el operador `|` ), no con lógica imperativa. 

Instrucciones paso a paso 

1. **Preparación del entorno** : 

   - Asegúrate de tener instalados `langchain` , `langchain-openai` (o `langchain-anthropic` ) y `python-dotenv` . 

2. **Configuración del Modelo** : 

   - Sustituye la inicialización del cliente manual por una instancia de 

      - `ChatOpenAI` o `ChatAnthropic` . 

- Configura parámetros base como `temperature` y `model_name` . 

## 3. **Definición del Template** : 

   - Usa `ChatPromptTemplate.from_messages` para crear un prompt que reciba al menos una variable (ej. `contexto` o `pregunta` ). 

4. **Construcción de la Cadena (Pipeline)** : 

   - Usa el operador `|` para conectar: `Prompt` | `Model` | `StrOutputParser()` . 

5. **Ejecución Asíncrona** : 

   - Crea una función asíncrona principal ( `main` ) que ejecute la cadena usando el método `.ainvoke()` . 

   - Pasa los inputs necesarios como un diccionario. 

## Criterios de Aceptación 

1. El código debe utilizar **exclusivamente** la sintaxis de LCEL ( `|` ) para el flujo principal. 

2. La ejecución de la cadena debe ser **asíncrona** ( `await chain.ainvoke(...)` ). 

3. La salida debe ser texto plano extraído mediante `StrOutputParser` (no el objeto `BaseMessage` completo). 

4. El prompt debe utilizar un `ChatPromptTemplate` con roles definidos (System/Human). 

## Errores comunes a evitar 

- **Olvidar el** **`await`** : Al usar `ainvoke` , si olvidas el await, obtendrás una corrutina en lugar del resultado. 

- **Manejo de variables** : Asegúrate de que las llaves `{variable}` en tu prompt coincidan exactamente con las llaves en el diccionario que pasas al `ainvoke` . 

