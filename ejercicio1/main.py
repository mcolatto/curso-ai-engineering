"""Ejercicio de orquestación concurrente de modelos."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from datetime import datetime


async def gpt_4_call() -> str:
    """Simular una llamada a GPT-4 y devolver una respuesta de texto."""
    hora_inicio = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{hora_inicio}] GPT-4: inicio de la llamada")
    latency = random.uniform(0.5, 2.5)
    await asyncio.sleep(latency)
    return "Respuesta simulada de GPT-4"


async def claude_3_call() -> str:
    """Simular una llamada a Claude 3 y devolver una respuesta de texto."""
    hora_inicio = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{hora_inicio}] Claude-3: inicio de la llamada")
    latency = random.uniform(0.5, 2.5)
    await asyncio.sleep(latency)
    return "Respuesta simulada de Claude 3"


async def local_llama_call() -> str:
    """Simular una llamada a Llama local y devolver una respuesta de texto."""
    hora_inicio = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{hora_inicio}] Local-llama: inicio de la llamada")
    latency = random.uniform(0.5, 2.5)
    await asyncio.sleep(latency)
    return "Respuesta simulada de Llama local"


async def main() -> None:
    """Coordinar las llamadas, limitar la concurrencia y manejar el timeout."""
    print("Primera ejecución: tres modelos simultáneamente")
    try:
        async with asyncio.timeout(2):
            resultados = await asyncio.gather(
                gpt_4_call(),
                claude_3_call(),
                local_llama_call(),
            )
        for resultado in resultados:
            print(resultado)
    except TimeoutError:
        print("Error: las tres llamadas superaron el límite total de 2 segundos.")

    semaforo = asyncio.Semaphore(2)

    async def ejecutar_con_limite(
        modelo: Callable[[], Awaitable[str]],
    ) -> str:
        """Ejecutar una llamada cuando haya un lugar disponible."""
        async with semaforo:
            return await modelo()

    print("\nSegunda ejecución: diez llamadas, como máximo dos a la vez")
    modelos = [gpt_4_call, claude_3_call, local_llama_call]
    llamadas = []
    for indice in range(10):
        modelo = modelos[indice % len(modelos)]
        llamadas.append(ejecutar_con_limite(modelo))

    try:
        async with asyncio.timeout(2):
            resultados = await asyncio.gather(*llamadas)
        for resultado in resultados:
            print(resultado)
    except TimeoutError:
        print("Error: las diez llamadas superaron el límite total de 2 segundos.")


if __name__ == "__main__":
    asyncio.run(main())
