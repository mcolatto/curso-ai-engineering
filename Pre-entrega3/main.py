"""Demostración asíncrona con preguntas respondibles y fuera del corpus."""

import asyncio
import logging

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from openai import (
    APIConnectionError, APIError, APITimeoutError,
    AuthenticationError, PermissionDeniedError, RateLimitError,
)

from chain import get_rag_response
from config import BASE_DIR


CASOS = [
    ("Con respuesta: RAG", "¿Cuáles son los beneficios de la RAG?"),
    ("Con respuesta: curación de datos", "¿Qué es la curación de datos?"),
    ("Con respuesta: vectores", "¿Qué son los vectores?"),
    ("Pregunta trampa: Pinecone", "¿Qué es Pinecone?"),
    (
        "Pregunta trampa: índices de Pinecone",
        "¿Qué algoritmo utiliza Pinecone para distribuir sus índices entre servidores?",
    ),
    ("Pregunta trampa: HNSW", "¿Qué es HNSW?"),
]


async def main() -> None:
    """Consultar los seis casos y mostrar respuestas, fuentes y fragmentos."""
    load_dotenv(BASE_DIR / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    for nombre, pregunta in CASOS:
        print(f"\n=== {nombre} ===", flush=True)
        print(pregunta, flush=True)
        try:
            result = await get_rag_response(pregunta)
        except OutputParserException:
            print("Error: la respuesta del modelo no cumple el formato esperado.")
        except AuthenticationError:
            print("Error: OpenAI rechazó la autenticación. Revisá OPENAI_API_KEY.")
        except PermissionDeniedError:
            print("Error: no hay permisos para acceder al recurso de OpenAI.")
        except RateLimitError:
            print("Error: se alcanzó el límite de solicitudes o la cuota de OpenAI.")
        except APITimeoutError:
            print("Error: se agotó el tiempo de espera de OpenAI.")
        except APIConnectionError:
            print("Error: no se pudo establecer conexión con OpenAI.")
        except APIError:
            print("Error de OpenAI. Revisá los modelos, sus parámetros y el acceso de la cuenta.")
        except ValueError:
            print(
                "Error: revisá .env según .env.example y comprobá la ingesta "
                "con ingest.py. La colección debe estar poblada y usar el mismo modelo de embeddings."
            )
        else:
            print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
