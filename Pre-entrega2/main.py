"""Demostración asíncrona del pipeline de extracción técnica."""

import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from chain import SalidaInvalidaError, process_text


async def main() -> None:
    """Cargar la configuración, procesar dos ejemplos y mostrar sus resultados."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    examples = [
        ("Texto completo", (
            "Una API desarrollada con FastAPI utiliza Redis como caché y PostgreSQL "
            "para persistencia. El agotamiento del pool de conexiones a PostgreSQL "
            "provocó la caída del servicio: todas las solicitudes devuelven error."
        )),
        ("Texto ambiguo", (
            "El sistema a veces tarda más de lo habitual, pero no tenemos métricas "
            "ni sabemos qué componente está involucrado."
        )),
    ]
    for name, text in examples:
        print(f"\n=== {name} ===", flush=True)
        try:
            result = await process_text(text)
        except SalidaInvalidaError:
            print("Error: no se obtuvo una respuesta válida tras los dos intentos.")
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
            print("Error de la API de OpenAI. Revisá el modelo, sus parámetros y el acceso de la cuenta.")
        except ValueError:
            print(
                "Error: revisá la configuración de .env según .env.example. "
                "La entrada o respuesta también puede haber sido rechazada."
            )
        else:
            print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
