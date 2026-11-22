"""Comparar reportes locales de concurrencia sin consultar APIs."""

import argparse
import json
from decimal import Decimal
from pathlib import Path

from scripts.prueba_concurrencia import CANTIDAD, Reporte, calcular_p95


def resumir(reporte: Reporte) -> dict:
    """Aceptar sólo cinco ejecuciones completas, independientes y con métricas."""
    ids = {m.job_id for m in reporte.mediciones if m.job_id is not None}
    if reporte.estado != "COMPLETO" or len(reporte.mediciones) != CANTIDAD or len(ids) != CANTIDAD:
        raise ValueError("Cada reporte debe contener cinco trabajos únicos de un lote completo")
    if any(
        m.estado != "DONE" or m.error or m.latencia_traza_s is None
        or m.tokens is None or m.costo_usd is None
        for m in reporte.mediciones
    ):
        raise ValueError("No se pueden comparar trabajos fallidos o métricas incompletas")
    latencias = [m.latencia_traza_s for m in reporte.mediciones]
    if reporte.latencia_p95_s != calcular_p95(latencias):
        raise ValueError("El p95 registrado no coincide con las cinco duraciones de traza")
    configuraciones = [m.configuracion for m in reporte.mediciones]
    configuracion = configuraciones[0] if all(c == configuraciones[0] for c in configuraciones) else None
    return {
        "lote_id": str(reporte.lote_id), "iniciado_en": reporte.iniciado_en.isoformat(),
        "trabajos_done": CANTIDAD, "prompt": reporte.prompt, "configuracion": configuracion,
        "respuestas_registradas": sum(bool(m.respuesta and m.respuesta.strip()) for m in reporte.mediciones),
        "latencias_traza_s": latencias,
        "latencia_media_s": sum(latencias) / CANTIDAD, "latencia_p95_s": reporte.latencia_p95_s,
        "tokens_por_ejecucion": [m.tokens for m in reporte.mediciones],
        "tokens_totales": sum(m.tokens for m in reporte.mediciones),
        "costos_por_ejecucion_usd": [str(m.costo_usd) for m in reporte.mediciones],
        "costo_total_usd": str(sum((m.costo_usd for m in reporte.mediciones), Decimal(0))),
    }


def comparar(referencia: Reporte, actual: Reporte) -> dict:
    """Calcular diferencias descriptivas; la calidad de respuestas requiere revisión."""
    if referencia.lote_id == actual.lote_id or (
        {m.job_id for m in referencia.mediciones} & {m.job_id for m in actual.mediciones}
    ):
        raise ValueError("La referencia y la corrida actual deben tener lotes y trabajos distintos")
    anterior, nuevo = resumir(referencia), resumir(actual)
    variaciones = {}
    for campo in ("latencia_media_s", "latencia_p95_s", "tokens_totales", "costo_total_usd"):
        base = Decimal(anterior[campo]) if campo == "costo_total_usd" else anterior[campo]
        valor = Decimal(nuevo[campo]) if campo == "costo_total_usd" else nuevo[campo]
        diferencia = valor - base
        variaciones[campo] = {
            "diferencia": str(diferencia) if isinstance(diferencia, Decimal) else diferencia,
            "porcentaje": float(diferencia / base * 100) if base != 0 else None,
        }
    avisos = []
    if anterior["configuracion"] is None:
        avisos.append("La referencia histórica no registra una configuración común; no se supone igualdad de parámetros.")
    elif anterior["configuracion"] != nuevo["configuracion"]:
        avisos.append("Las configuraciones registradas difieren; considerar ese cambio al interpretar las métricas.")
    if anterior["prompt"] is None:
        avisos.append("La referencia histórica no conserva el prompt en el JSON.")
    elif anterior["prompt"] != nuevo["prompt"]:
        avisos.append("Los prompts registrados difieren; las solicitudes no son equivalentes.")
    if anterior["respuestas_registradas"] < CANTIDAD:
        avisos.append("La referencia histórica no conserva todas las respuestas; la comparación textual requiere sus trazas originales.")
    if nuevo["respuestas_registradas"] < CANTIDAD:
        avisos.append("La corrida actual no conserva todas las respuestas para su revisión.")
    return {"referencia_historica": anterior, "corrida_actual": nuevo, "variaciones": variaciones, "avisos": avisos}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path, help="JSON del lote nuevo.")
    parser.add_argument("--referencia", type=Path, required=True, help="JSON de otro lote completo de referencia.")
    parser.add_argument("--salida", type=Path, help="Guardar también la comparación en un archivo nuevo.")
    args = parser.parse_args(argv)
    try:
        referencia = Reporte.model_validate_json(args.referencia.read_text(encoding="utf-8"))
        actual = Reporte.model_validate_json(args.actual.read_text(encoding="utf-8"))
        resultado = comparar(referencia, actual)
        texto = json.dumps(resultado, ensure_ascii=False, indent=2)
        if args.salida:
            args.salida.parent.mkdir(parents=True, exist_ok=True)
            with args.salida.open("x", encoding="utf-8") as archivo:
                archivo.write(texto + "\n")
    except (OSError, ValueError) as error:
        print(f"No se pudo comparar o guardar los reportes ({type(error).__name__}).")
        return 1
    print(texto)
    print("Comparación descriptiva: las variaciones no determinan aprobación ni calidad de las respuestas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
