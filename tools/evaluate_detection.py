"""Evalúa la detección automática contra las anotaciones manuales (SOLO desarrollo).

Uso:
    python tools/evaluate_detection.py Col0_0_crop
    python tools/evaluate_detection.py Col0_0_crop --completo

Por defecto asume anotación COMPLEMENTARIA, que es el flujo normal: marcaste los
pelos que el algoritmo se perdió y rechazaste con click derecho las detecciones
equivocadas, sin volver a marcar las que ya estaban bien. Entonces:

    pelos reales   = detecciones no rechazadas + marcas manuales que no coinciden
                     con ninguna detección
    recall         = detecciones no rechazadas / pelos reales
    precisión      = detecciones no rechazadas / detecciones totales

Con --completo se asume que marcaste TODOS los pelos de la imagen, y ahí las
marcas manuales sí son el ground truth completo.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
RESULTS_DIR = BASE_DIR / "data" / "results"

# Un pelo detectado cuenta como el mismo que el anotado si su punta y su base
# caen dentro de este radio (en píxeles).
MATCH_RADIUS_PX = 40


def read_segments(path: Path) -> list[dict]:
    with open(path) as f:
        return [
            {
                "hair_id": int(row["hair_id"]),
                "base": np.array([float(row["base_x"]), float(row["base_y"])]),
                "tip": np.array([float(row["tip_x"]), float(row["tip_y"])]),
                "length_px": float(row["length_px"]),
            }
            for row in csv.DictReader(f)
        ]


def match(truth: list[dict], detections: list[dict]):
    unmatched_det = list(detections)
    pairs, missed = [], []
    for t in truth:
        best, best_cost = None, MATCH_RADIUS_PX * 2
        for d in unmatched_det:
            cost = np.linalg.norm(t["tip"] - d["tip"]) + np.linalg.norm(t["base"] - d["base"])
            if cost < best_cost:
                best, best_cost = d, cost
        if best is None:
            missed.append(t)
        else:
            pairs.append((t, best))
            unmatched_det.remove(best)
    return pairs, missed, unmatched_det


def read_rejected(path: Path) -> list[dict]:
    """Detecciones que el anotador marcó como incorrectas.

    Se identifican por posición y no por id, porque los ids se renumeran cada vez
    que se cambia el algoritmo de detección.
    """
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if rows and "base_x" not in rows[0]:
        print("aviso: anotación de rechazos en formato viejo (solo ids); "
              "volvé a marcarlos para que se emparejen por posición.\n")
        return []
    return [
        {
            "base": np.array([float(r["base_x"]), float(r["base_y"])]),
            "tip": np.array([float(r["tip_x"]), float(r["tip_y"])]),
        }
        for r in rows
    ]


def count_rejected_still_present(rejected: list[dict], detections: list[dict]) -> int:
    """Cuántas de las detecciones rechazadas siguen apareciendo en el resultado actual."""
    remaining = list(detections)
    count = 0
    for rej in rejected:
        for det in remaining:
            if (np.linalg.norm(rej["tip"] - det["tip"]) + np.linalg.norm(rej["base"] - det["base"])) < MATCH_RADIUS_PX:
                remaining.remove(det)
                count += 1
                break
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stem", help="nombre base, ej: Col0_0_crop")
    parser.add_argument("--completo", action="store_true",
                        help="las marcas manuales cubren TODOS los pelos de la imagen")
    args = parser.parse_args()

    truth_path = ANNOTATIONS_DIR / f"{args.stem}_truth.csv"
    det_path = RESULTS_DIR / f"{args.stem}_hairs.csv"
    if not truth_path.exists():
        raise SystemExit(f"faltan anotaciones manuales: {truth_path}")
    if not det_path.exists():
        raise SystemExit(f"faltan detecciones: {det_path}")

    truth = read_segments(truth_path)
    detections = read_segments(det_path)
    rejected = read_rejected(ANNOTATIONS_DIR / f"{args.stem}_rejected.csv")
    pairs, missed, unmatched_det = match(truth, detections)

    if args.completo:
        print(f"pelos anotados a mano  : {len(truth)}  (ground truth completo)")
        print(f"detecciones automáticas: {len(detections)}")
        print(f"emparejados            : {len(pairs)}")
        if truth:
            print(f"recall    : {len(pairs) / len(truth):.0%}  (pelos reales encontrados)")
        if detections:
            print(f"precisión : {len(pairs) / len(detections):.0%}  (detecciones que son pelos reales)")
        print(f"no detectados : {len(missed)}   falsos positivos : {len(unmatched_det)}")
    else:
        still_bad = count_rejected_still_present(rejected, detections)
        good_detections = len(detections) - still_bad
        real_hairs = good_detections + len(missed)
        print("modo complementario: las marcas manuales son pelos que faltaron,")
        print("las detecciones no rechazadas se asumen correctas.\n")
        print(f"detecciones automáticas : {len(detections)}")
        print(f"  marcadas mal por vos y todavía presentes: {still_bad} (de {len(rejected)} rechazos anotados)")
        print(f"  asumidas correctas    : {good_detections}")
        print(f"pelos que marcaste y faltaban: {len(missed)}")
        print(f"pelos reales estimados       : {real_hairs}")
        if real_hairs:
            print(f"\nrecall    : {good_detections / real_hairs:.0%}  (pelos reales encontrados)")
        if detections:
            print(f"precisión : {good_detections / len(detections):.0%}  (detecciones correctas)")
        if pairs:
            print(f"\n{len(pairs)} marca(s) manual(es) coincidieron con una detección existente;")
            print("se usan abajo para medir el error de longitud.")

    if pairs:
        errors = np.array([d["length_px"] - t["length_px"] for t, d in pairs])
        rel = np.array([(d["length_px"] - t["length_px"]) / t["length_px"] for t, d in pairs if t["length_px"] > 0])
        print(f"\nerror de longitud: mediana {np.median(errors):+.1f} px | "
              f"|error| medio {np.abs(errors).mean():.1f} px | relativo medio {np.abs(rel).mean():.0%}")
        print("\n  manual  detectado   dif")
        for t, d in sorted(pairs, key=lambda p: p[0]["hair_id"]):
            print(f"  {t['length_px']:6.0f}  {d['length_px']:8.0f}  {d['length_px'] - t['length_px']:+6.0f}")


if __name__ == "__main__":
    main()
