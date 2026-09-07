"""Ajusta automáticamente los parámetros de detección usando tus anotaciones.

Puntúa cada combinación de parámetros según:

    + encontrar los pelos que marcaste como faltantes
    - volver a producir las detecciones que marcaste como incorrectas
    - error en la longitud de los pelos que anotaste

y guarda la mejor en src/parametros_deteccion.json, que es el archivo que usa la
detección de ahí en adelante.

En vez de probar la rejilla completa (cientos de combinaciones, inviable sobre
imágenes enteras) hace descenso coordenado: parte de los parámetros actuales y
ajusta uno por vez, quedándose con el mejor valor de cada uno, y repite hasta
que ninguna mejora aparece. Las combinaciones de cada paso se reparten entre
todos los núcleos del procesador.

Uso:
    python tools/ajustar_parametros.py
    python tools/ajustar_parametros.py --aplicar     # además guarda el resultado
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

BASE_DIR = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
SAMPLES_DIR = BASE_DIR / "data" / "samples"
sys.path.insert(0, str(BASE_DIR / "src"))

import root_hairs  # noqa: E402

MATCH_RADIUS_PX = 40

# Valores candidatos de cada parámetro, ordenados de menor a mayor.
SEARCH_GRID = {
    "hysteresis_low_pct": [82, 85, 88, 91, 94, 96],
    "hysteresis_high_pct": [96, 97, 98, 99],
    "min_straightness": [0.70, 0.80, 0.88, 0.93],
    "max_bow_ratio": [0.06, 0.08, 0.10, 0.12, 1.0],
    "min_hair_length_px": [15, 20, 25, 30],
    "root_margin_px": [2, 4, 6, 8],
    "base_reach_px": [2, 4, 6, 8, 12],
    # filtros pensados para quedarse solo con pelos nítidos y no cruzados,
    # que es el mismo criterio con el que se anotó a mano
    "min_sharpness_pct": [0, 80, 90, 95, 97],
    "max_angle_deg": [40, 55, 70, 180],
    "max_blob_overlap": [0.15, 0.3, 0.6, 1.0],
    "filtrar_cruces": [False, True],
    "junction_min_dist_px": [40, 70, 100],
}

MAX_PASSES = 3  # pasadas de descenso coordenado antes de cortar


# --------------------------------------------------------------------------
# anotaciones
# --------------------------------------------------------------------------

def read_segments(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if rows and "base_x" not in rows[0]:
        return []
    return [
        {
            "base": np.array([float(r["base_x"]), float(r["base_y"])]),
            "tip": np.array([float(r["tip_x"]), float(r["tip_y"])]),
            "length_px": float(r["length_px"]),
        }
        for r in rows
    ]


def find_annotated_cases() -> list[dict]:
    """Imágenes que tienen anotaciones manuales asociadas."""
    cases = []
    for truth_path in sorted(ANNOTATIONS_DIR.glob("*_truth.csv")):
        stem = truth_path.name[: -len("_truth.csv")]
        image = SAMPLES_DIR / f"{stem}.tif"
        if not image.exists():
            print(f"aviso: no encuentro la imagen {image}, salteo {stem}")
            continue
        zona_completa = None
        meta_path = ANNOTATIONS_DIR / f"{stem}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                zona_completa = json.load(f).get("zona_completa")
        cases.append({
            "stem": stem,
            "image": image,
            "truth": read_segments(truth_path),
            "rejected": read_segments(ANNOTATIONS_DIR / f"{stem}_rejected.csv"),
            # región donde el usuario declaró haber marcado TODOS los pelos
            "zona_completa": zona_completa,
        })
    return cases


# --------------------------------------------------------------------------
# puntuación
# --------------------------------------------------------------------------

def matches(a: dict, b: dict) -> bool:
    return (np.linalg.norm(a["tip"] - b["tip"]) + np.linalg.norm(a["base"] - b["base"])) < MATCH_RADIUS_PX


def same_hair(a: dict, b: dict) -> bool:
    """Dos segmentos son el mismo pelo si comparten la punta, aunque la base
    esté corrida (es lo que pasa cuando se corrige una medición)."""
    return np.linalg.norm(a["tip"] - b["tip"]) < MATCH_RADIUS_PX


def score_case(hairs: list[dict], case: dict) -> dict:
    """Puntúa una corrida contra las anotaciones de una imagen.

    Una detección rechazada que además tiene una marca manual sobre el mismo pelo
    no es un falso positivo: es una corrección de longitud. Solo se cuentan como
    falsos positivos los rechazos sin marca manual asociada.
    """
    detections = [
        {"base": np.array(h["base"], dtype=float),
         "tip": np.array(h["tip"], dtype=float),
         "length_px": h["length_px"]}
        for h in hairs
    ]

    spurious = [r for r in case["rejected"] if not any(same_hair(r, t) for t in case["truth"])]

    found, length_errors = 0, []
    for t in case["truth"]:
        hit = next((d for d in detections if same_hair(t, d)), None)
        if hit is not None:
            found += 1
            if t["length_px"] > 0:
                length_errors.append(abs(hit["length_px"] - t["length_px"]) / t["length_px"])

    reproduced_bad = sum(1 for r in spurious if any(matches(r, d) for d in detections))

    # Falsos positivos medibles: en una zona declarada completa, el usuario marcó
    # todos los pelos que queremos medir (los nítidos y no cruzados). Por lo tanto
    # cualquier otra detección ahí es indeseable: o es basura, o es un pelo
    # borroso/cruzado cuya longitud no sería confiable.
    false_positives, n_in_zone = 0, 0
    zona = case.get("zona_completa")
    if zona:
        y0, y1, x0, x1 = zona
        for d in detections:
            bx, by = d["base"]
            tx, ty = d["tip"]
            dentro = (x0 <= bx <= x1 and y0 <= by <= y1) and (x0 <= tx <= x1 and y0 <= ty <= y1)
            if not dentro:
                continue
            n_in_zone += 1
            if not any(same_hair(t, d) for t in case["truth"]):
                false_positives += 1

    return {
        "n_truth": len(case["truth"]),
        "found": found,
        "reproduced_bad": reproduced_bad,
        "n_spurious": len(spurious),
        "n_detections": len(detections),
        "false_positives": false_positives,
        "n_in_zone": n_in_zone,
        "has_zone": bool(zona),
        "length_errors": length_errors,
    }


def total_score(totals: dict) -> float:
    """Puntaje a maximizar.

    Encontrar un pelo marcado suma; un falso positivo dentro de una zona
    completamente anotada resta lo mismo, para que no sea negocio bajar los
    umbrales e inundar la imagen de detecciones.

    Si no hay ninguna zona completa no se pueden medir los falsos positivos, y
    entonces se penaliza la cantidad total de detecciones como sustituto pobre:
    sin eso, el óptimo es siempre el umbral más permisivo posible.
    """
    score = (
        totals["found"] * 10
        - totals["reproduced_bad"] * 8
        - totals["false_positives"] * 10
        - totals["mean_length_error"] * 5
    )
    if not totals["has_zone"]:
        score -= totals["n_detections"] * 0.2
    return score


# --------------------------------------------------------------------------
# evaluación en paralelo
# --------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(cases: list[dict], sigmas: list):
    """Cada proceso calcula una sola vez la parte cara (el filtro de crestas) y
    la reusa para todas las combinaciones que le toquen."""
    warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")
    prepared = []
    for case in cases:
        gray = root_hairs.load_gray(case["image"])
        root_mask = root_hairs.segment_root_body(gray)
        ridges = root_hairs.compute_ridges(gray, sigmas)
        prepared.append((case, gray, root_mask, ridges))
    _WORKER["cases"] = prepared


def _eval_combo(combo_and_base: tuple) -> tuple:
    combo, base_params = combo_and_base
    params = {**base_params, **combo}
    totals = {"n_truth": 0, "found": 0, "reproduced_bad": 0, "n_spurious": 0,
              "n_detections": 0, "false_positives": 0, "n_in_zone": 0,
              "has_zone": False, "length_errors": []}
    for case, gray, root_mask, ridges in _WORKER["cases"]:
        hair_mask = root_hairs.detect_hair_mask(gray, root_mask, params, ridges=ridges)
        # ridges y gray son imprescindibles: sin ellos los filtros de nitidez y
        # de manchas brillantes no se aplican, y la búsqueda estaría evaluando
        # algo distinto de lo que después corre en producción
        hairs = root_hairs.measure_hairs(hair_mask, root_mask, params,
                                         ridges=ridges, gray=gray)
        s = score_case(hairs, case)
        for k in ("n_truth", "found", "reproduced_bad", "n_spurious", "n_detections",
                  "false_positives", "n_in_zone"):
            totals[k] += s[k]
        totals["has_zone"] |= s["has_zone"]
        totals["length_errors"].extend(s["length_errors"])

    errors = totals["length_errors"]
    totals["mean_length_error"] = float(np.mean(errors)) if errors else 0.0
    totals["score"] = total_score(totals)
    return combo, totals


def _eval_batch(pool, combos: list[dict], base_params: dict) -> list[tuple]:
    return pool.map(_eval_combo, [(c, base_params) for c in combos])


class Tuner:
    """Descenso coordenado sobre la rejilla, evaluando en paralelo."""

    def __init__(self, cases: list[dict], base_params: dict, n_procs: int | None = None,
                 progress=None):
        self.cases = cases
        self.base_params = base_params
        self.n_procs = n_procs or max(1, (os.cpu_count() or 2) - 1)
        self.progress = progress or (lambda *a: None)
        self.n_evals = 0

    def run(self) -> tuple[dict, dict]:
        ctx = mp.get_context("spawn")
        with ctx.Pool(self.n_procs, initializer=_init_worker,
                      initargs=(self.cases, self.base_params["sato_sigmas"])) as pool:
            start = {k: self.base_params[k] for k in SEARCH_GRID}
            best, best_totals = _eval_batch(pool, [start], self.base_params)[0]
            best = dict(best)
            self.n_evals += 1
            self.progress(self.n_evals, best_totals["score"])

            for _ in range(MAX_PASSES):
                mejoro = False
                for name, values in SEARCH_GRID.items():
                    candidates = [{**best, name: v} for v in values if v != best[name]]
                    candidates = [c for c in candidates
                                  if c["hysteresis_low_pct"] < c["hysteresis_high_pct"]]
                    if not candidates:
                        continue
                    for combo, totals in _eval_batch(pool, candidates, self.base_params):
                        self.n_evals += 1
                        if totals["score"] > best_totals["score"]:
                            best, best_totals, mejoro = dict(combo), totals, True
                        self.progress(self.n_evals, best_totals["score"])
                if not mejoro:
                    break  # ningún parámetro mejora solo: estamos en un óptimo local
        return best, best_totals


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aplicar", action="store_true",
                        help="guardar los mejores parámetros en src/parametros_deteccion.json")
    parser.add_argument("--procesos", type=int, default=None,
                        help="cuántos núcleos usar (por defecto, todos menos uno)")
    args = parser.parse_args()

    cases = find_annotated_cases()
    if not cases:
        raise SystemExit(f"no hay anotaciones en {ANNOTATIONS_DIR}. Usá tools/entrenar.py primero.")

    print(f"casos anotados: {len(cases)} "
          f"({sum(len(c['truth']) for c in cases)} pelos marcados, "
          f"{sum(len(c['rejected']) for c in cases)} detecciones rechazadas)")

    base_params = root_hairs.load_params()
    tuner = Tuner(cases, base_params, args.procesos,
                  progress=lambda n, s: print(f"\r  {n} combinaciones probadas, mejor score {s:6.1f}   ",
                                              end="", flush=True))
    print(f"buscando con {tuner.n_procs} procesos en paralelo...")
    best_combo, best_totals = tuner.run()
    print()

    print(f"\n{'=' * 60}\nmejor combinación (score {best_totals['score']:.1f}, "
          f"{tuner.n_evals} evaluaciones):")
    for k, v in best_combo.items():
        print(f"  {k}: {v}")
    print(f"  pelos marcados encontrados : {best_totals['found']}/{best_totals['n_truth']}")
    print(f"  detecciones malas repetidas: {best_totals['reproduced_bad']}")
    print(f"  detecciones totales        : {best_totals['n_detections']}")
    if best_totals["has_zone"]:
        print(f"  falsos positivos (en zona completa): {best_totals['false_positives']}"
              f" de {best_totals['n_in_zone']} detecciones en zona")
    print(f"  error medio de longitud    : {best_totals['mean_length_error']:.0%}")

    if args.aplicar:
        params = {**base_params, **best_combo}
        with open(root_hairs.CONFIG_PATH, "w") as f:
            json.dump(params, f, indent=4)
        print(f"\nguardado en {root_hairs.CONFIG_PATH}")
    else:
        print("\n(no se guardó nada; volvé a correr con --aplicar para usar estos parámetros)")


if __name__ == "__main__":
    main()
