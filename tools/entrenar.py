"""Ciclo de entrenamiento en UN SOLO COMANDO (SOLO desarrollo).

    python tools/entrenar.py

Cada ronda hace, sin que tengas que escribir nada más:

    1. detecta los pelos con los parámetros actuales
    2. te abre cada imagen para que corrijas (marcar los que faltan, tachar los
       que están mal)
    3. busca los mejores parámetros contra TODAS tus correcciones acumuladas
       (las de esta ronda y las anteriores) y los aplica
    4. te muestra cómo quedó y pregunta si querés seguir

Se repite hasta que digas que no. Las correcciones se van acumulando, así que
cada ronda parte de todo lo corregido antes.

Opciones:
    --zona Y0 Y1 X0 X1   acotar la vista para anotar cómodo (la detección
                         siempre usa la imagen completa, no se corta ningún pelo)
    --imagenes ...       usar solo algunas imágenes (por defecto, data/samples)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "tools"))

import root_hairs  # noqa: E402
import ajustar_parametros as tuner  # noqa: E402
from annotate_hairs import Annotator, load_detections, load_gray  # noqa: E402


def detectar(images: list[Path], params: dict):
    print("\n[1/3] detectando pelos con los parámetros actuales...")
    for image in images:
        root_hairs.process(image, params=params)


def corregir(images: list[Path], zona: tuple | None):
    print("\n[2/3] abriendo las imágenes para que corrijas")
    print("      click izq: base->punta de un pelo que falta")
    print("      click der: tachar una detección incorrecta")
    print("      q: guardar y pasar a la siguiente\n")
    for image in images:
        stem = image.stem
        gray = load_gray(image)
        # la referencia tiene que seguir viva o matplotlib pierde los callbacks
        annotator = Annotator(gray, load_detections(stem), stem, image, zona)
        plt.show()
        # red de seguridad: si se cerró de un modo que no disparó el guardado
        # (según el backend, cerrar con la X no siempre avisa), guardar igual
        if not annotator.saved:
            annotator.save()
        del annotator


def ajustar() -> dict | None:
    print("\n[3/3] buscando los mejores parámetros con todas tus correcciones...")
    cases = tuner.find_annotated_cases()
    usable = cases
    if not usable:
        print("      todavía no hay correcciones guardadas, no hay nada que ajustar")
        return None

    n_truth = sum(len(c["truth"]) for c in usable)
    n_rejected = sum(len(c["rejected"]) for c in usable)
    print(f"      usando {len(usable)} imagen(es): {n_truth} pelos marcados, "
          f"{n_rejected} detecciones tachadas")

    base_params = root_hairs.load_params()
    t0 = time.perf_counter()

    def progreso(n, score):
        print(f"\r      {n} combinaciones probadas, mejor score {score:6.1f} "
              f"({time.perf_counter() - t0:.0f}s)   ", end="", flush=True)

    buscador = tuner.Tuner(usable, base_params, progress=progreso)
    print(f"      buscando con {buscador.n_procs} procesos en paralelo...")
    best_combo, best_totals = buscador.run()
    print()
    print(f"\n      mejores parámetros (score {best_totals['score']:.1f}):")
    for k, v in best_combo.items():
        print(f"        {k}: {v}")
    print(f"      pelos que marcaste y ahora encuentra: {best_totals['found']}/{best_totals['n_truth']}")
    print(f"      detecciones tachadas que repite     : {best_totals['reproduced_bad']}")
    print(f"      error medio de longitud             : {best_totals['mean_length_error']:.0%}")

    params = {**base_params, **best_combo}
    with open(root_hairs.CONFIG_PATH, "w") as f:
        json.dump(params, f, indent=4)
    print(f"      aplicados y guardados en {root_hairs.CONFIG_PATH.name}")
    return params


def preguntar_seguir() -> bool:
    while True:
        r = input("\n¿Otra ronda de correcciones? [s/n]: ").strip().lower()
        if r in ("s", "si", "sí", "y", ""):
            return True
        if r in ("n", "no"):
            return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zona", nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"),
                        help="acotar la vista para anotar comodo (la deteccion siempre usa la imagen completa)")
    parser.add_argument("--imagenes", nargs="+", type=Path)
    args = parser.parse_args()

    images = args.imagenes or sorted(SAMPLES_DIR.glob("*.tif"))
    if not images:
        raise SystemExit(f"no hay imágenes en {SAMPLES_DIR}")
    zona = tuple(args.zona) if args.zona else None

    ronda = 1
    while True:
        print(f"\n{'=' * 64}\nRONDA {ronda}  ({len(images)} imagen(es))\n{'=' * 64}")
        params = root_hairs.load_params()
        detectar(images, params)
        corregir(images, zona)
        nuevos = ajustar()
        if nuevos:
            print("\nvolviendo a detectar con los parámetros nuevos para que veas cómo quedó...")
            detectar(images, nuevos)
        if not preguntar_seguir():
            break
        ronda += 1

    print(f"\nlisto tras {ronda} ronda(s). Parámetros finales en {root_hairs.CONFIG_PATH}")
    print("Los resultados están en data/results/ (CSV con las longitudes + overlays).")


if __name__ == "__main__":
    main()
