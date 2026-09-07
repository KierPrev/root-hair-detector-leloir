"""Ciclo completo de revisión en un solo comando (SOLO desarrollo).

Para cada imagen: corre la detección, te la abre para que marques lo que falta o
lo que está mal, y al cerrar imprime la evaluación. Con varias imágenes las
recorre una atrás de otra.

Uso:
    python tools/revisar.py                          # todas las de data/samples
    python tools/revisar.py data/samples/Col0_1.tif  # una en particular
    python tools/revisar.py --zona 900 1400 1100 1700

Controles dentro de la ventana:
    click izquierdo    marca inicio (base) y luego fin (punta) de un pelo que faltó
    click derecho      sobre una detección: la marca como INCORRECTA
    u                  deshace la última marca
    s                  guarda
    q                  guarda, cierra y pasa a la siguiente imagen
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "tools"))

import root_hairs  # noqa: E402
from annotate_hairs import Annotator, load_detections, load_gray  # noqa: E402


def review(image: Path, zona: tuple | None):
    stem = image.stem
    print(f"\n{'=' * 60}\n{image.name}\n{'=' * 60}")

    print("detectando pelos...")
    root_hairs.process(image)

    gray = load_gray(image)

    print("abriendo ventana de anotación (cerrala con 'q' para continuar)...")
    # hay que mantener la referencia viva: matplotlib guarda los callbacks con
    # referencias débiles, si se recolecta el objeto dejan de responder los clicks
    annotator = Annotator(gray, load_detections(stem), stem, image, zona)
    plt.show()
    # red de seguridad: si se cerró de un modo que no disparó el guardado
    # (según el backend, cerrar con la X no siempre avisa), guardar igual
    if not annotator.saved:
        annotator.save()
    del annotator

    print("\n--- evaluación ---")
    subprocess.run([sys.executable, str(BASE_DIR / "tools" / "evaluate_detection.py"), stem])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path,
                        help="imágenes a revisar (por defecto, todas las de data/samples)")
    parser.add_argument("--zona", nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"),
                        help="revisar solo un recorte, más rápido de anotar")
    args = parser.parse_args()

    images = args.images or sorted(SAMPLES_DIR.glob("*.tif"))
    if not images:
        raise SystemExit(f"no hay imágenes en {SAMPLES_DIR}")

    zona = tuple(args.zona) if args.zona else None
    for image in images:
        review(image, zona)

    print(f"\nlisto: {len(images)} imagen(es) revisada(s). "
          "Las anotaciones quedaron en data/annotations/")


if __name__ == "__main__":
    main()
