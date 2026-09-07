"""Herramienta de anotación manual (SOLO desarrollo, no va en el producto final).

Sirve para construir un "ground truth": vos marcás a mano el inicio y el fin de
los pelos, y con eso se puede medir objetivamente qué tan bien anda la detección
automática (cuántos encuentra, cuántos inventa, y cuánto error tienen los largos).

Uso:
    python tools/annotate_hairs.py data/samples/Col0_0.tif
    python tools/annotate_hairs.py data/samples/Col0_0.tif --zona 900 1400 1100 1700

Controles:
    click izquierdo    marca inicio (base) y luego fin (punta) de un pelo
    click derecho      sobre una detección automática: la marca como INCORRECTA
    c                  declara la vista actual como COMPLETAMENTE anotada
                       (imprescindible para poder medir falsos positivos)
    u                  deshace la última marca
    s                  guarda las anotaciones
    q                  guarda y cierra (cerrar con la X también guarda)
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

BASE_DIR = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
RESULTS_DIR = BASE_DIR / "data" / "results"


def load_gray(path: Path) -> np.ndarray:
    im = tifffile.imread(str(path)).astype(np.float32)
    lo, hi = np.percentile(im, (0.5, 99.5))
    return np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)


def _load_previous_truth(stem: str) -> list[tuple]:
    """Marcas manuales ya guardadas para esta imagen, para no perderlas."""
    path = ANNOTATIONS_DIR / f"{stem}_truth.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return [
            (float(r["base_x"]), float(r["base_y"]), float(r["tip_x"]), float(r["tip_y"]))
            for r in csv.DictReader(f)
        ]


def _load_previous_rejected(stem: str) -> list[dict]:
    """Detecciones tachadas en rondas anteriores (guardadas por coordenadas)."""
    path = ANNOTATIONS_DIR / f"{stem}_rejected.csv"
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if "base_x" in r]


def _load_previous_zone(stem: str):
    path = ANNOTATIONS_DIR / f"{stem}_meta.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("zona_completa")


def load_detections(stem: str) -> list[dict]:
    csv_path = RESULTS_DIR / f"{stem}_hairs.csv"
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return [
            {
                "hair_id": int(row["hair_id"]),
                "length_px": float(row["length_px"]),
                "base": (float(row["base_x"]), float(row["base_y"])),
                "tip": (float(row["tip_x"]), float(row["tip_y"])),
            }
            for row in csv.DictReader(f)
        ]


def _backup_if_shrinking(path: Path, nuevas: int):
    """Copia de seguridad si el guardado tiene menos marcas que lo ya guardado.

    Anotar cuesta tiempo y perderlo es caro: ante cualquier sospecha de pérdida
    se conserva el archivo anterior con marca de tiempo.
    """
    if not path.exists():
        return
    with open(path) as f:
        previas = sum(1 for _ in csv.DictReader(f))
    if nuevas >= previas:
        return
    respaldo = path.with_name(f"{path.stem}.{datetime.now():%Y%m%d_%H%M%S}.bak.csv")
    shutil.copy2(path, respaldo)
    print(f"[!] el guardado tiene {nuevas} marcas y había {previas}: "
          f"respaldo en {respaldo.name}")


class Annotator:
    def __init__(self, gray: np.ndarray, detections: list[dict], stem: str,
                 image_path: Path | None = None, zona: tuple | None = None):
        self.stem = stem
        self.image_path = image_path
        self.zona = zona
        # arrancar con lo ya anotado antes para esta imagen: si no, cada ronda
        # nueva pisaría el trabajo de las anteriores
        self.manual: list[tuple] = _load_previous_truth(stem)
        self.pending: tuple | None = None  # base a la espera de su punta
        self.rejected: set[int] = set()
        self.detections = detections
        self.saved = False
        # zona declarada como "acá marqué TODOS los pelos": es lo único que
        # permite medir falsos positivos, porque fuera de ella no se sabe si una
        # detección sin marca es un pelo real que no anotaste o basura
        self.zona_completa = _load_previous_zone(stem)
        # los tachados de rondas anteriores se conservan por coordenadas: los
        # hair_id se renumeran en cada corrida, las posiciones no
        self.rejected_previos = _load_previous_rejected(stem)

        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.ax.imshow(gray, cmap="gray")
        if zona:
            # solo se acota la vista: la imagen y las coordenadas siguen siendo
            # las completas, así ningún pelo queda cortado por el borde
            y0, y1, x0, x1 = zona
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(y1, y0)
        self.ax.set_title(
            "click izq: base->punta  |  click der: marcar detección mala  |  "
            "c: zona completa  |  u: deshacer  s: guardar  q: salir"
        )
        for det in detections:
            bx, by = det["base"]
            tx, ty = det["tip"]
            self.ax.plot([bx, tx], [by, ty], "-", color="red", linewidth=1.2, alpha=0.8)
            self.ax.text(tx + 4, ty - 4, str(det["hair_id"]), color="orange", fontsize=7)

        # dibujar las marcas heredadas de rondas anteriores
        for (bx, by, tx, ty) in self.manual:
            self.ax.plot([bx, tx], [by, ty], "-", color="cyan", linewidth=1.5)
            self.ax.plot(tx, ty, "o", color="magenta", markersize=5)
        if self.manual:
            print(f"se cargaron {len(self.manual)} marcas de rondas anteriores")
        if self.zona_completa:
            print(f"zona completa previa: {self.zona_completa}")

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        # cerrar con la X del mouse también guarda: si no, se pierde el trabajo
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        # con el zoom o el pan activados la barra de matplotlib se queda con los
        # clicks: hay que desactivarlos para poder marcar
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            print(f"[!] el modo '{toolbar.mode}' de la barra está activo y bloquea las marcas; "
                  "desactivalo (click en la lupa o la crucecita) para poder anotar")
            return
        x, y = event.xdata, event.ydata

        if event.button == 3:  # derecho: rechazar la detección más cercana
            nearest = self._nearest_detection(x, y)
            if nearest is not None:
                self.rejected.symmetric_difference_update({nearest["hair_id"]})
                self.saved = False
                self._redraw_rejections()
            return

        if self.pending is None:
            self.pending = (x, y)
            self.ax.plot(x, y, "o", color="lime", markersize=5)
        else:
            bx, by = self.pending
            self.manual.append((bx, by, x, y))
            self.pending = None
            self.ax.plot([bx, x], [by, y], "-", color="cyan", linewidth=1.5)
            self.ax.plot(x, y, "o", color="magenta", markersize=5)
            self.saved = False
            print(f"pelo manual #{len(self.manual)}: largo {np.hypot(x - bx, y - by):.1f} px")
        self.fig.canvas.draw_idle()

    def _nearest_detection(self, x, y):
        best, best_dist = None, 30.0
        for det in self.detections:
            for px, py in (det["base"], det["tip"]):
                d = np.hypot(px - x, py - y)
                if d < best_dist:
                    best, best_dist = det, d
        return best

    def _redraw_rejections(self):
        for artist in list(self.ax.texts):
            if artist.get_color() == "red" and artist.get_text() == "X":
                artist.remove()
        for det in self.detections:
            if det["hair_id"] in self.rejected:
                tx, ty = det["tip"]
                self.ax.text(tx, ty, "X", color="red", fontsize=12, fontweight="bold")
        print(f"detecciones marcadas como incorrectas: {sorted(self.rejected)}")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "u":
            if self.pending is not None:
                self.pending = None
            elif self.manual:
                self.manual.pop()
            self.saved = False
            print(f"deshecho. pelos manuales: {len(self.manual)}")
        elif event.key == "c":
            self._marcar_zona_completa()
        elif event.key == "s":
            self.save()
        elif event.key == "q":
            self.save()
            plt.close(self.fig)

    def on_close(self, event):
        if not self.saved:
            self.save()

    def _marcar_zona_completa(self):
        """Declara la vista actual como completamente anotada."""
        x0, x1 = self.ax.get_xlim()
        y1, y0 = self.ax.get_ylim()
        self.zona_completa = [round(y0, 1), round(y1, 1), round(x0, 1), round(x1, 1)]
        self.saved = False
        self.ax.set_title("ZONA COMPLETA declarada: acá tienen que estar TODOS los pelos marcados",
                          color="darkgreen")
        print(f"zona completa: y {self.zona_completa[0]:.0f}-{self.zona_completa[1]:.0f}, "
              f"x {self.zona_completa[2]:.0f}-{self.zona_completa[3]:.0f}")
        print("   -> ahí toda detección sin marca tuya cuenta como falso positivo")
        self.fig.canvas.draw_idle()

    def save(self):
        self.saved = True
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        _backup_if_shrinking(ANNOTATIONS_DIR / f"{self.stem}_truth.csv", len(self.manual))
        truth_path = ANNOTATIONS_DIR / f"{self.stem}_truth.csv"
        with open(truth_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["hair_id", "base_x", "base_y", "tip_x", "tip_y", "length_px"])
            for i, (bx, by, tx, ty) in enumerate(self.manual, start=1):
                w.writerow([i, round(bx, 1), round(by, 1), round(tx, 1), round(ty, 1),
                            round(float(np.hypot(tx - bx, ty - by)), 2)])

        # Se guardan las coordenadas además del id: los ids cambian cada vez que
        # se ajusta el algoritmo, las posiciones no.
        rejected_path = ANNOTATIONS_DIR / f"{self.stem}_rejected.csv"
        by_id = {d["hair_id"]: d for d in self.detections}
        filas = [
            [r["hair_id"], r["base_x"], r["base_y"], r["tip_x"], r["tip_y"], r["length_px"]]
            for r in self.rejected_previos
        ]
        for hid in sorted(self.rejected):
            det = by_id[hid]
            fila = [hid, det["base"][0], det["base"][1], det["tip"][0], det["tip"][1],
                    round(det["length_px"], 2)]
            # no duplicar un tachado que ya venía de una ronda anterior
            ya_esta = any(
                abs(float(f[1]) - fila[1]) < 5 and abs(float(f[2]) - fila[2]) < 5
                and abs(float(f[3]) - fila[3]) < 5 and abs(float(f[4]) - fila[4]) < 5
                for f in filas
            )
            if not ya_esta:
                filas.append(fila)

        with open(rejected_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["hair_id", "base_x", "base_y", "tip_x", "tip_y", "length_px"])
            w.writerows(filas)

        # queda registrado sobre qué imagen y qué recorte se anotó, para que el
        # ajuste de parámetros pueda reproducir exactamente la misma vista
        with open(ANNOTATIONS_DIR / f"{self.stem}_meta.json", "w") as f:
            json.dump({
                "image": str(self.image_path) if self.image_path else None,
                # zona es solo la vista usada para anotar; las coordenadas de las
                # anotaciones son siempre las de la imagen completa
                "zona": list(self.zona) if self.zona else None,
                "zona_completa": self.zona_completa,
                "crop": None,
            }, f, indent=4)

        print(f"guardado: {len(self.manual)} pelos manuales -> {truth_path}")
        print(f"guardado: {len(filas)} detecciones incorrectas -> {rejected_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--zona", nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"),
                        help="acotar la vista a esta region (no recorta la imagen)")
    args = parser.parse_args()

    gray = load_gray(args.image)
    stem = args.image.stem

    annotator = Annotator(gray, load_detections(stem), stem, args.image,
                          tuple(args.zona) if args.zona else None)
    print(__doc__)
    plt.show()


if __name__ == "__main__":
    main()
