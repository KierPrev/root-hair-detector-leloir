"""Detección de pelos radiculares y medición de su longitud (en píxeles).

Uso:
    python src/root_hairs.py data/samples/Col0_0.tif
    python src/root_hairs.py data/samples/Col0_0.tif --crop 900 1400 1100 1700   # modo desarrollo

Genera en data/results/:
    <nombre>_hairs.csv      tabla de longitudes por pelo
    <nombre>_overlay.png    imagen de revisión con los pelos detectados numerados
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
from pathlib import Path

import warnings

import cv2
import numpy as np
import tifffile
from scipy.ndimage import find_objects, binary_fill_holes, convolve, distance_transform_edt
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.measure import label
from skimage.morphology import (
    closing,
    dilation,
    disk,
    remove_small_objects,
    skeletonize,
)

# Las deprecaciones de skimage son ruido: la migración ya está hecha donde
# corresponde y estos avisos vienen de dependencias internas.
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

# Congelado con PyInstaller: __file__ vive dentro de la carpeta temporal de
# auto-extracción (se borra al cerrar el programa), así que los resultados
# tienen que ir relativos al directorio de trabajo, no a __file__.
if getattr(sys, "frozen", False):
    BASE_DIR = Path.cwd()
    CONFIG_PATH = Path(sys._MEIPASS) / "parametros_deteccion.json"
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = Path(__file__).resolve().with_name("parametros_deteccion.json")

RESULTS_DIR = BASE_DIR / "data" / "results"

# Calibrado con data/rule_reference.tif: 1 píxel equivale a 2 micrómetros.
UM_PER_PX = 2.0

DEFAULT_PARAMS = {
    # Un pelo real mide bastante más que unos pocos píxeles; por debajo de esto
    # son artefactos del filtro o textura de la superficie de la raíz.
    "min_hair_length_px": 25,
    # Los pelos son rectos: la distancia base-punta debe ser casi igual al recorrido.
    "min_straightness": 0.80,
    # Arqueo máximo permitido: separación máxima entre el recorrido y la recta
    # base-punta, como fracción de la distancia base-punta. Un recorrido que se
    # arquea saltó de un pelo a otro, aunque su cociente de rectitud sea alto.
    "max_bow_ratio": 0.12,
    # Percentiles del umbral por histéresis sobre la respuesta del filtro Sato.
    "hysteresis_low_pct": 94,
    "hysteresis_high_pct": 98,
    # Escalas del filtro de crestas, en píxeles (grosor esperado de un pelo).
    "sato_sigmas": [1, 1.5, 2],
    # Fracción del recorrido que dos candidatos pueden compartir antes de
    # considerar que son el mismo pelo (uno es una espuela del esqueleto).
    "max_path_overlap": 0.5,
    # Margen alrededor del cuerpo de la raíz: define dónde se considera que
    # empieza el pelo. Si queda corto, el trazado se mete dentro de la raíz y
    # las longitudes salen infladas.
    "root_margin_px": 2,
    # Cuánto más lejos que el margen se acepta que empiece un pelo. Sube el
    # alcance de la búsqueda de la base sin ensanchar el recorte de la máscara.
    "base_reach_px": 6,
    # Al aceptar un pelo cuyo esqueleto arranca despegado de la raíz, se lo
    # prolonga en su propia dirección hasta la superficie: así la base queda
    # donde corresponde y la longitud incluye el tramo que faltaba.
    "max_base_extension_px": 25,
    # Un pelo enfocado da una respuesta fuerte del filtro de crestas a lo largo
    # de todo su recorrido. Los desenfocados quedan por debajo de este umbral
    # (percentil de la respuesta global) y se descartan.
    "min_sharpness_pct": 97.0,
    # Los pelos salen aproximadamente perpendiculares a la superficie de la raíz.
    # Ángulo máximo, en grados, respecto de la normal a la superficie.
    "max_angle_deg": 55,
    # Brillo por encima del cual se considera mancha/burbuja fuera de foco: sus
    # bordes imitan pelos finos y generan detecciones falsas.
    "bright_blob_pct": 99.0,
    # Descartar pelos cuyo recorrido pasa por un cruce con otra estructura.
    # Desactivado por defecto: medido contra las anotaciones manuales, saca más
    # pelos buenos que malos. El filtro de rectitud ya descarta los recorridos
    # que saltan de un pelo a otro en un cruce.
    "filtrar_cruces": False,
    # Las espuelas más cortas que esto son ruido de la esqueletización, no
    # cruces reales: se podan antes de buscar uniones.
    "spur_prune_px": 12,
    # Solo cuentan como cruce las uniones a más de esta distancia de la raíz:
    # cerca de la base todos los pelos nacen juntos y la máscara es una maraña,
    # así que ahí una unión no significa que dos pelos se crucen.
    "junction_min_dist_px": 40,
    # Fracción máxima del pelo que puede caer sobre una mancha brillante.
    "max_blob_overlap": 0.3,
}

MAX_COMPONENT_SIZE = 20000


def p(params: dict, key: str):
    """Valor de un parámetro, cayendo al valor por defecto si falta.

    Evita que una configuración vieja (o un proceso que quedó con la versión
    anterior en memoria) rompa la corrida por una clave nueva que todavía no
    conoce.
    """
    return params.get(key, DEFAULT_PARAMS[key])


def load_params() -> dict:
    """Parámetros de detección: los del archivo de configuración si existe,
    completados con los valores por defecto."""
    params = dict(DEFAULT_PARAMS)
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            params.update(json.load(f))
    return params


def load_gray(path: Path) -> np.ndarray:
    """Carga el TIFF (12/16 bit) y lo normaliza a float 0-1 usando percentiles."""
    im = tifffile.imread(str(path)).astype(np.float32)
    lo, hi = np.percentile(im, (0.5, 99.5))
    # float32 en toda la cadena: el filtro de crestas es bastante más rápido que
    # en float64 y la precisión sobra para datos de 12 bits
    return np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def segment_root_body(gray: np.ndarray) -> np.ndarray:
    """Máscara del cuerpo grueso de la raíz: estructura oscura y ancha.

    Se separa de los pelos por el grosor: una apertura morfológica con un disco
    grande borra las líneas finas y deja solo el cuerpo.
    """
    dark8 = ((1 - gray) * 255).astype(np.uint8)
    _, mask = cv2.threshold(dark8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    body = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    body = binary_fill_holes(body > 0)
    return body


_ROOT_DILATION_CACHE: dict = {}


def dilated_root(root_mask: np.ndarray, margin: int) -> np.ndarray:
    """Raíz dilatada, cacheada por (identidad de la máscara, margen).

    Solo depende del margen, así que al probar muchas combinaciones de
    parámetros no hace falta recalcularla cada vez.
    """
    key = (id(root_mask), margin)
    cached = _ROOT_DILATION_CACHE.get(key)
    if cached is None:
        cached = dilation(root_mask, disk(margin))
        _ROOT_DILATION_CACHE[key] = cached
    return cached


def compute_ridges(gray: np.ndarray, sigmas: list) -> np.ndarray:
    """Respuesta del filtro de crestas, normalizada a 0-1.

    Es con diferencia la parte más cara del proceso (~85% del tiempo) y solo
    depende de la imagen y de las escalas, así que conviene cachearla cuando se
    prueban varios umbrales sobre la misma imagen.
    """
    ridges = sato(gray, sigmas=sigmas, black_ridges=True)
    return (ridges - ridges.min()) / (ridges.max() - ridges.min() + 1e-9)


def detect_hair_mask(gray: np.ndarray, root_mask: np.ndarray, params: dict,
                     ridges: np.ndarray | None = None) -> np.ndarray:
    """Realza líneas finas oscuras (filtro Sato) y las umbraliza por histéresis."""
    if ridges is None:
        ridges = compute_ridges(gray, p(params, "sato_sigmas"))

    low = np.percentile(ridges, p(params, "hysteresis_low_pct"))
    high = np.percentile(ridges, p(params, "hysteresis_high_pct"))
    hair_mask = apply_hysteresis_threshold(ridges, low, high)
    hair_mask = closing(hair_mask, disk(2))
    hair_mask = binary_fill_holes(hair_mask)  # evita esqueletos en anillo sobre el pelo
    hair_mask &= ~dilated_root(root_mask, p(params, "root_margin_px"))
    hair_mask = remove_small_objects(hair_mask, max_size=p(params, "min_hair_length_px"))
    return hair_mask


def bright_blobs(gray: np.ndarray, pct: float) -> np.ndarray:
    """Manchas brillantes (material fuera de foco, burbujas).

    Sus bordes producen líneas finas que el filtro de crestas confunde con
    pelos, así que se excluyen junto con un pequeño halo alrededor.
    """
    thr = np.percentile(gray, pct)
    blobs = gray > thr
    return dilation(blobs, disk(3))


_NEIGHBOR_OFFSETS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def _geodesic_from(pts: set, sources: list) -> tuple:
    """Distancia geodésica sobre el esqueleto desde un conjunto de orígenes.
    Devuelve (punto más lejano, distancia, mapa de distancias).

    Dijkstra con cola de prioridad: cada píxel se cierra una sola vez. Da lo
    mismo que relajar repetidamente, pero sin revisitar.
    """
    dist = {p: 0.0 for p in sources}
    heap = [(0.0, p) for p in sources]
    heapq.heapify(heap)
    farthest, farthest_dist = sources[0], 0.0
    closed = set()
    while heap:
        d, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        if d > farthest_dist:
            farthest, farthest_dist = cur, d
        cx, cy = cur
        for dx, dy in _NEIGHBOR_OFFSETS:
            nb = (cx + dx, cy + dy)
            if nb not in pts or nb in closed:
                continue
            nd = d + (1.41421356 if dx and dy else 1.0)
            if nd < dist.get(nb, np.inf):
                dist[nb] = nd
                heapq.heappush(heap, (nd, nb))
    return farthest, farthest_dist, dist


def measure_hairs(hair_mask: np.ndarray, root_mask: np.ndarray, params: dict,
                  ridges: np.ndarray | None = None,
                  gray: np.ndarray | None = None,
                  motivos: dict | None = None) -> list[dict]:
    """Mide cada pelo: longitud geodésica desde su base (donde toca la raíz)
    hasta la punta, siguiendo el recorrido del pelo.

    Cada extremo libre del esqueleto es un pelo candidato, de modo que dos pelos
    que se tocan cerca de la base se miden por separado. Se filtran los que no
    cumplen las restricciones biológicas: los pelos son rectos, no se ramifican
    y salen aproximadamente perpendiculares a la superficie de la raíz.
    """
    skel = skeletonize(hair_mask)
    root_near = dilated_root(root_mask, p(params, "root_margin_px") + p(params, "base_reach_px"))

    # Uniones del esqueleto: donde dos pelos se cruzan o se tocan. Un recorrido
    # que pasa por una unión puede haber saltado de un pelo a otro, así que la
    # longitud no es confiable y se descarta.
    # Normal a la superficie de la raíz: el gradiente de la distancia al cuerpo
    # apunta hacia afuera, que es la dirección en la que crece un pelo sano.
    dist_to_root = distance_transform_edt(~root_mask)
    normal_gy, normal_gx = np.gradient(dist_to_root)

    # Podar espuelas antes de buscar uniones: sin esto casi todo pelo parece
    # tener un cruce, porque la esqueletización deja pequeñas ramas laterales.
    pruned = _prune_spurs(skel, p(params, "spur_prune_px"))
    neighbor_count = convolve(pruned.astype(np.uint8), _NEIGHBOR_KERNEL, mode="constant")
    junctions = pruned & (neighbor_count >= 3)
    junctions &= dist_to_root > p(params, "junction_min_dist_px")
    junctions = dilation(junctions, disk(2))

    sharpness_threshold = (np.percentile(ridges, p(params, "min_sharpness_pct"))
                           if ridges is not None else None)
    blobs = bright_blobs(gray, p(params, "bright_blob_pct")) if gray is not None else None
    max_blob_overlap = p(params, "max_blob_overlap")
    cos_limit = np.cos(np.deg2rad(p(params, "max_angle_deg")))
    filtrar_cruces = p(params, "filtrar_cruces")
    max_bow_ratio = p(params, "max_bow_ratio")
    max_base_extension = p(params, "max_base_extension_px")

    labeled = label(skel, connectivity=2)
    hairs = []
    for comp_id, sl in enumerate(find_objects(labeled), start=1):
        if sl is None:
            continue
        sub = labeled[sl] == comp_id
        ys, xs = np.nonzero(sub)
        if len(xs) < 5 or len(xs) > MAX_COMPONENT_SIZE:
            _contar(motivos, "componente_chica")
            continue

        y0, x0 = sl[0].start, sl[1].start
        pts = set(zip(xs.tolist(), ys.tolist()))
        root_sub = root_near[sl]
        bases = [(int(x), int(y)) for x, y in zip(xs, ys) if root_sub[y, x]]
        if not bases:
            _contar(motivos, "no_toca_raiz")
            continue  # descartar líneas sueltas que no nacen de la raíz

        _, _, dist = _geodesic_from(pts, bases)
        base_set = set(bases)
        for tip in _free_endpoints(sub):
            if tip in base_set or tip not in dist:
                continue
            length = dist[tip]
            if length < p(params, "min_hair_length_px"):
                _contar(motivos, "muy_corto")
                continue
            path = _trace_back(tip, dist, pts)
            base = path[-1]
            straightness = np.hypot(tip[0] - base[0], tip[1] - base[1]) / max(length, 1e-6)
            if straightness < p(params, "min_straightness"):
                _contar(motivos, "poco_recto")
                continue

            abs_path = [(x + x0, y + y0) for x, y in path]

            # recorrido arqueado: saltó de un pelo a otro en un cruce
            arqueo = bow_ratio(abs_path)
            if arqueo > max_bow_ratio:
                _contar(motivos, "arqueado")
                continue

            # pelos que se cruzan con otros: la medición no es confiable
            if filtrar_cruces and any(junctions[y, x] for x, y in abs_path[:-2]):
                _contar(motivos, "cruce")
                continue

            # pelos apoyados sobre una mancha brillante: no son medibles
            if blobs is not None:
                sobre_mancha = sum(1 for x, y in abs_path if blobs[y, x]) / len(abs_path)
                if sobre_mancha > max_blob_overlap:
                    _contar(motivos, "sobre_mancha")
                    continue

            # pelos desenfocados: respuesta débil del filtro a lo largo del pelo
            if sharpness_threshold is not None:
                sharpness = float(np.median([ridges[y, x] for x, y in abs_path]))
                if sharpness < sharpness_threshold:
                    _contar(motivos, "desenfocado")
                    continue
            else:
                sharpness = float("nan")

            # dirección del pelo contra la normal a la superficie de la raíz
            bx, by = abs_path[-1]
            nx, ny = normal_gx[by, bx], normal_gy[by, bx]
            norm = np.hypot(nx, ny)
            hx, hy = (tip[0] + x0) - bx, (tip[1] + y0) - by
            hnorm = np.hypot(hx, hy)
            if norm > 1e-6 and hnorm > 1e-6:
                cos_ang = (nx * hx + ny * hy) / (norm * hnorm)
                if cos_ang < cos_limit:
                    _contar(motivos, "mal_angulo")
                    continue
            else:
                cos_ang = float("nan")
            # anclar la base en la superficie de la raíz y sumar ese tramo
            base_real, extension = extender_hasta_raiz(abs_path, root_mask, max_base_extension)
            length += extension

            hairs.append({
                "length_px": length,
                "base_extension_px": extension,
                "straightness": straightness,
                "bow_ratio": arqueo,
                "sharpness": sharpness,
                "cos_normal": cos_ang,
                # un pelo que llega al borde de la foto está cortado: su largo
                # real es mayor que el medido, así que no sirve como dato
                "truncated": _touches_border(path, x0, y0, hair_mask.shape),
                "base": base_real,
                "tip": (tip[0] + x0, tip[1] + y0),
                "path": abs_path,
            })
    return _deduplicate(hairs, p(params, "max_path_overlap"))


BORDER_MARGIN_PX = 3


def _touches_border(path: list, x0: int, y0: int, shape: tuple) -> bool:
    """¿Algún punto del pelo toca el borde de la imagen?"""
    h, w = shape
    return any(
        (x + x0) < BORDER_MARGIN_PX or (x + x0) >= w - BORDER_MARGIN_PX
        or (y + y0) < BORDER_MARGIN_PX or (y + y0) >= h - BORDER_MARGIN_PX
        for x, y in path
    )


def _deduplicate(hairs: list[dict], max_overlap: float) -> list[dict]:
    """Un pelo con espuelas en el esqueleto genera varios candidatos que comparten
    casi todo el recorrido. Se queda con el más largo de cada grupo."""
    kept: list[dict] = []
    kept_pixels: list[set] = []
    for hair in sorted(hairs, key=lambda h: h["length_px"], reverse=True):
        path_set = set(hair["path"])
        if any(len(path_set & seen) > max_overlap * len(path_set) for seen in kept_pixels):
            continue
        kept.append(hair)
        kept_pixels.append(path_set)
    return kept


def _contar(motivos: dict | None, motivo: str):
    """Contador de diagnóstico: por qué se descartó cada candidato."""
    if motivos is not None:
        motivos[motivo] = motivos.get(motivo, 0) + 1


def extender_hasta_raiz(path: list, root_mask: np.ndarray, max_ext: int) -> tuple:
    """Prolonga el pelo desde su extremo proximal hasta la superficie de la raíz.

    El esqueleto suele arrancar unos píxeles despegado de la raíz, porque el pelo
    pierde contraste justo donde nace. Como los pelos son rectos, se puede
    avanzar en la dirección que traía el propio pelo hasta tocar la máscara de
    raíz. Devuelve (punto de la base, píxeles agregados); si no encuentra la raíz
    dentro del máximo, deja la base como estaba.
    """
    if len(path) < 2:
        return path[-1], 0.0
    base = np.asarray(path[-1], dtype=float)
    # dirección local del tramo proximal, promediando algunos píxeles para que
    # el ruido del esqueleto no desvíe la prolongación
    referencia = np.asarray(path[max(0, len(path) - 16)], dtype=float)
    direccion = base - referencia
    norma = np.hypot(direccion[0], direccion[1])
    if norma < 1e-6:
        return path[-1], 0.0
    direccion /= norma

    alto, ancho = root_mask.shape
    for paso in range(1, int(max_ext) + 1):
        punto = base + direccion * paso
        x, y = int(round(punto[0])), int(round(punto[1]))
        if not (0 <= x < ancho and 0 <= y < alto):
            break
        if root_mask[y, x]:
            return (x, y), float(paso)
    return path[-1], 0.0


def bow_ratio(path: list) -> float:
    """Cuánto se arquea un recorrido respecto de la recta que une sus extremos.

    Devuelve la distancia perpendicular máxima entre el recorrido y esa recta,
    dividida por la longitud de la recta (adimensional, así vale igual para
    pelos largos y cortos). Un pelo recto da casi 0; un recorrido que se desvía
    hacia un pelo vecino da un valor alto aunque la relación entre distancia
    directa y recorrido siga siendo cercana a 1.
    """
    pts = np.asarray(path, dtype=float)
    if len(pts) < 3:
        return 0.0
    extremo_a, extremo_b = pts[0], pts[-1]
    cuerda = extremo_b - extremo_a
    largo_cuerda = float(np.hypot(cuerda[0], cuerda[1]))
    if largo_cuerda < 1e-6:
        return float("inf")  # empieza y termina en el mismo punto: no es un pelo
    # distancia punto-recta vía producto cruz en 2D
    rel = pts - extremo_a
    desvios = np.abs(rel[:, 0] * cuerda[1] - rel[:, 1] * cuerda[0]) / largo_cuerda
    return float(desvios.max() / largo_cuerda)


def _prune_spurs(skel: np.ndarray, n: int) -> np.ndarray:
    """Acorta todas las ramas del esqueleto n píxeles quitando extremos.

    Las espuelas cortas (ruido de la esqueletización) desaparecen del todo; los
    cruces reales entre pelos sobreviven.
    """
    out = skel.copy()
    for _ in range(int(n)):
        counts = convolve(out.astype(np.uint8), _NEIGHBOR_KERNEL, mode="constant")
        endpoints = out & (counts <= 1)
        if not endpoints.any():
            break
        out = out & ~endpoints
    return out


def _free_endpoints(sub: np.ndarray) -> list:
    """Píxeles del esqueleto con un solo vecino: puntas de pelo candidatas.

    Cuenta vecinos con una convolución en vez de recorrer píxel por píxel.
    """
    neighbors = convolve(sub.astype(np.uint8), _NEIGHBOR_KERNEL, mode="constant")
    ys, xs = np.nonzero(sub & (neighbors == 1))
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def _trace_back(tip, dist: dict, pts: set) -> list:
    """Reconstruye el camino desde la punta hasta la base siguiendo distancias decrecientes."""
    path = [tip]
    cur = tip
    while dist[cur] > 0:
        candidates = [
            (cur[0] + dx, cur[1] + dy)
            for dx, dy in _NEIGHBOR_OFFSETS
            if (cur[0] + dx, cur[1] + dy) in dist
        ]
        if not candidates:
            break
        nxt = min(candidates, key=lambda p: dist[p])
        if dist[nxt] >= dist[cur]:
            break
        path.append(nxt)
        cur = nxt
    return path


def draw_overlay(gray: np.ndarray, hairs: list[dict], root_mask: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor((gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay[dilation(root_mask, disk(1)) & ~root_mask] = (255, 200, 0)
    for i, hair in enumerate(hairs, start=1):
        for (x, y) in hair["path"]:
            cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)
        bx, by = hair["base"]
        tx, ty = hair["tip"]
        cv2.circle(overlay, (bx, by), 4, (0, 255, 0), 1)
        cv2.circle(overlay, (tx, ty), 4, (255, 0, 255), 1)
        cv2.putText(overlay, f"{i}:{hair['length_px']:.0f}", (tx + 5, ty - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1, cv2.LINE_AA)
    return overlay


def process(path: Path, crop: tuple | None = None, params: dict | None = None, quiet: bool = False):
    params = params or load_params()
    gray = load_gray(path)
    suffix = ""
    if crop:
        y0, y1, x0, x1 = crop
        gray = gray[y0:y1, x0:x1]
        suffix = "_crop"

    root_mask = segment_root_body(gray)
    ridges = compute_ridges(gray, p(params, "sato_sigmas"))
    hair_mask = detect_hair_mask(gray, root_mask, params, ridges=ridges)
    hairs = measure_hairs(hair_mask, root_mask, params, ridges=ridges, gray=gray)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = path.stem + suffix

    csv_path = RESULTS_DIR / f"{stem}_hairs.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "hair_id", "length_px", "length_um", "base_x", "base_y", "tip_x", "tip_y", "cortado"])
        for i, hair in enumerate(hairs, start=1):
            writer.writerow([stem, i, round(hair["length_px"], 2), round(hair["length_px"] * UM_PER_PX, 2),
                             hair["base"][0], hair["base"][1], hair["tip"][0], hair["tip"][1],
                             int(hair["truncated"])])

    overlay_path = RESULTS_DIR / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), draw_overlay(gray, hairs, root_mask))

    lengths = [h["length_px"] for h in hairs]
    if lengths:
        lengths_um = [l * UM_PER_PX for l in lengths]
        print(f"{stem}: {len(hairs)} pelos | mediana {np.median(lengths_um):.1f} µm | "
              f"rango {min(lengths_um):.1f}-{max(lengths_um):.1f} µm")
    else:
        print(f"{stem}: sin pelos detectados")
    return hairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--crop", nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"))
    args = parser.parse_args()
    for image in args.images:
        process(image, tuple(args.crop) if args.crop else None)
