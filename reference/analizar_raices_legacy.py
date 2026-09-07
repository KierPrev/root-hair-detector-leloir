import cv2
import numpy as np
import pandas as pd
import os
import glob
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from skimage.morphology import skeletonize

try:
    import zxingcpp
except ImportError:
    zxingcpp = None

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DIRECTORIO_RAIZ = str(BASE_DIR / "data" / "experiments")
ARCHIVO_SALIDA = "root_lengths.csv"
ARCHIVO_CONFIG = str(Path(__file__).resolve().with_name("parametros_calibracion.json"))
GUARDAR_IMAGENES_TESTIGO = False
CARPETA_TESTIGOS = "procesadas_debug"
CARPETA_CACHE_IMAGENES = ".analisis_cache"
EXTENSION_CACHE_IMAGEN = ".png"
HEIC_SUFFIXES = {".heic", ".heif"}

LARGO_REAL_REFERENCIA_CM = 11.09
LARGO_REAL_REFERENCIA_MM = LARGO_REAL_REFERENCIA_CM * 10.0
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
A4_WIDTH_PX = 2480
A4_HEIGHT_PX = 3508
ETIQUETA_PAGE_MARGIN_X = 120
ETIQUETA_PAGE_MARGIN_Y = 120
ETIQUETA_COLS = 3
ETIQUETA_ROWS = 9
ETIQUETA_GAP_X = 40
ETIQUETA_GAP_Y = 30


def calcular_medidas_etiqueta_mm():
    usable_width = A4_WIDTH_PX - (2 * ETIQUETA_PAGE_MARGIN_X)
    usable_height = A4_HEIGHT_PX - (2 * ETIQUETA_PAGE_MARGIN_Y)
    cell_w_px = (usable_width - ((ETIQUETA_COLS - 1) * ETIQUETA_GAP_X)) / ETIQUETA_COLS
    cell_h_px = (usable_height - ((ETIQUETA_ROWS - 1) * ETIQUETA_GAP_Y)) / ETIQUETA_ROWS
    cell_w_mm = cell_w_px * (A4_WIDTH_MM / A4_WIDTH_PX)
    cell_h_mm = cell_h_px * (A4_HEIGHT_MM / A4_HEIGHT_PX)
    return cell_w_mm, cell_h_mm


ETIQUETA_ANCHO_MM, ETIQUETA_ALTO_MM = calcular_medidas_etiqueta_mm()
ETIQUETA_LADO_MAYOR_MM = max(ETIQUETA_ANCHO_MM, ETIQUETA_ALTO_MM)
QR_LADO_REAL_MM = 17.274193548387096
MAX_DIM_ANALISIS = 2600
BLACK_BG_GRAY_MAX = 95
BLACK_BG_INSET_PX = 18
QR_BOTTOM_EXCLUSION_START = 0.76
QR_BOTTOM_EXCLUSION_MARGIN_PX = 6
MAX_ROOT_LENGTH_MM = 120.0

# Valores por defecto
PARAMS = {
    "blur_k": 5,
    "clean_k": 1,  # Default bajo
    "seed_glue": 7,
    "seed_h_max": 30,
    "seed_s_min": 75,
    "seed_v_min": 65,
    "root_s_max": 65,
    "root_v_min": 150,
    "min_area_seed": 90,
    "root_bridge": 3,
    "min_area_root": 30,  # Default bajo para raíces pequeñas
    "hsv_h_min": 15,
    "hsv_h_max": 40,
}

# ==========================================
# GESTIÓN DE CONFIGURACIÓN
# ==========================================


def cargar_configuracion(archivo_config=ARCHIVO_CONFIG):
    global PARAMS
    if os.path.exists(archivo_config):
        try:
            with open(archivo_config, "r") as f:
                datos = json.load(f)
                for k, v in datos.items():
                    if k in PARAMS:
                        PARAMS[k] = v
            print(f"\n[INFO] Configuración cargada desde '{archivo_config}'")
        except Exception as e:
            print(f"\n[ALERTA] Error leyendo config: {e}")
    else:
        print("\n[INFO] Usando valores por defecto.")


def guardar_configuracion(archivo_config=ARCHIVO_CONFIG):
    try:
        with open(archivo_config, "w") as f:
            json.dump(PARAMS, f, indent=4)
        print(f"[INFO] Configuración guardada en '{archivo_config}'")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar config: {e}")


# ==========================================
# FUNCIONES AUXILIARES
# ==========================================


def recortar_bordes_blancos(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img.shape[:2]

    # El fondo útil es la cartulina negra. Detectamos la región oscura principal
    # y recortamos un poco hacia adentro para descartar el exterior claro.
    mask_black = cv2.inRange(gray, 0, BLACK_BG_GRAY_MAX)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 0.15 * float(h_img * w_img):
            x, y, w, h = cv2.boundingRect(c)
            inset = BLACK_BG_INSET_PX
            x0 = min(max(x + inset, 0), w_img - 1)
            y0 = min(max(y + inset, 0), h_img - 1)
            x1 = max(min(x + w - inset, w_img), x0 + 1)
            y1 = max(min(y + h - inset, h_img), y0 + 1)
            return img[y0:y1, x0:x1]

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    puntos = cv2.findNonZero(thresh)
    if puntos is not None:
        x, y, w, h = cv2.boundingRect(puntos)
        margen = 30
        x = max(0, x - margen)
        y = max(0, y - margen)
        w = min(w_img - x, w + (margen * 2))
        h = min(h_img - y, h + (margen * 2))
        return img[y : y + h, x : x + w]
    return img


def decodificar_imagen_desde_archivo(ruta_archivo):
    try:
        data = np.fromfile(str(ruta_archivo), dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass

    return None


def guardar_imagen_cache(ruta_cache, img):
    img_cache = redimensionar_para_analisis(img)
    cv2.imwrite(str(ruta_cache), img_cache)
    return img_cache


def cargar_imagen_cache(ruta_cache):
    img_cache = decodificar_imagen_desde_archivo(ruta_cache)
    if img_cache is None:
        return None

    img_reducida = redimensionar_para_analisis(img_cache)
    if img_reducida.shape[:2] != img_cache.shape[:2]:
        guardar_imagen_cache(ruta_cache, img_reducida)
        return img_reducida

    return img_cache


def obtener_ruta_cache_imagen(ruta_imagen):
    ruta = Path(ruta_imagen)
    return ruta.parent / CARPETA_CACHE_IMAGENES / f"{ruta.stem}{EXTENSION_CACHE_IMAGEN}"


def leer_imagen(ruta_imagen):
    ruta = Path(ruta_imagen)
    cache_ruta = None

    if ruta.suffix.lower() in HEIC_SUFFIXES:
        cache_ruta = obtener_ruta_cache_imagen(ruta)
        try:
            if (
                cache_ruta.exists()
                and cache_ruta.stat().st_mtime >= ruta.stat().st_mtime
            ):
                img_cache = cargar_imagen_cache(cache_ruta)
                if img_cache is not None:
                    return img_cache
        except OSError:
            pass

    img = decodificar_imagen_desde_archivo(ruta)
    if img is not None:
        return img

    if ruta.suffix.lower() in HEIC_SUFFIXES and shutil.which("sips"):
        if cache_ruta is None:
            cache_ruta = obtener_ruta_cache_imagen(ruta)

        cache_ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=EXTENSION_CACHE_IMAGEN, delete=False
            ) as tmp:
                temporal = Path(tmp.name)

            resultado = subprocess.run(
                [
                    "sips",
                    "-s",
                    "format",
                    EXTENSION_CACHE_IMAGEN.lstrip("."),
                    str(ruta),
                    "--out",
                    str(temporal),
                ],
                capture_output=True,
                text=True,
            )
            if resultado.returncode == 0 and temporal.exists():
                temporal.replace(cache_ruta)
                img_cache = cargar_imagen_cache(cache_ruta)
                if img_cache is not None:
                    return img_cache
        except Exception:
            pass
        finally:
            if temporal is not None:
                temporal.unlink(missing_ok=True)

    return None


def redimensionar_para_analisis(img, max_dim=MAX_DIM_ANALISIS):
    if img is None:
        return None

    h, w = img.shape[:2]
    mayor = max(h, w)
    if mayor <= max_dim:
        return img

    escala = max_dim / float(mayor)
    return cv2.resize(
        img, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA
    )


def obtener_largo_esqueleto(img_binaria_radicula):
    bool_image = img_binaria_radicula > 0
    skeleton = skeletonize(bool_image)
    return np.count_nonzero(skeleton)


def obtener_largo_esqueleto_contorno(cnt, shape_img, margen=2):
    x, y, w, h = cv2.boundingRect(cnt)
    x0 = max(x - margen, 0)
    y0 = max(y - margen, 0)
    x1 = min(x + w + margen, shape_img[1])
    y1 = min(y + h + margen, shape_img[0])

    mask_single = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cnt_local = cnt.astype(np.int32, copy=True)
    cnt_local[:, :, 0] -= x0
    cnt_local[:, :, 1] -= y0
    cv2.drawContours(mask_single, [cnt_local], -1, 255, -1)
    return obtener_largo_esqueleto(mask_single)


def detectar_mascara_semillas(hsv, mask_total, params):
    h, s, v = cv2.split(hsv)

    # Las semillas son cálidas y saturadas; las radículas son muy claras y
    # poco saturadas. Aprovechamos esa diferencia para separar ambas clases.
    mask_semillas_color = np.where(
        ((h <= params["seed_h_max"]) | (h >= 170))
        & (s >= params["seed_s_min"])
        & (v >= params["seed_v_min"]),
        255,
        0,
    ).astype(np.uint8)

    mask_radiculas_blancas = np.where(
        (s <= params["root_s_max"]) & (v >= params["root_v_min"]),
        255,
        0,
    ).astype(np.uint8)

    mask_semillas = cv2.bitwise_and(mask_semillas_color, mask_total)
    mask_semillas = cv2.bitwise_and(
        mask_semillas, cv2.bitwise_not(mask_radiculas_blancas)
    )

    k_glue = (
        params["seed_glue"] if params["seed_glue"] % 2 == 1 else params["seed_glue"] + 1
    )
    if k_glue < 1:
        k_glue = 1

    kernel_seed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_glue, k_glue))
    mask_semillas = cv2.morphologyEx(mask_semillas, cv2.MORPH_CLOSE, kernel_seed)
    mask_semillas = cv2.morphologyEx(
        mask_semillas, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )

    return mask_semillas


def extraer_semillas_desde_mascara(mask_semillas, area_minima):
    detalles_semillas = []
    cnts_sem, _ = cv2.findContours(
        mask_semillas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for i, cnt in enumerate(cnts_sem):
        area = cv2.contourArea(cnt)
        if area < area_minima:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h if h > 0 else 0
        if not (0.35 < aspect < 3.5):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue

        solidez = float(area) / hull_area
        if solidez < 0.55:
            continue

        detalles_semillas.append(
            {
                "id": i,
                "area_px": area,
                "bbox": (x, y, w, h),
                "contour": cnt,
            }
        )

    return detalles_semillas


def limite_zona_qr_inferior(alto_img):
    return int(alto_img * QR_BOTTOM_EXCLUSION_START)


def excluir_zona_qr_inferior(mask):
    mask_filtrada = mask.copy()
    y_inicio = limite_zona_qr_inferior(mask.shape[0])
    mask_filtrada[y_inicio:, :] = 0
    return mask_filtrada, y_inicio


def contorno_toca_zona_qr_inferior(y, h, y_limite):
    return (y + h) >= (y_limite - QR_BOTTOM_EXCLUSION_MARGIN_PX)


def rotar_imagen_90(img, rotaciones):
    rotaciones = rotaciones % 4
    if rotaciones == 0:
        return img
    if rotaciones == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotaciones == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def recortar_margen(img, top=0.0, bottom=0.0, left=0.0, right=0.0):
    h, w = img.shape[:2]
    y0 = min(max(int(h * top), 0), h - 1)
    y1 = max(min(int(h * (1.0 - bottom)), h), y0 + 1)
    x0 = min(max(int(w * left), 0), w - 1)
    x1 = max(min(int(w * (1.0 - right)), w), x0 + 1)
    return img[y0:y1, x0:x1]


def normalizar_tamano_qr(img, max_dim=1100):
    h, w = img.shape[:2]
    mayor = max(h, w)
    if mayor <= max_dim:
        return img

    escala = max_dim / float(mayor)
    return cv2.resize(img, (0, 0), fx=escala, fy=escala, interpolation=cv2.INTER_AREA)


def generar_regiones_qr(img):
    bases = [img]
    img_crop = recortar_bordes_blancos(img)
    if img_crop is not None and img_crop.shape != img.shape:
        bases.append(img_crop)

    regiones = []
    for base in bases:
        regiones.extend(
            [
                recortar_margen(base, top=0.70, left=0.20, right=0.20),
                recortar_margen(base, top=0.76, left=0.25, right=0.25),
                recortar_margen(base, top=0.80, left=0.30, right=0.30),
                recortar_margen(base, left=0.78, top=0.16, bottom=0.12),
                recortar_margen(base, left=0.76, top=0.12, bottom=0.10),
                recortar_margen(base, left=0.72, top=0.10, bottom=0.10),
                recortar_margen(base, right=0.78, top=0.16, bottom=0.12),
                recortar_margen(base, right=0.76, top=0.12, bottom=0.10),
                base,
            ]
        )

    return [region for region in regiones if region is not None and region.size > 0]


def generar_variantes_qr(img):
    base = normalizar_tamano_qr(img)
    variantes = [base]

    gris = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    variantes.append(gris)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
    variantes.append(clahe)

    _, otsu = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(otsu)

    adapt = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    variantes.append(adapt)

    nitida = cv2.GaussianBlur(base, (0, 0), 3)
    nitida = cv2.addWeighted(base, 1.6, nitida, -0.6, 0)
    variantes.append(nitida)

    escalas = [1.8]
    variantes_escaladas = []
    for variante in variantes[:3]:
        for escala in escalas:
            variantes_escaladas.append(
                cv2.resize(
                    variante,
                    (0, 0),
                    fx=escala,
                    fy=escala,
                    interpolation=cv2.INTER_CUBIC,
                )
            )

    return variantes + variantes_escaladas


def normalizar_tray_id_qr(texto_qr, exp_id):
    if texto_qr is None:
        return None

    texto = str(texto_qr).strip().upper()
    if not texto:
        return None

    texto = texto.replace("-", "_")
    texto = re.sub(r"\s+", "", texto)

    match_corto = re.fullmatch(r"([A-Z])(\d{1,2})", texto)
    if match_corto:
        fila, posicion = match_corto.groups()
        return f"{exp_id}_{fila}{int(posicion)}"

    match_largo = re.fullmatch(r"(\d{8})_?([A-Z])(\d{1,2})", texto)
    if match_largo:
        exp_qr, fila, posicion = match_largo.groups()
        return f"{exp_qr}_{fila}{int(posicion)}"

    return None


def leer_textos_qr(candidata, detector):
    textos = []

    if zxingcpp is not None:
        try:
            resultados = zxingcpp.read_barcodes(
                candidata, formats=zxingcpp.BarcodeFormat.QRCode
            )
            textos.extend(
                [resultado.text for resultado in resultados if resultado.text]
            )
        except Exception:
            pass

    try:
        detectado, textos_multi, _, _ = detector.detectAndDecodeMulti(candidata)
        if detectado:
            textos.extend(textos_multi)
    except Exception:
        pass

    try:
        texto_simple, _, _ = detector.detectAndDecode(candidata)
        if texto_simple:
            textos.append(texto_simple)
    except Exception:
        pass

    return textos


def detectar_qr_rapido(region, detector):
    preview = normalizar_tamano_qr(region, max_dim=2000)
    escala = region.shape[1] / float(preview.shape[1])
    variantes = [preview, cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)]

    for variante in variantes:
        for rotacion in (0, 1, 3):
            candidata = rotar_imagen_90(variante, rotacion)
            texto_simple = None
            puntos = None
            texto_multi = None
            puntos_multi = None

            try:
                texto_simple, puntos, _ = detector.detectAndDecode(candidata)
            except Exception:
                texto_simple = None
                puntos = None

            try:
                detectado_multi, textos_multi, puntos_multi, _ = (
                    detector.detectAndDecodeMulti(candidata)
                )
                if detectado_multi and textos_multi:
                    for texto_candidato in textos_multi:
                        if texto_candidato:
                            texto_multi = texto_candidato
                            break
            except Exception:
                texto_multi = None
                puntos_multi = None

            textos = []
            if texto_simple:
                textos.append(texto_simple)
            if texto_multi:
                textos.append(texto_multi)
            textos.extend(leer_textos_qr(candidata, detector))

            puntos_validos = None
            if puntos is not None and len(puntos) > 0:
                puntos_validos = puntos[0]
            elif puntos_multi is not None and len(puntos_multi) > 0:
                puntos_validos = puntos_multi[0]

            for texto in textos:
                if texto:
                    return texto, puntos_validos, escala

    return None, None, None


def calcular_lado_promedio_qr(puntos):
    if puntos is None or len(puntos) != 4:
        return None

    lados = []
    for i in range(4):
        p1 = puntos[i]
        p2 = puntos[(i + 1) % 4]
        lados.append(float(np.linalg.norm(p1 - p2)))

    if not lados:
        return None

    return sum(lados) / len(lados)


def detectar_info_qr(img, exp_id=None):
    detector = cv2.QRCodeDetector()
    regiones = generar_regiones_qr(img)
    tray_id = None
    px_por_mm_qr = None

    for region in regiones[:5]:
        texto, puntos, escala = detectar_qr_rapido(region, detector)

        if tray_id is None:
            tray_id = normalizar_tray_id_qr(texto, exp_id) if exp_id else None

        if px_por_mm_qr is None and texto and puntos is not None and escala is not None:
            lado_promedio = calcular_lado_promedio_qr(puntos)
            if lado_promedio is not None and lado_promedio > 0:
                px_por_mm_qr = (lado_promedio * escala) / QR_LADO_REAL_MM

        if tray_id is not None and px_por_mm_qr is not None:
            return tray_id, px_por_mm_qr

    if exp_id is None or tray_id is not None:
        return tray_id, px_por_mm_qr

    for region_idx, region in enumerate(regiones):
        rotaciones = (0, 1, 3) if region_idx < 5 else (0, 1)
        for variante in generar_variantes_qr(region):
            for rotacion in rotaciones:
                candidata = rotar_imagen_90(variante, rotacion)
                for texto in leer_textos_qr(candidata, detector):
                    tray_id = normalizar_tray_id_qr(texto, exp_id)
                    if tray_id:
                        return tray_id, px_por_mm_qr

    return tray_id, px_por_mm_qr


def detectar_tray_id(img, exp_id):
    tray_id, _ = detectar_info_qr(img, exp_id)
    return tray_id


def detectar_referencia_qr(img):
    _, px_por_mm_qr = detectar_info_qr(img)
    return px_por_mm_qr, px_por_mm_qr is not None


def detectar_referencia_etiqueta(img):
    for region in generar_regiones_qr(img)[:5]:
        preview = normalizar_tamano_qr(region, max_dim=1600)
        escala = region.shape[1] / float(preview.shape[1])

        hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 80, 255]))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        mejor_largo_px = 0.0
        mejor_score = 0.0

        alto, ancho = mask.shape[:2]
        area_total = float(alto * ancho)
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            toca_borde = x == 0 or y == 0 or (x + w) >= ancho or (y + h) >= alto
            if toca_borde or area < 5000:
                continue

            cobertura = area / area_total
            if cobertura < 0.005 or cobertura > 0.25:
                continue

            puntos = np.column_stack(np.where(labels == label))
            puntos = np.flip(puntos, axis=1).astype(np.float32)
            rect = cv2.minAreaRect(puntos)
            largo = max(rect[1])
            ancho_rect = min(rect[1])
            if ancho_rect <= 0:
                continue

            aspect_ratio = largo / ancho_rect
            if aspect_ratio < 1.3 or aspect_ratio > 3.0:
                continue

            score = area * aspect_ratio
            if score > mejor_score:
                mejor_score = score
                mejor_largo_px = largo * escala

        if mejor_largo_px > 0:
            return mejor_largo_px / ETIQUETA_LADO_MAYOR_MM, True

    return None, False


def detectar_referencia(hsv, img_debug, params, img_original=None, px_por_mm_qr=None):
    if px_por_mm_qr is None and img_original is not None:
        px_por_mm_qr, _ = detectar_referencia_qr(img_original)

    if px_por_mm_qr is not None:
        if img_debug is not None:
            cv2.putText(
                img_debug,
                "REF: QR",
                (20, img_debug.shape[0] - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        return px_por_mm_qr, True

    if img_original is not None:
        px_por_mm_label, ref_label = detectar_referencia_etiqueta(img_original)
        if ref_label and px_por_mm_label is not None:
            if img_debug is not None:
                cv2.putText(
                    img_debug,
                    "REF: LABEL",
                    (20, img_debug.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            return px_por_mm_label, True

    amarillo_min = np.array([params["hsv_h_min"], 80, 80])
    amarillo_max = np.array([params["hsv_h_max"], 255, 255])
    mask_ref_color = cv2.inRange(hsv, amarillo_min, amarillo_max)

    mask_ref_color = cv2.morphologyEx(
        mask_ref_color, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    mask_ref_color = cv2.morphologyEx(
        mask_ref_color, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )

    cnts_ref, _ = cv2.findContours(
        mask_ref_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mejor_ref = None
    max_area_ref = 0

    for c in cnts_ref:
        area = cv2.contourArea(c)
        if area > 500:
            rect = cv2.minAreaRect(c)
            (center), (w_rect, h_rect), angle = rect
            largo = max(w_rect, h_rect)
            ancho = min(w_rect, h_rect)
            if ancho > 0:
                aspect_ratio = largo / ancho
                if aspect_ratio > 2.5:
                    if area > max_area_ref:
                        max_area_ref = area
                        mejor_ref = c

    if mejor_ref is not None:
        rect = cv2.minAreaRect(mejor_ref)
        if img_debug is not None:
            box = np.int64(cv2.boxPoints(rect))
            cv2.drawContours(img_debug, [box], 0, (255, 255, 0), 3)
        largo_px = max(rect[1])
        if largo_px > 0:
            return largo_px / LARGO_REAL_REFERENCIA_MM, True

    return 1.0, False


def procesar_imagen(img_original, params, px_por_mm_qr=None, generar_debug=True):
    img = recortar_bordes_blancos(img_original.copy())
    if img is None or img.size == 0:
        img = img_original.copy()
    img_debug = img.copy() if generar_debug else None
    h_img, w_img = img.shape[:2]

    # 1. Suavizado
    k_blur = params["blur_k"] if params["blur_k"] % 2 == 1 else params["blur_k"] + 1
    if k_blur < 1:
        k_blur = 1
    img_blur = cv2.GaussianBlur(img, (k_blur, k_blur), 0)

    # 2. Silueta Total
    b, g, r = cv2.split(img_blur)
    _, mask_total = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_clean = params["clean_k"]
    if k_clean > 0:
        mask_total = cv2.morphologyEx(
            mask_total, cv2.MORPH_OPEN, np.ones((k_clean, k_clean), np.uint8)
        )

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Semillas
    mask_semillas = detectar_mascara_semillas(hsv, mask_total, params)

    # 4. Raíces
    mask_radiculas = cv2.subtract(mask_total, mask_semillas)

    k_bridge = (
        params["root_bridge"]
        if params["root_bridge"] % 2 == 1
        else params["root_bridge"] + 1
    )
    if k_bridge > 1:
        kernel_connect = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_bridge, k_bridge)
        )
        mask_radiculas = cv2.morphologyEx(
            mask_radiculas, cv2.MORPH_CLOSE, kernel_connect
        )

    if k_clean > 0:
        mask_radiculas = cv2.morphologyEx(
            mask_radiculas, cv2.MORPH_OPEN, np.ones((k_clean, k_clean), np.uint8)
        )

    mask_radiculas, y_limite_qr = excluir_zona_qr_inferior(mask_radiculas)

    # 5. Calibración
    px_por_mm, ref_encontrada = detectar_referencia(
        hsv, img_debug, params, img_original=img, px_por_mm_qr=px_por_mm_qr
    )
    px_por_cm = px_por_mm * 10.0

    # Texto de escala
    if img_debug is not None and ref_encontrada:
        cv2.putText(
            img_debug,
            f"Scale: {px_por_cm:.1f} px/cm",
            (20, h_img - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
    elif img_debug is not None:
        cv2.putText(
            img_debug,
            "NO REF",
            (20, h_img - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    detalles_semillas = []
    detalles_radiculas = []

    # Procesar Semillas
    detalles_semillas = extraer_semillas_desde_mascara(
        mask_semillas, params["min_area_seed"]
    )
    if img_debug is not None:
        for semilla in detalles_semillas:
            cv2.drawContours(img_debug, [semilla["contour"]], -1, (255, 0, 0), -1)

    # Procesar Raíces
    cnts_rad, _ = cv2.findContours(
        mask_radiculas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for i, cnt in enumerate(cnts_rad):
        area = cv2.contourArea(cnt)
        if area > params["min_area_root"]:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue

            solidez = float(area) / hull_area
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / h if h > 0 else 0

            if contorno_toca_zona_qr_inferior(y, h, y_limite_qr):
                continue
            if solidez > 0.6:
                continue
            if aspect > 8.0 or aspect < 0.12:
                continue

            largo_px = obtener_largo_esqueleto_contorno(cnt, (h_img, w_img))
            largo_mm = largo_px / px_por_mm
            if largo_mm > MAX_ROOT_LENGTH_MM:
                continue

            detalles_radiculas.append(
                {
                    "id": i,
                    "length_mm": largo_mm,
                    "area_cm2": area / (px_por_cm**2),
                }
            )
            if img_debug is not None:
                cv2.drawContours(img_debug, [cnt], -1, (0, 255, 0), 2)

    # ========================================================
    #  VISUALIZACIÓN DE ESTADÍSTICAS EN TIEMPO REAL (UPDATE)
    # ========================================================
    cantidad_semillas = len(detalles_semillas)
    largos_encontrados = [r["length_mm"] for r in detalles_radiculas]

    if img_debug is not None:
        color_semillas = (255, 255, 0) if cantidad_semillas > 0 else (0, 0, 255)
        cv2.putText(
            img_debug,
            f"Semillas: {cantidad_semillas}",
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_semillas,
            2,
        )

    if img_debug is not None and largos_encontrados:
        val_min = min(largos_encontrados)
        val_max = max(largos_encontrados)
        cant = len(largos_encontrados)

        # Texto en Cyan (Azul+Verde)
        texto_stats = f"Raices: {cant} | Min: {val_min:.2f} mm | Max: {val_max:.2f} mm"
        cv2.putText(
            img_debug,
            texto_stats,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
    elif img_debug is not None:
        cv2.putText(
            img_debug,
            "Sin Raices",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return {
        "lista_semillas": detalles_semillas,
        "lista_radiculas": detalles_radiculas,
        "img_debug": img_debug,
        "ref_encontrada": ref_encontrada,
        "px_por_mm": px_por_mm,
        "px_por_cm": px_por_cm,
    }


# ==========================================
# PANEL DE CONTROL (GUI)
# ==========================================


def nothing(x):
    pass


def abrir_panel_calibracion(ruta_primera_imagen, archivo_config=ARCHIVO_CONFIG):
    print("\n--- MODO CALIBRACIÓN ---")
    print("Ajusta los sliders. Presiona 'Q' para GUARDAR y procesar todo.")

    img_raw = leer_imagen(ruta_primera_imagen)
    if img_raw is None:
        print(
            f"[ALERTA] No se pudo leer la imagen para calibración: {ruta_primera_imagen}"
        )
        return

    img_raw = redimensionar_para_analisis(img_raw)

    img_crop = recortar_bordes_blancos(img_raw)

    h, w = img_crop.shape[:2]
    factor = 1.0
    if h > 900:
        factor = 900 / h
        img_crop = cv2.resize(img_crop, (0, 0), fx=factor, fy=factor)

    px_por_mm_qr, _ = detectar_referencia_qr(img_crop)
    win_name = "Panel de Calibracion"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Trackbars
    cv2.createTrackbar("Suavizado (Blur)", win_name, PARAMS["blur_k"], 15, nothing)
    cv2.createTrackbar("Hue Max Semilla", win_name, PARAMS["seed_h_max"], 60, nothing)
    cv2.createTrackbar("Sat Min Semilla", win_name, PARAMS["seed_s_min"], 255, nothing)
    cv2.createTrackbar("Val Min Semilla", win_name, PARAMS["seed_v_min"], 255, nothing)
    cv2.createTrackbar("Limpieza Ruido", win_name, PARAMS["clean_k"], 10, nothing)
    cv2.createTrackbar("Pegamento Semilla", win_name, PARAMS["seed_glue"], 20, nothing)
    cv2.createTrackbar("Sat Max Raiz", win_name, PARAMS["root_s_max"], 255, nothing)
    cv2.createTrackbar("Val Min Raiz", win_name, PARAMS["root_v_min"], 255, nothing)
    cv2.createTrackbar("Area Min Sem", win_name, PARAMS["min_area_seed"], 1000, nothing)
    cv2.createTrackbar("Puente Raices", win_name, PARAMS["root_bridge"], 10, nothing)
    cv2.createTrackbar("Area Min Raiz", win_name, PARAMS["min_area_root"], 200, nothing)
    cv2.createTrackbar("Ref Amarillo Min", win_name, PARAMS["hsv_h_min"], 179, nothing)

    ultimos_valores = None
    ultima_vista = None

    while True:
        # Leer valores
        valores_actuales = {
            "blur_k": cv2.getTrackbarPos("Suavizado (Blur)", win_name),
            "seed_h_max": cv2.getTrackbarPos("Hue Max Semilla", win_name),
            "seed_s_min": cv2.getTrackbarPos("Sat Min Semilla", win_name),
            "seed_v_min": cv2.getTrackbarPos("Val Min Semilla", win_name),
            "clean_k": cv2.getTrackbarPos("Limpieza Ruido", win_name),
            "seed_glue": cv2.getTrackbarPos("Pegamento Semilla", win_name),
            "root_s_max": cv2.getTrackbarPos("Sat Max Raiz", win_name),
            "root_v_min": cv2.getTrackbarPos("Val Min Raiz", win_name),
            "min_area_seed": cv2.getTrackbarPos("Area Min Sem", win_name),
            "root_bridge": cv2.getTrackbarPos("Puente Raices", win_name),
            "min_area_root": cv2.getTrackbarPos("Area Min Raiz", win_name),
            "hsv_h_min": cv2.getTrackbarPos("Ref Amarillo Min", win_name),
        }

        if valores_actuales != ultimos_valores or ultima_vista is None:
            PARAMS.update(valores_actuales)
            res = procesar_imagen(
                img_crop,
                PARAMS,
                px_por_mm_qr=px_por_mm_qr,
                generar_debug=True,
            )
            ultima_vista = res["img_debug"]
            ultimos_valores = valores_actuales.copy()

        if ultima_vista is not None:
            cv2.imshow(win_name, ultima_vista)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break

        try:
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyAllWindows()
    guardar_configuracion(archivo_config)
    print("\n[OK] Procesando lote completo con los nuevos parámetros...")


# ==========================================
# MAIN
# ==========================================


def redimensionar_para_ver(imagen, alto_maximo):
    h, w = imagen.shape[:2]
    if h > alto_maximo:
        ratio = alto_maximo / float(h)
        return cv2.resize(imagen, (int(w * ratio), int(h * ratio)))
    return imagen


def seleccionar_carpetas(directorio_raiz):
    if not os.path.exists(directorio_raiz):
        return []
    todas = obtener_carpetas_disponibles(directorio_raiz)
    if not todas:
        return []
    print("\nCarpetas encontradas:")
    for i, n in enumerate(todas):
        print(f"[{i+1}] {n}")
    sel = input("\nElige números (separados por coma) o ENTER para todas: ")
    if not sel.strip():
        return todas
    try:
        idxs = [int(x) - 1 for x in sel.split(",")]
        return [todas[i] for i in idxs if 0 <= i < len(todas)]
    except (ValueError, IndexError):
        return []


def obtener_carpetas_disponibles(directorio_raiz=DIRECTORIO_RAIZ):
    if not os.path.exists(directorio_raiz):
        return []
    return sorted(
        [
            d
            for d in os.listdir(directorio_raiz)
            if os.path.isdir(os.path.join(directorio_raiz, d))
        ]
    )


def obtener_directorio_fotos(ruta_experimento):
    ruta_experimento = Path(ruta_experimento)
    ruta_fotos = ruta_experimento / "raw" / "photos"
    if ruta_fotos.is_dir():
        return str(ruta_fotos)
    return str(ruta_experimento)


def obtener_archivos_imagen(ruta_carpeta):
    patrones = (
        "*.jpg",
        "*.JPG",
        "*.jpeg",
        "*.JPEG",
        "*.png",
        "*.PNG",
        "*.heic",
        "*.HEIC",
        "*.heif",
        "*.HEIF",
    )
    archivos = []
    for patron in patrones:
        archivos.extend(glob.glob(os.path.join(ruta_carpeta, patron)))
    return sorted(set(archivos))


def resolver_archivo_salida_exp(ruta_experimento, archivo_salida):
    ruta_experimento = Path(ruta_experimento)
    processed_dir = ruta_experimento / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if archivo_salida:
        ruta = Path(archivo_salida)
        if ruta.is_absolute() or len(ruta.parts) > 1:
            return str(ruta)
        return str(processed_dir / ruta)

    return str(processed_dir / ARCHIVO_SALIDA)


def resolver_carpeta_testigos_exp(ruta_experimento, carpeta_testigos):
    ruta_experimento = Path(ruta_experimento)
    processed_dir = ruta_experimento / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if carpeta_testigos:
        ruta = Path(carpeta_testigos)
        if ruta.is_absolute() or len(ruta.parts) > 1:
            return str(ruta)
        return str(processed_dir / ruta)

    return str(processed_dir / CARPETA_TESTIGOS)


def calcular_resumen_mediciones(filas_csv, semillas_detectadas):
    total_raices = len(filas_csv)
    media_largo_raiz_mm = float("nan")

    if filas_csv:
        largos = [
            float(fila["length_mm"])
            for fila in filas_csv
            if fila.get("length_mm") is not None
        ]
        if largos:
            media_largo_raiz_mm = float(np.mean(largos))

    return {
        "total_semillas": int(semillas_detectadas),
        "total_raices": int(total_raices),
        "media_largo_raiz_mm": media_largo_raiz_mm,
    }


def limpiar_outliers_lejanos(
    df,
    *,
    columna="length_mm",
    factor_iqr=6.0,
):
    if df.empty or columna not in df.columns:
        return df.copy(), pd.DataFrame(columns=df.columns), None

    df_limpio = df.copy()
    df_limpio[columna] = pd.to_numeric(df_limpio[columna], errors="coerce")
    valores = df_limpio[columna].dropna()
    if valores.empty:
        return df_limpio, pd.DataFrame(columns=df.columns), None

    q1 = float(valores.quantile(0.25))
    q3 = float(valores.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return df_limpio, pd.DataFrame(columns=df.columns), None

    umbral_superior = q3 + (factor_iqr * iqr)
    mask_outliers = df_limpio[columna] > umbral_superior
    outliers = df_limpio.loc[mask_outliers].copy()
    filtrado = df_limpio.loc[~mask_outliers].copy()

    return filtrado, outliers, umbral_superior


def imprimir_resumen_outliers(outliers, umbral_superior, *, prefijo=""):
    cantidad = len(outliers)
    if cantidad == 0 or umbral_superior is None:
        print(f"{prefijo}Outliers lejanos removidos: 0")
        return

    print(
        f"{prefijo}Outliers lejanos removidos: {cantidad} "
        f"(umbral superior length_mm > {umbral_superior:.2f})"
    )
    top_outliers = outliers.sort_values("length_mm", ascending=False).head(10)
    for _, fila in top_outliers.iterrows():
        tray_id = fila.get("tray_id", "sin_tray")
        seed_id = fila.get("seed_id", "sin_seed")
        length_mm = fila.get("length_mm", float("nan"))
        print(f"{prefijo}  - {tray_id} / {seed_id}: {length_mm:.2f} mm")


def imprimir_resumen_mediciones(resumen, *, prefijo=""):
    media = resumen["media_largo_raiz_mm"]
    media_texto = f"{media:.2f} mm" if not np.isnan(media) else "sin datos"
    print(f"{prefijo}Semillas detectadas: {resumen['total_semillas']}")
    print(f"{prefijo}Raices detectadas: {resumen['total_raices']}")
    print(f"{prefijo}Media largo de raiz: {media_texto}")


def procesar_carpetas(
    carpetas,
    directorio_raiz=DIRECTORIO_RAIZ,
    archivo_salida=None,
    carpeta_testigos=None,
    abrir_calibracion=False,
    archivo_config=ARCHIVO_CONFIG,
    guardar_imagenes_testigo=GUARDAR_IMAGENES_TESTIGO,
):
    cargar_configuracion(archivo_config)

    if not carpetas:
        print("[ALERTA] No hay carpetas seleccionadas para procesar.")
        return None

    if guardar_imagenes_testigo:
        for carpeta in carpetas:
            ruta_exp = os.path.join(directorio_raiz, carpeta)
            os.makedirs(
                resolver_carpeta_testigos_exp(ruta_exp, carpeta_testigos),
                exist_ok=True,
            )

    imagen_ejemplo = None
    for carpeta in carpetas:
        ruta_experimento = os.path.join(directorio_raiz, carpeta)
        ruta_fotos = obtener_directorio_fotos(ruta_experimento)
        archivos = obtener_archivos_imagen(ruta_fotos)
        if archivos:
            imagen_ejemplo = archivos[0]
            break

    if abrir_calibracion and imagen_ejemplo:
        abrir_panel_calibracion(imagen_ejemplo, archivo_config=archivo_config)
    elif abrir_calibracion:
        print("[ALERTA] No se encontraron imágenes para calibrar.")

    resultados = []
    columnas_csv = ["tray_id", "seed_id", "length_mm", "area_cm2"]
    total_semillas_global = 0
    filas_csv_global = []

    for carpeta in carpetas:
        ruta_experimento = os.path.join(directorio_raiz, carpeta)
        ruta_fotos = obtener_directorio_fotos(ruta_experimento)
        archivos = obtener_archivos_imagen(ruta_fotos)
        filas_csv = []
        fotos_sin_qr = []
        fotos_sin_referencia = []
        fotos_no_legibles = []
        seed_counters = {}
        semillas_detectadas_exp = 0

        print(f"\nProcesando {carpeta} ({len(archivos)} imágenes)...")

        for idx_archivo, ruta_img in enumerate(archivos, start=1):
            imagen = leer_imagen(ruta_img)
            if imagen is None:
                print(f"[ALERTA] No se pudo leer la imagen: {ruta_img}")
                fotos_no_legibles.append(os.path.basename(ruta_img))
                continue

            imagen = redimensionar_para_analisis(imagen)
            imagen = recortar_bordes_blancos(imagen)
            if imagen is None or imagen.size == 0:
                print(f"[ALERTA] No se pudo recortar el borde blanco en: {ruta_img}")
                fotos_no_legibles.append(os.path.basename(ruta_img))
                continue

            nombre_archivo = os.path.basename(ruta_img)
            print(f"  [{idx_archivo}/{len(archivos)}] {nombre_archivo}")
            tray_id, px_por_mm_qr = detectar_info_qr(imagen, carpeta)
            res = procesar_imagen(
                imagen,
                PARAMS,
                px_por_mm_qr=px_por_mm_qr,
                generar_debug=guardar_imagenes_testigo,
            )
            if not res:
                continue

            if tray_id is None:
                print(
                    f"[ALERTA] No se pudo leer un QR válido en {nombre_archivo}. Se omite."
                )
                fotos_sin_qr.append(nombre_archivo)

            if res["img_debug"] is not None:
                etiqueta = tray_id if tray_id else "NO QR"
                color = (255, 255, 0) if tray_id else (0, 0, 255)
                cv2.putText(
                    res["img_debug"],
                    etiqueta,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            if not res["ref_encontrada"]:
                print(
                    f"[ALERTA] No se encontró la referencia de escala en {nombre_archivo}. "
                    "Se omite."
                )
                fotos_sin_referencia.append(nombre_archivo)

            if tray_id is not None and res["ref_encontrada"]:
                if not tray_id.startswith(f"{carpeta}_"):
                    print(
                        f"[ALERTA] El QR de {nombre_archivo} apunta a {tray_id}, "
                        f"pero la foto está en {carpeta}."
                    )

                semillas_detectadas_exp += len(res["lista_semillas"])
                contador_actual = seed_counters.get(tray_id, 0)
                for radicula in res["lista_radiculas"]:
                    contador_actual += 1
                    filas_csv.append(
                        {
                            "tray_id": tray_id,
                            "seed_id": f"seed_{contador_actual:03d}",
                            "length_mm": radicula["length_mm"],
                            "area_cm2": radicula["area_cm2"],
                        }
                    )
                seed_counters[tray_id] = contador_actual

            if res["img_debug"] is not None and guardar_imagenes_testigo:
                carpeta_testigos_exp = resolver_carpeta_testigos_exp(
                    ruta_experimento, carpeta_testigos
                )
                nombre_unico = f"proc_{Path(nombre_archivo).stem}.png"
                cv2.imwrite(
                    os.path.join(carpeta_testigos_exp, nombre_unico), res["img_debug"]
                )

        archivo_salida_exp = resolver_archivo_salida_exp(
            ruta_experimento, archivo_salida
        )
        carpeta_salida = os.path.dirname(archivo_salida_exp)
        if carpeta_salida:
            os.makedirs(carpeta_salida, exist_ok=True)

        df = pd.DataFrame(filas_csv, columns=columnas_csv)
        df_limpio, df_outliers, umbral_outliers = limpiar_outliers_lejanos(df)
        df_limpio.to_csv(archivo_salida_exp, index=False)
        print(f"\nListo. Datos guardados en {archivo_salida_exp}")
        imprimir_resumen_outliers(df_outliers, umbral_outliers, prefijo="  ")

        filas_csv_limpias = df_limpio.to_dict("records")
        resumen_exp = calcular_resumen_mediciones(
            filas_csv_limpias, semillas_detectadas_exp
        )
        print("Resumen del experimento:")
        imprimir_resumen_mediciones(resumen_exp, prefijo="  ")

        if fotos_sin_qr:
            print(f"[INFO] Fotos omitidas por QR no legible: {', '.join(fotos_sin_qr)}")
        if fotos_sin_referencia:
            print(
                "[INFO] Fotos omitidas por referencia no detectada: "
                + ", ".join(fotos_sin_referencia)
            )
        if fotos_no_legibles:
            print(f"[INFO] Fotos no legibles: {', '.join(fotos_no_legibles)}")

        total_semillas_global += resumen_exp["total_semillas"]
        filas_csv_global.extend(filas_csv_limpias)
        resultados.append(
            {
                "exp_id": carpeta,
                "archivo_salida": archivo_salida_exp,
                "cantidad_filas": len(df_limpio),
                "cantidad_outliers_removidos": int(len(df_outliers)),
                "resumen_mediciones": resumen_exp,
                "carpeta_testigos": (
                    resolver_carpeta_testigos_exp(ruta_experimento, carpeta_testigos)
                    if guardar_imagenes_testigo
                    else None
                ),
            }
        )

    if not resultados:
        print("[ALERTA] No se generaron mediciones.")
        return None

    filas_totales = sum(r["cantidad_filas"] for r in resultados)
    resumen_global = calcular_resumen_mediciones(
        filas_csv_global, total_semillas_global
    )
    print("\nResumen global:")
    imprimir_resumen_mediciones(resumen_global, prefijo="  ")

    return {
        "carpetas_procesadas": list(carpetas),
        "experimentos": resultados,
        "cantidad_filas_total": filas_totales,
        "resumen_mediciones": resumen_global,
    }


def main():
    carpetas = seleccionar_carpetas(DIRECTORIO_RAIZ)
    if not carpetas:
        return

    procesar_carpetas(
        carpetas,
        directorio_raiz=DIRECTORIO_RAIZ,
        archivo_salida=ARCHIVO_SALIDA,
        carpeta_testigos=CARPETA_TESTIGOS,
        abrir_calibracion=True,
        archivo_config=ARCHIVO_CONFIG,
        guardar_imagenes_testigo=GUARDAR_IMAGENES_TESTIGO,
    )
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
