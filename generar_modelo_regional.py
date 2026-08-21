# -*- coding: utf-8 -*-
"""
Genera una version del clasificador ajustada a la region donde se instala
el dispositivo: a las 193 neuronas locales (las unicas que este proyecto
entrena y controla) se les suma un sesgo (bias) proporcional al logaritmo
de su frecuencia real de observacion en la region, para que especies
localmente comunes le ganen mas facil a especies localmente raras cuando
el sonido es ambiguo -- sin descartar nada por completo: el ajuste esta
acotado (DELTA_MAX) y ninguna especie queda con probabilidad cero.

No toca la capa de decision original de BirdNET (6522 especies globales):
solo ajusta las 193 propias, que son las que este proyecto construye a
mano con pesos (W, b) explicitos.

IMPORTANTE sobre la fuente de datos: la API publica de eBird NO expone la
estadistica de frecuencia por especie (eso es una funcion del sitio web,
"bar chart", que requiere sesion logueada y esta sujeta a terminos de uso
que restringen el uso comercial sin un acuerdo de licencia aparte). Para
no atar el dispositivo -- ni un eventual producto comercial -- a esos
terminos, ESTE SCRIPT NUNCA HACE NINGUN LLAMADO A EBIRD. En cambio, usa
un archivo de bar chart ya descargado a mano (mismo formato que se usa
desde el principio de este proyecto, ver ebird_ranking.csv/informe), que
el equipo baja una vez por region desde su propia cuenta de eBird y
versiona en este repositorio bajo frecuencias/<REGION>.txt. El
dispositivo en el campo nunca se conecta a eBird: solo lee el archivo ya
incluido que corresponda a su region.

Paso 1: reverse geocoding (lat, lon) -> codigo de region tipo ISO 3166-2
(ej. "AR-B"), via Nominatim/OpenStreetMap (datos OpenStreetMap, licencia
abierta, sin restriccion de uso comercial, gratis, sin API key).

Paso 2: buscar frecuencias/<codigo>.txt en este repositorio. Si no existe
(todavia no se descargo esa region a mano), el script sale sin generar
nada y el dispositivo sigue con el modelo universal sin ajustar.

Paso 3: parsear el archivo de bar chart (formato de exportacion de
eBird: una fila "Sample Size" con 48 valores quincenales, despues una
fila por especie con nombre comun en ingles + 48 frecuencias
quincenales). Se promedia a frecuencia anual por especie. El cruce
nombre comun -> nombre cientifico se hace contra el archivo de labels
global de BirdNET (ya instalado localmente con birdnet_analyzer, mismo
formato "Nombre cientifico_Nombre comun" para las ~6522 especies), asi
que tampoco hace falta ningun llamado de red para esto.

Paso 4: ajuste de bias, acotado:
    delta_b_especie = clip(ALPHA * (ln(frecuencia_especie) - ln(frecuencia_geomedia)), -DELTA_MAX, +DELTA_MAX)
Centrado en la media geometrica de las 193 para que el ajuste, en
promedio, ni infle ni desinfle el conjunto completo -- solo reordena
relativo, que es la intencion. Especies de las 193 que no aparecen en el
archivo de la region (o aparecen con frecuencia 0 en las 48 columnas)
reciben un piso bajo, nunca cero ni -infinito.

Paso 5: reexporta el .tflite en modo Append, mismo pipeline ya usado y
validado (build_linear_classifier + save_linear_classifier), con
b_ajustado en vez de b.

Robustez: si CUALQUIER paso falla (geocoding, archivo de region
faltante, parseo), el script aborta sin generar nada, dejando el modelo
universal (sin ajuste regional) como esta -- nunca deja el dispositivo
peor de lo que estaba, y nunca hace un solo request a eBird.
"""
import argparse
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PESOS_PATH = os.path.join(SCRIPT_DIR, "modelo", "pesos_193_locales.npz")
FRECUENCIAS_DIR = os.path.join(SCRIPT_DIR, "frecuencias")
OUT_DIR_DEFAULT = os.path.join(SCRIPT_DIR, "modelo_regional")

ALPHA_DEFAULT = 0.6
DELTA_MAX = 4.0          # tope del ajuste, en unidades de logit
FLOOR_FRACCION = 0.05    # piso de frecuencia (respecto de la minima real observada) para
                         # especies ausentes del archivo de la region, nunca cero
TIMEOUT_RED = 20
USER_AGENT = "LSDTector-BirdNET-retrain-bsas/1.0 (contacto: LSDArroyoGold)"


def log(msg):
    print(f"[modelo-regional] {msg}", flush=True)


def reverse_geocode(lat, lon):
    """(lat, lon) -> codigo de region tipo ISO 3166-2 (ej. 'AR-B'), via Nominatim.
    Datos OpenStreetMap (ODbL), sin restriccion de uso comercial."""
    url = (
        "https://nominatim.openstreetmap.org/reverse?"
        + urllib.parse.urlencode({"lat": lat, "lon": lon, "format": "json", "zoom": 8})
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_RED) as resp:
        data = json.load(resp)
    codigo = data.get("address", {}).get("ISO3166-2-lvl4")
    if not codigo:
        raise RuntimeError(f"Nominatim no devolvio ISO3166-2-lvl4 para ({lat},{lon}): {data.get('address')}")
    return codigo


def cargar_crosswalk_nombres(labels_file):
    """Nombre comun en ingles (en minuscula) -> nombre cientifico, a partir
    del archivo de labels global de BirdNET ("SciName_CommonName" por
    linea), ya instalado localmente. Sin red."""
    crosswalk = {}
    with open(labels_file, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if "_" not in linea:
                continue
            sci, comun = linea.split("_", 1)
            crosswalk[comun.strip().lower()] = sci.strip()
    return crosswalk


def parsear_barchart(path):
    """Parsea un archivo de bar chart de eBird (exportacion TXT: fila de
    encabezado, fila 'Sample Size' con 48 valores, despues una fila por
    especie con nombre comun + 48 frecuencias quincenales). Devuelve
    {nombre_comun_lower: frecuencia_anual_promedio}."""
    frecuencias = {}
    with open(path, encoding="utf-8") as f:
        for linea in f:
            partes = linea.rstrip("\n").split("\t")
            if len(partes) < 2:
                continue
            nombre = partes[0].strip()
            if not nombre or nombre.lower().startswith("sample size"):
                continue
            valores = []
            for v in partes[1:]:
                v = v.strip()
                if not v:
                    continue
                try:
                    valores.append(float(v))
                except ValueError:
                    valores = None
                    break
            if not valores:
                continue
            frecuencias[nombre.lower()] = sum(valores) / len(valores)
    if not frecuencias:
        raise RuntimeError(f"No se pudo parsear ninguna especie de {path}, formato inesperado")
    return frecuencias


def calcular_frecuencias(frec_por_nombre_comun, crosswalk, especies_193):
    """especies_193 son nombres cientificos (nuestras etiquetas). Se
    busca el nombre comun de cada una via el crosswalk, y su frecuencia
    en el archivo de la region. Piso para las que no matchean o dan 0."""
    frecuencias = {}
    no_encontradas = []
    comun_por_especie = {}
    inv_crosswalk = {}
    for comun_lower, sci in crosswalk.items():
        inv_crosswalk.setdefault(sci, comun_lower)

    for esp in especies_193:
        comun = inv_crosswalk.get(esp)
        comun_por_especie[esp] = comun
        val = frec_por_nombre_comun.get(comun) if comun else None
        frecuencias[esp] = val if val and val > 0 else None
        if frecuencias[esp] is None:
            no_encontradas.append(esp)

    positivas = [v for v in frecuencias.values() if v is not None]
    piso = (min(positivas) * FLOOR_FRACCION) if positivas else 0.001
    piso = max(piso, 1e-5)

    for esp in especies_193:
        if frecuencias[esp] is None:
            frecuencias[esp] = piso

    return frecuencias, no_encontradas, comun_por_especie


def calcular_ajuste_bias(frecuencias, especies, alpha, delta_max):
    log_frecs = np.array([math.log(frecuencias[e]) for e in especies], dtype=np.float64)
    centro = log_frecs.mean()
    delta = alpha * (log_frecs - centro)
    delta = np.clip(delta, -delta_max, delta_max)
    return delta.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--barchart", default=None,
                     help="Ruta a un archivo de bar chart especifico. Si se omite, se busca "
                          "frecuencias/<region>.txt segun el reverse geocoding de lat/lon.")
    ap.add_argument("--frecuencias-raw-base", default=None,
                     help="URL base (ej. raw.githubusercontent.com/.../frecuencias) de donde "
                          "descargar frecuencias/<region>.txt si no existe localmente. Evita "
                          "tener que traer los archivos de TODAS las regiones al dispositivo, "
                          "solo la que corresponde. Sin llamados a eBird en ningun caso.")
    ap.add_argument("--labels-file", default=None,
                     help="Archivo de labels global de BirdNET (SciName_CommonName). "
                          "Si se omite, se usa el que trae instalado birdnet_analyzer.")
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    ap.add_argument("--pesos", default=PESOS_PATH)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--out-nombre", default="LSDTector_Classifier_regional")
    args = ap.parse_args()

    log(f"Cargando pesos base: {args.pesos}")
    data = np.load(args.pesos, allow_pickle=True)
    W = data["W"].astype(np.float32)
    b = data["b"].astype(np.float32)
    especies = [str(e) for e in data["especies"]]
    log(f"{len(especies)} especies, W {W.shape}, b {b.shape}")

    log(f"Reverse geocoding ({args.lat}, {args.lon})...")
    try:
        region_code = reverse_geocode(args.lat, args.lon)
    except Exception as e:
        log(f"ERROR en reverse geocoding: {e}. Abortando sin generar nada.")
        sys.exit(1)
    log(f"Region detectada: {region_code}")

    barchart_path = args.barchart or os.path.join(FRECUENCIAS_DIR, f"{region_code}.txt")
    if not os.path.exists(barchart_path) and args.frecuencias_raw_base:
        url = f"{args.frecuencias_raw_base.rstrip('/')}/{region_code}.txt"
        log(f"No hay archivo local para {region_code}, probando descargarlo de {url} "
            f"(archivo ya versionado por el equipo, sin llamados a eBird)...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT_RED) as resp:
                contenido = resp.read()
            os.makedirs(args.out_dir, exist_ok=True)
            descargado_path = os.path.join(args.out_dir, f"frecuencias_{region_code}.txt")
            with open(descargado_path, "wb") as f:
                f.write(contenido)
            barchart_path = descargado_path
            log(f"Descargado OK: {barchart_path}")
        except Exception as e:
            log(f"No se pudo descargar frecuencias para {region_code} ({e}). "
                f"Nada que ajustar todavia para esta region. Abortando sin generar nada (no es un error).")
            sys.exit(2)

    if not os.path.exists(barchart_path):
        log(f"No hay archivo de frecuencias para {region_code} ({barchart_path}). "
            f"Nada que ajustar todavia para esta region. Abortando sin generar nada (no es un error).")
        sys.exit(2)
    log(f"Usando archivo de frecuencias: {barchart_path}")

    if args.labels_file:
        labels_file = args.labels_file
    else:
        import birdnet_analyzer.config as cfg
        labels_file = cfg.BIRDNET_LABELS_FILE
        if not os.path.isabs(labels_file):
            import birdnet_analyzer
            labels_file = os.path.join(os.path.dirname(birdnet_analyzer.__file__), labels_file)
    log(f"Crosswalk de nombres desde: {labels_file}")
    crosswalk = cargar_crosswalk_nombres(labels_file)

    frec_por_nombre_comun = parsear_barchart(barchart_path)
    log(f"{len(frec_por_nombre_comun)} especies parseadas del archivo de frecuencias")

    frecuencias, no_encontradas, comun_por_especie = calcular_frecuencias(frec_por_nombre_comun, crosswalk, especies)
    log(f"{len(especies) - len(no_encontradas)}/{len(especies)} de las 193 especies matchearon "
        f"con una frecuencia real en {region_code}")
    if no_encontradas:
        log(f"Sin match (quedan con piso minimo): {', '.join(no_encontradas[:15])}"
            + (" ..." if len(no_encontradas) > 15 else ""))

    delta_b = calcular_ajuste_bias(frecuencias, especies, args.alpha, DELTA_MAX)
    b_ajustado = b + delta_b

    orden = np.argsort(delta_b)
    log("Mayores refuerzos (especies mas frecuentes localmente):")
    for i in orden[::-1][:5]:
        log(f"  {especies[i]:32} frec={frecuencias[especies[i]]:.4f}  delta_b={delta_b[i]:+.2f}")
    log("Mayores atenuaciones (especies menos frecuentes localmente):")
    for i in orden[:5]:
        log(f"  {especies[i]:32} frec={frecuencias[especies[i]]:.4f}  delta_b={delta_b[i]:+.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "ajuste_regional_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "region_code": region_code,
            "lat": args.lat, "lon": args.lon,
            "alpha": args.alpha, "delta_max": DELTA_MAX,
            "barchart_usado": barchart_path,
            "generado_epoch": time.time(),
            "frecuencias": frecuencias,
            "sin_match": no_encontradas,
            "delta_b": {especies[i]: float(delta_b[i]) for i in range(len(especies))},
        }, f, indent=2, ensure_ascii=False)
    log(f"Metadata guardada: {meta_path}")

    log("Exportando .tflite con bias ajustado (modo Append, mismo pipeline validado)...")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    from birdnet_analyzer import model as bn_model
    import birdnet_analyzer.config as cfg
    cfg.LABELS_FILE = cfg.BIRDNET_LABELS_FILE

    classifier = bn_model.build_linear_classifier(len(especies), W.shape[0], hidden_units=0, dropout=0.0)
    dense_layer = None
    for layer in classifier.layers:
        if layer.__class__.__name__ == "Dense":
            dense_layer = layer
            break
    assert dense_layer is not None
    dense_layer.set_weights([W, b_ajustado])
    classifier.pop()

    out_path = os.path.join(args.out_dir, args.out_nombre)
    bn_model.save_linear_classifier(classifier, out_path, especies, mode="append")
    log(f"Listo. Modelo regional guardado en: {out_path}.tflite (region {region_code})")


if __name__ == "__main__":
    main()
