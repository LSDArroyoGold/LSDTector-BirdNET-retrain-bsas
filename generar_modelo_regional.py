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

Fuente de datos, en orden de preferencia:

1. Archivo de bar chart ya descargado a mano (mismo formato que se usa
   desde el principio de este proyecto, ver ebird_ranking.csv/informe),
   que el equipo baja una vez por region desde su propia cuenta de eBird
   y versiona en este repositorio bajo frecuencias/<REGION>.txt. Sin
   llamado de red a eBird (solo, como mucho, a este mismo repositorio via
   raw.githubusercontent.com para traer el archivo si el dispositivo no
   lo tiene local todavia).
2. Si no existe ese archivo para la region Y se paso una API key de eBird
   (--ebird-key, ver https://ebird.org/api/keygen), se usa la API publica
   de eBird como respaldo: observaciones recientes (ultimos 30 dias, el
   maximo que permite la API) de la region, contando en cuantos
   checklists distintos aparece cada especie como proxy de frecuencia.

   ADVERTENCIA para uso comercial: la API publica de eBird esta sujeta a
   los terminos de uso de eBird/Cornell Lab, que restringen el uso
   comercial sin un acuerdo de licencia aparte (ver el flujo de alta en
   ebird.org/api/keygen, seccion "Solicite un acuerdo de licencia"). Este
   camino se agrega a pedido explicito, para uso academico/de
   investigacion actual del proyecto. Si el dispositivo llega a
   comercializarse, revisar esto con eBird antes de seguir usando la API
   en produccion -- el camino 1 (archivos ya descargados a mano) no tiene
   esta restriccion porque nunca llama a la API en el dispositivo.

Si ninguna de las dos fuentes esta disponible, el script sale sin generar
nada y el dispositivo sigue con el modelo universal sin ajustar.

Paso 1: reverse geocoding (lat, lon) -> codigo de region tipo ISO 3166-2
(ej. "AR-B"), via Nominatim/OpenStreetMap (datos OpenStreetMap, licencia
abierta, sin restriccion de uso comercial, gratis, sin API key).

Paso 2: buscar o traer la frecuencia por especie para esa region, segun
el orden de fuentes de arriba.

Paso 3 (solo fuente 1, bar chart): el cruce nombre comun -> nombre
cientifico se hace contra el archivo de labels global de BirdNET (ya
instalado localmente con birdnet_analyzer, mismo formato "Nombre
cientifico_Nombre comun" para las ~6522 especies), sin red.

Paso 4: ajuste de bias, acotado:
    delta_b_especie = clip(ALPHA * (ln(frecuencia_especie) - ln(frecuencia_geomedia)), -DELTA_MAX, +DELTA_MAX)
Centrado en la media geometrica de las 193 para que el ajuste, en
promedio, ni infle ni desinfle el conjunto completo -- solo reordena
relativo, que es la intencion. Especies de las 193 sin frecuencia
encontrada reciben un piso bajo, nunca cero ni -infinito.

Paso 5: reexporta el .tflite en modo Append, mismo pipeline ya usado y
validado (build_linear_classifier + save_linear_classifier), con
b_ajustado en vez de b.

Robustez: si CUALQUIER paso falla (geocoding, ambas fuentes sin datos,
parseo), el script aborta sin generar nada, dejando el modelo universal
(sin ajuste regional) como esta -- nunca deja el dispositivo peor de lo
que estaba.
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
FLOOR_FRACCION = 0.3     # piso de frecuencia (respecto de la minima real observada) para
                         # especies ausentes del archivo/observaciones, nunca cero
TIMEOUT_RED = 20
USER_AGENT = "LSDTector-BirdNET-retrain-bsas/1.0 (contacto: LSDArroyoGold)"
VENTANA_DIAS_API = 30    # maximo que permite la API de eBird para /recent
MAX_RESULTS_API = 10000


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
            if not nombre or nombre.lower().startswith("sample size") or nombre.lower().startswith("number of taxa"):
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
            # Una fila de especie real tiene 48 frecuencias quincenales.
            # Filas de encabezado sueltas (ej. "Number of taxa: \t645")
            # dan un unico valor y no deben tratarse como especie.
            if not valores or len(valores) < 24:
                continue
            frecuencias[nombre.lower()] = sum(valores) / len(valores)
    if not frecuencias:
        raise RuntimeError(f"No se pudo parsear ninguna especie de {path}, formato inesperado")
    return frecuencias


def calcular_frecuencias_barchart(frec_por_nombre_comun, crosswalk, especies_193):
    """especies_193 son nombres cientificos (nuestras etiquetas). Se
    busca el nombre comun de cada una via el crosswalk, y su frecuencia
    en el archivo de la region. Piso para las que no matchean o dan 0."""
    frecuencias = {}
    no_encontradas = []
    inv_crosswalk = {}
    for comun_lower, sci in crosswalk.items():
        inv_crosswalk.setdefault(sci, comun_lower)

    for esp in especies_193:
        comun = inv_crosswalk.get(esp)
        val = frec_por_nombre_comun.get(comun) if comun else None
        frecuencias[esp] = val if val and val > 0 else None
        if frecuencias[esp] is None:
            no_encontradas.append(esp)

    return aplicar_piso(frecuencias, especies_193), no_encontradas


def aplicar_piso(frecuencias, especies_193):
    positivas = [v for v in frecuencias.values() if v is not None]
    piso = (min(positivas) * FLOOR_FRACCION) if positivas else 0.001
    piso = max(piso, 1e-5)
    for esp in especies_193:
        if frecuencias.get(esp) is None:
            frecuencias[esp] = piso
    return frecuencias


def obtener_observaciones_recientes_ebird(region_code, api_key):
    """Observaciones de especies en la region, ultimos VENTANA_DIAS_API
    dias, via la API publica de eBird (requiere API key de aplicacion).
    Ver advertencia de uso comercial en el docstring del modulo."""
    url = (
        f"https://api.ebird.org/v2/data/obs/{urllib.parse.quote(region_code)}/recent?"
        + urllib.parse.urlencode({"back": VENTANA_DIAS_API, "cat": "species", "maxResults": MAX_RESULTS_API})
    )
    req = urllib.request.Request(url, headers={"X-eBirdApiToken": api_key, "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_RED) as resp:
        return json.load(resp)


def calcular_frecuencias_api(observaciones, especies_193):
    """Cuenta checklists distintos (subId) por especie (sciName) entre las
    observaciones recientes, y arma la frecuencia relativa para las 193
    especies locales. Piso para las que no aparecieron en la ventana."""
    vistos_por_especie = {}
    for obs in observaciones:
        sci = obs.get("sciName")
        sub_id = obs.get("subId")
        if not sci:
            continue
        vistos_por_especie.setdefault(sci, set())
        if sub_id:
            vistos_por_especie[sci].add(sub_id)

    frecuencias = {}
    no_encontradas = []
    for esp in especies_193:
        n = len(vistos_por_especie.get(esp, ()))
        frecuencias[esp] = float(n) if n > 0 else None
        if frecuencias[esp] is None:
            no_encontradas.append(esp)

    return aplicar_piso(frecuencias, especies_193), no_encontradas


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
    ap.add_argument("--ebird-key", default=None,
                     help="API key de eBird (https://ebird.org/api/keygen), solo como respaldo "
                          "si no hay archivo de bar chart para la region. Ver advertencia de uso "
                          "comercial en el docstring del modulo antes de usar esto en produccion.")
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
            log(f"No se pudo descargar frecuencias para {region_code} ({e}).")
            barchart_path = None

    fuente = None
    frecuencias = None
    no_encontradas = None

    if barchart_path and os.path.exists(barchart_path):
        log(f"Usando archivo de frecuencias (bar chart): {barchart_path}")
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

        frecuencias, no_encontradas = calcular_frecuencias_barchart(frec_por_nombre_comun, crosswalk, especies)
        fuente = f"barchart:{barchart_path}"

    elif args.ebird_key:
        log(f"No hay archivo de bar chart para {region_code}. Usando la API de eBird como respaldo "
            f"(observaciones de los ultimos {VENTANA_DIAS_API} dias). Ver advertencia de uso comercial "
            f"en el docstring del modulo.")
        try:
            observaciones = obtener_observaciones_recientes_ebird(region_code, args.ebird_key)
        except Exception as e:
            log(f"ERROR consultando la API de eBird: {e}. Abortando sin generar nada.")
            sys.exit(1)
        log(f"{len(observaciones)} observaciones recibidas de la API de eBird")
        frecuencias, no_encontradas = calcular_frecuencias_api(observaciones, especies)
        fuente = f"api_ebird:{region_code}"

    else:
        log(f"No hay archivo de frecuencias para {region_code}, y no se paso --ebird-key para usar "
            f"la API como respaldo. Nada que ajustar todavia para esta region. "
            f"Abortando sin generar nada (no es un error).")
        sys.exit(2)

    log(f"Fuente de frecuencias: {fuente}")
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
            "fuente": fuente,
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
