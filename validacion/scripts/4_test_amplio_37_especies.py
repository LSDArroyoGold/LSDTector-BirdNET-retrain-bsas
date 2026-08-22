import sys, os, glob, re, json, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import librosa
from tensorflow import lite as tflite
from collections import defaultdict

MODEL = os.path.expanduser('~/Desktop/Tector/LSDTector-BirdNET-retrain-bsas/modelo/LSDTector_Classifier_v2.tflite')
LABELS = os.path.expanduser('~/Desktop/Tector/LSDTector-BirdNET-retrain-bsas/modelo/LSDTector_Classifier_v2_Labels.txt')
DET_DIR = os.path.expanduser('~/Desktop/Tector/BirdNET_Detecciones')

with open(LABELS) as f:
    labels = [l.strip() for l in f]

interp = tflite.Interpreter(MODEL)
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()

def sens_from_conf(s):
    return max(0.5, min(1.0 - (s - 1.0), 1.5))
sens = sens_from_conf(1.1)
UMBRAL_CONFIANZA_ARCHIVO = 50

comun_a_sci = {}
for l in labels:
    if '_' in l:
        sci, com = l.split('_', 1)
        comun_a_sci[com.strip().lower()] = sci.strip()

def indices_para_especie(nombre_comun):
    nombre_comun_l = nombre_comun.lower()
    idx_orig = None
    idx_bare = None
    for i, l in enumerate(labels):
        if '_' in l and l.split('_', 1)[1].strip().lower() == nombre_comun_l:
            idx_orig = i
            break
    sci = comun_a_sci.get(nombre_comun_l)
    if sci:
        try:
            idx_bare = labels.index(sci)
        except ValueError:
            idx_bare = None
    return idx_orig, idx_bare

patron = re.compile(r'^(.+)-(\d+)-\d{4}-\d{2}-\d{2}-birdnet-.*\.mp3$')

archivos_por_especie = defaultdict(list)
for especie_dir in sorted(glob.glob(os.path.join(DET_DIR, '*', '*'))):
    if not os.path.isdir(especie_dir):
        continue
    especie = os.path.basename(especie_dir)
    for f in glob.glob(os.path.join(especie_dir, '*.mp3')):
        m = patron.match(os.path.basename(f))
        if not m:
            continue
        conf_original = int(m.group(2))
        if conf_original >= UMBRAL_CONFIANZA_ARCHIVO:
            archivos_por_especie[especie].append((f, conf_original))

registros = []  # dict por archivo procesado

for especie, archivos in sorted(archivos_por_especie.items()):
    nombre_comun = especie.replace('_', ' ')
    idx_orig, idx_bare = indices_para_especie(nombre_comun)
    if idx_orig is None and idx_bare is None:
        continue
    for f, conf_orig in archivos:
        try:
            sig, sr = librosa.load(f, sr=48000, mono=True, res_type='kaiser_fast')
        except Exception:
            continue
        if len(sig) < 48000 * 1.0:
            continue
        mejor_top1_idx = None
        mejor_top1_score = -1
        mejor_propio = -1
        for start in range(0, len(sig), 48000*3):
            chunk = sig[start:start+48000*3]
            if len(chunk) < 48000 * 1.0:
                continue
            if len(chunk) < 48000*3:
                tmp = np.zeros(48000*3, dtype=np.float32); tmp[:len(chunk)] = chunk; chunk = tmp
            interp.set_tensor(inp[0]['index'], np.array(chunk, dtype='float32')[np.newaxis, :])
            interp.invoke()
            logits = interp.get_tensor(out[0]['index'])[0]
            scores = 1/(1+np.exp(-sens*logits))
            propio = 0.0
            if idx_bare is not None:
                propio = max(propio, scores[idx_bare])
            if idx_orig is not None:
                propio = max(propio, scores[idx_orig])
            if propio > mejor_propio:
                mejor_propio = propio
            top1_i = int(np.argmax(scores))
            if scores[top1_i] > mejor_top1_score:
                mejor_top1_score = scores[top1_i]
                mejor_top1_idx = top1_i
        if mejor_top1_idx is None:
            continue
        es_propio_top1 = (mejor_top1_idx == idx_orig) or (mejor_top1_idx == idx_bare)
        registros.append({
            'archivo': f,
            'especie_original': especie,
            'conf_original': conf_orig,
            'propio_score': round(float(mejor_propio), 4),
            'top1_correcto': es_propio_top1,
            'top1_label': labels[mejor_top1_idx],
            'top1_score': round(float(mejor_top1_score), 4),
        })

out_csv = '/tmp/claude-1000/-home-diego/3658a387-500c-480b-a242-e48d9d832d75/scratchpad/registros_broad_50.csv'
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(registros[0].keys()))
    w.writeheader()
    w.writerows(registros)
print(f'{len(registros)} registros guardados en {out_csv}')
