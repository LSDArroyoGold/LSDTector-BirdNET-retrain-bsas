import os, glob, re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import librosa
from tensorflow import lite as tflite

MODEL = os.path.expanduser('~/Desktop/Tector/LSDTector-BirdNET-retrain-bsas/modelo/LSDTector_Classifier_v2.tflite')
LABELS = os.path.expanduser('~/Desktop/Tector/LSDTector-BirdNET-retrain-bsas/modelo/LSDTector_Classifier_v2_Labels.txt')
DET_DIR = os.path.expanduser('~/Desktop/Tector/Datasets_prueba/BirdNET_Detecciones')

with open(LABELS) as f:
    labels = [l.strip() for l in f]

interp = tflite.Interpreter(MODEL)
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()

def sens_from_conf(s):
    return max(0.5, min(1.0 - (s - 1.0), 1.5))
sens = sens_from_conf(1.1)  # produccion actual, sin tocar

patron = re.compile(r'^(.+)-(\d+)-\d{4}-\d{2}-\d{2}-birdnet-.*\.mp3$')

def archivos_de(carpeta_especie, umbral_conf_original=50):
    resultado = []
    for f in glob.glob(os.path.join(DET_DIR, '*', carpeta_especie, '*.mp3')):
        m = patron.match(os.path.basename(f))
        if m and int(m.group(2)) >= umbral_conf_original:
            resultado.append(f)
    return resultado

def max_score_por_clip(archivos, idx_clase):
    scores = []
    for f in archivos:
        try:
            sig, sr = librosa.load(f, sr=48000, mono=True, res_type='kaiser_fast')
        except Exception:
            continue
        if len(sig) < 48000 * 1.0:
            continue
        mejor = 0.0
        for start in range(0, len(sig), 48000*3):
            chunk = sig[start:start+48000*3]
            if len(chunk) < 48000 * 1.0:
                continue
            if len(chunk) < 48000*3:
                tmp = np.zeros(48000*3, dtype=np.float32); tmp[:len(chunk)] = chunk; chunk = tmp
            interp.set_tensor(inp[0]['index'], np.array(chunk, dtype='float32')[np.newaxis, :])
            interp.invoke()
            logits = interp.get_tensor(out[0]['index'])[0]
            s = 1/(1+np.exp(-sens*logits[idx_clase]))
            mejor = max(mejor, s)
        scores.append(mejor)
    return np.array(scores)

# especie ladrona -> (nombre cientifico/bare label, carpeta propia de datos reales, carpetas victimas)
LADRONES = {
    'Zonotrichia capensis': {
        'carpeta_propia': 'Rufous-collared_Sparrow',
        'victimas': ['Great_Kiskadee', 'Southern_Lapwing', 'Gray-cowled_Wood-Rail', 'Saffron_Finch', 'Solitary_Sandpiper'],
    },
    'Guira guira': {
        'carpeta_propia': 'Guira_Cuckoo',
        'victimas': ['Great_Kiskadee', 'Southern_Lapwing', 'Buff-browed_Foliage-gleaner', 'Creamy-bellied_Thrush', 'Yellow-chinned_Spinetail'],
    },
    'Agelaioides badius': {
        'carpeta_propia': 'Grayish_Baywing',
        'victimas': ['Great_Kiskadee', 'Southern_Lapwing', 'Picazuro_Pigeon'],
    },
}

for especie, cfg in LADRONES.items():
    idx = labels.index(especie)
    propios = archivos_de(cfg['carpeta_propia'])
    victimas_archivos = []
    for v in cfg['victimas']:
        victimas_archivos += archivos_de(v)

    scores_propios = max_score_por_clip(propios, idx)
    scores_victimas = max_score_por_clip(victimas_archivos, idx)

    print(f'\n===== {especie} (carpeta propia: {cfg["carpeta_propia"]}, n={len(scores_propios)}) =====')
    print(f'  Audio de victimas combinado (n={len(scores_victimas)}): {cfg["victimas"]}')
    print(f'  Score propio: media={scores_propios.mean():.3f} mediana={np.median(scores_propios):.3f} min={scores_propios.min():.3f}')
    print(f'  Score en victimas: media={scores_victimas.mean():.3f} mediana={np.median(scores_victimas):.3f} max={scores_victimas.max():.3f}')
    print(f'  {"Umbral":>8s} {"Recall propio":>14s} {"FalsosPos en victimas":>22s}')
    for umbral in [0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]:
        recall = (scores_propios >= umbral).mean() * 100
        fp = (scores_victimas >= umbral).mean() * 100
        print(f'  {umbral:8.3f} {recall:13.1f}% {fp:21.1f}%')
