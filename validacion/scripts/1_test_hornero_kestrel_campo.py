import sys, os, glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import librosa
from tensorflow import lite as tflite

MODEL = os.path.expanduser('~/Desktop/Tector/LSDTector-BirdNET-retrain-bsas/modelo/LSDTector_Classifier_v2.tflite')
LABELS = os.path.expanduser('~/Desktop/Tector/LSDTector-BirdNET-retrain-bsas/modelo/LSDTector_Classifier_v2_Labels.txt')

with open(LABELS) as f:
    labels = [l.strip() for l in f]
print('total labels:', len(labels))

interp = tflite.Interpreter(MODEL)
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()

# SENSITIVITY de produccion actual en tector2: 1.1 -> escala 0.9. Probamos
# tambien con 1.25 (default clasico de BirdNET-Pi) para comparar.
def sens_from_conf(s):
    return max(0.5, min(1.0 - (s - 1.0), 1.5))

IDX_H_BARE = labels.index('Furnarius rufus')
IDX_K_BARE = labels.index('Falco sparverius')
IDX_H_ORIG = labels.index('Furnarius rufus_Rufous Hornero')
IDX_K_ORIG = labels.index('Falco sparverius_American Kestrel')

def analizar(path, sens):
    try:
        sig, sr = librosa.load(path, sr=48000, mono=True, res_type='kaiser_fast')
    except Exception as e:
        return None
    if len(sig) < 48000 * 1.0:
        return None
    # Tomamos el mejor (max) segmento de 3s del clip, como hace BirdNET
    n_chunks = max(1, len(sig) // (48000*3))
    mejor_h, mejor_k = 0.0, 0.0
    mejor_h_orig, mejor_k_orig = 0.0, 0.0
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
        mejor_h = max(mejor_h, scores[IDX_H_BARE])
        mejor_k = max(mejor_k, scores[IDX_K_BARE])
        mejor_h_orig = max(mejor_h_orig, scores[IDX_H_ORIG])
        mejor_k_orig = max(mejor_k_orig, scores[IDX_K_ORIG])
    return mejor_h, mejor_k, mejor_h_orig, mejor_k_orig

CARPETAS = {
    'Rufous_Hornero (etiquetado Hornero)': os.path.expanduser('~/Desktop/Tector/Audios confusion/Rufous_Hornero'),
    'American_Kestrel (etiquetado Kestrel, en realidad Hornero segun el usuario)': os.path.expanduser('~/Desktop/Tector/Audios confusion/American_Kestrel'),
}

UMBRAL = 0.7

for sens_conf, nombre_sens in [(1.1, 'SENSITIVITY=1.1 (produccion actual)'), (1.25, 'SENSITIVITY=1.25 (default BirdNET-Pi)')]:
    sens = sens_from_conf(sens_conf)
    print(f'\n========== {nombre_sens} (escala sigmoide={sens:.3f}) ==========')
    for nombre, carpeta in CARPETAS.items():
        archivos = sorted(glob.glob(os.path.join(carpeta, '*.mp3')))
        n = len(archivos)
        n_reentrenada_cruza = 0
        n_original_cruza = 0
        n_reentrenada_hornero_gana = 0
        n_procesados = 0
        hs, ks = [], []
        for a in archivos:
            r = analizar(a, sens)
            if r is None:
                continue
            h, k, h_orig, k_orig = r
            n_procesados += 1
            hs.append(h); ks.append(k)
            if h >= UMBRAL:
                n_reentrenada_cruza += 1
            if h_orig >= UMBRAL:
                n_original_cruza += 1
            if h > k:
                n_reentrenada_hornero_gana += 1
        if n_procesados == 0:
            print(f'  {nombre}: 0 archivos procesados')
            continue
        print(f'  {nombre}: {n_procesados} archivos')
        print(f'    Neurona reentrenada Hornero cruza {UMBRAL}: {n_reentrenada_cruza}/{n_procesados} ({100*n_reentrenada_cruza/n_procesados:.1f}%)')
        print(f'    Neurona original    Hornero cruza {UMBRAL}: {n_original_cruza}/{n_procesados} ({100*n_original_cruza/n_procesados:.1f}%)')
        print(f'    Hornero > Kestrel (reentrenada): {n_reentrenada_hornero_gana}/{n_procesados} ({100*n_reentrenada_hornero_gana/n_procesados:.1f}%)')
        print(f'    Hornero score: media={np.mean(hs):.3f} mediana={np.median(hs):.3f}')
        print(f'    Kestrel score: media={np.mean(ks):.3f} mediana={np.median(ks):.3f}')

print('\n\n========== KESTREL REAL (Xeno-canto, ground truth confirmado) ==========')
sens = sens_from_conf(1.1)
kestrel_files = sorted(glob.glob(os.path.expanduser('~/Desktop/Tector/xenocanto_test/*.mp3')))
for a in kestrel_files:
    sig, sr = librosa.load(a, sr=48000, mono=True, res_type='kaiser_fast')
    print(f'--- {os.path.basename(a)} ({len(sig)/48000:.1f}s) ---')
    n_h_cruza = n_k_cruza = n_chunks = 0
    for start in range(0, len(sig), 48000*3):
        chunk = sig[start:start+48000*3]
        if len(chunk) < 48000 * 1.5:
            continue
        if len(chunk) < 48000*3:
            tmp = np.zeros(48000*3, dtype=np.float32); tmp[:len(chunk)] = chunk; chunk = tmp
        interp.set_tensor(inp[0]['index'], np.array(chunk, dtype='float32')[np.newaxis, :])
        interp.invoke()
        logits = interp.get_tensor(out[0]['index'])[0]
        scores = 1/(1+np.exp(-sens*logits))
        h, k = scores[IDX_H_BARE], scores[IDX_K_BARE]
        n_chunks += 1
        if h >= UMBRAL: n_h_cruza += 1
        if k >= UMBRAL: n_k_cruza += 1
        gana = 'KESTREL' if k > h else 'hornero (MAL)'
        print(f'  {start/48000:.0f}-{start/48000+3:.0f}s: kestrel={k:.4f} hornero={h:.4f} ({gana})')
    print(f'  Resumen: {n_chunks} segmentos, Kestrel cruza umbral {n_k_cruza}/{n_chunks}, Hornero cruza umbral (falso positivo) {n_h_cruza}/{n_chunks}')

print('\n\n========== HORNERO INDEPENDIENTE (Xeno-canto/Wikimedia, NO en el set de entrenamiento) ==========')
hornero_files = sorted(glob.glob(os.path.expanduser('~/Desktop/Tector/xenocanto_test/hornero_*')))
for a in hornero_files:
    sig, sr = librosa.load(a, sr=48000, mono=True, res_type='kaiser_fast')
    print(f'--- {os.path.basename(a)} ({len(sig)/48000:.1f}s) ---')
    n_h_cruza = n_k_cruza = n_chunks = 0
    for start in range(0, len(sig), 48000*3):
        chunk = sig[start:start+48000*3]
        if len(chunk) < 48000 * 1.5:
            continue
        if len(chunk) < 48000*3:
            tmp = np.zeros(48000*3, dtype=np.float32); tmp[:len(chunk)] = chunk; chunk = tmp
        interp.set_tensor(inp[0]['index'], np.array(chunk, dtype='float32')[np.newaxis, :])
        interp.invoke()
        logits = interp.get_tensor(out[0]['index'])[0]
        scores = 1/(1+np.exp(-sens*logits))
        h, k = scores[IDX_H_BARE], scores[IDX_K_BARE]
        n_chunks += 1
        if h >= UMBRAL: n_h_cruza += 1
        if k >= UMBRAL: n_k_cruza += 1
        gana = 'HORNERO' if h > k else 'kestrel (MAL)'
        print(f'  {start/48000:.0f}-{start/48000+3:.0f}s: hornero={h:.4f} kestrel={k:.4f} ({gana})')
    print(f'  Resumen: {n_chunks} segmentos, Hornero cruza umbral {n_h_cruza}/{n_chunks}, Kestrel cruza umbral (falso positivo) {n_k_cruza}/{n_chunks}')
