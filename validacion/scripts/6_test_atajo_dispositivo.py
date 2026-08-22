import os, glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import librosa
from tensorflow import lite as tflite

MODEL = os.path.expanduser('~/Desktop/Tector/LSDTector-BirdNET-retrain-bsas/modelo/LSDTector_Classifier_v2.tflite')
LABELS = os.path.expanduser('~/Desktop/Tector/LSDTector-BirdNET-retrain-bsas/modelo/LSDTector_Classifier_v2_Labels.txt')

with open(LABELS) as f:
    labels = [l.strip() for l in f]

interp = tflite.Interpreter(MODEL)
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()

def sens_from_conf(s):
    return max(0.5, min(1.0 - (s - 1.0), 1.5))
sens = sens_from_conf(1.1)

LADRONES = ['Zonotrichia capensis', 'Guira guira', 'Agelaioides badius']
idxs = {l: labels.index(l) for l in LADRONES}

def max_scores(archivos, idxs_dict):
    resultados = {l: [] for l in idxs_dict}
    for f in archivos:
        try:
            sig, sr = librosa.load(f, sr=48000, mono=True, res_type='kaiser_fast')
        except Exception:
            continue
        if len(sig) < 48000 * 1.0:
            continue
        mejores = {l: 0.0 for l in idxs_dict}
        for start in range(0, len(sig), 48000*3):
            chunk = sig[start:start+48000*3]
            if len(chunk) < 48000 * 1.0:
                continue
            if len(chunk) < 48000*3:
                tmp = np.zeros(48000*3, dtype=np.float32); tmp[:len(chunk)] = chunk; chunk = tmp
            interp.set_tensor(inp[0]['index'], np.array(chunk, dtype='float32')[np.newaxis, :])
            interp.invoke()
            logits = interp.get_tensor(out[0]['index'])[0]
            for l, idx in idxs_dict.items():
                s = 1/(1+np.exp(-sens*logits[idx]))
                mejores[l] = max(mejores[l], s)
        for l in idxs_dict:
            resultados[l].append(mejores[l])
    return {l: np.array(v) for l, v in resultados.items()}

xc_hornero = sorted(glob.glob(os.path.expanduser('~/Desktop/Tector/Datasets_prueba/xc_hornero_ar/*.mp3')))
xc_kestrel = sorted(glob.glob(os.path.expanduser('~/Desktop/Tector/Datasets_prueba/xc_kestrel_ar/*.mp3')))
externos = xc_hornero + xc_kestrel

print(f'Audio EXTERNO (Xeno-canto, NO tector1): {len(externos)} archivos')
r_ext = max_scores(externos, idxs)
for l in LADRONES:
    v = r_ext[l]
    print(f'  {l}: media={v.mean():.3f} mediana={np.median(v):.3f} max={v.max():.3f}  (>=0.6: {(v>=0.6).mean()*100:.1f}%)  (>=0.9: {(v>=0.9).mean()*100:.1f}%)')

print()
print('(comparar con lo ya medido en audio de tector1 ajeno a estas 3 especies: mediana 0.19-0.51, con colas hasta ~0.999)')
