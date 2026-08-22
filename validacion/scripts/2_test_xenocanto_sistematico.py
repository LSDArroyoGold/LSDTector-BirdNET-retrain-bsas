import sys, os, glob
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
UMBRAL = 0.6

IDX_H = labels.index('Furnarius rufus')
IDX_K = labels.index('Falco sparverius')

def procesar_carpeta(carpeta, idx_propio, idx_otro, nombre_propio, nombre_otro):
    archivos = sorted(glob.glob(os.path.join(carpeta, '*.mp3')))
    total_seg = 0
    propio_cruza = 0
    otro_cruza = 0
    propio_gana = 0
    archivos_con_deteccion = 0
    archivos_procesados = 0
    for a in archivos:
        try:
            sig, sr = librosa.load(a, sr=48000, mono=True, res_type='kaiser_fast')
        except Exception:
            continue
        if len(sig) < 48000 * 1.0:
            continue
        archivos_procesados += 1
        detecto_algo = False
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
            p, o = scores[idx_propio], scores[idx_otro]
            total_seg += 1
            if p >= UMBRAL:
                propio_cruza += 1
                detecto_algo = True
            if o >= UMBRAL:
                otro_cruza += 1
            if p > o:
                propio_gana += 1
        if detecto_algo:
            archivos_con_deteccion += 1
    print(f'  Archivos procesados: {archivos_procesados}, segmentos totales: {total_seg}')
    print(f'  Archivos con AL MENOS 1 segmento cruzando umbral ({nombre_propio}): {archivos_con_deteccion}/{archivos_procesados} ({100*archivos_con_deteccion/max(1,archivos_procesados):.1f}%)')
    print(f'  Segmentos {nombre_propio} cruza umbral: {propio_cruza}/{total_seg} ({100*propio_cruza/max(1,total_seg):.1f}%)')
    print(f'  Segmentos {nombre_otro} cruza umbral (falso positivo): {otro_cruza}/{total_seg} ({100*otro_cruza/max(1,total_seg):.1f}%)')
    print(f'  Segmentos donde {nombre_propio} > {nombre_otro}: {propio_gana}/{total_seg} ({100*propio_gana/max(1,total_seg):.1f}%)')

print('===== HORNERO: 63 grabaciones reales de Argentina, Xeno-canto calidad A =====')
procesar_carpeta(os.path.expanduser('~/Desktop/Tector/Datasets_prueba/xc_hornero_ar'), IDX_H, IDX_K, 'Hornero', 'Kestrel')

print()
print('===== KESTREL: 18 grabaciones reales de Argentina, Xeno-canto calidad A =====')
procesar_carpeta(os.path.expanduser('~/Desktop/Tector/Datasets_prueba/xc_kestrel_ar'), IDX_K, IDX_H, 'Kestrel', 'Hornero')
