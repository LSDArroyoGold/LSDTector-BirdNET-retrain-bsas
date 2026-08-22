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

IDX_H = labels.index('Furnarius rufus')
IDX_K = labels.index('Falco sparverius')
UMBRAL = 0.6

# Precomputamos TODOS los logits crudos de cada archivo, una sola vez,
# para poder probar varios SENSITIVITY sin re-correr el modelo cada vez.
def cargar_logits(carpeta_o_lista):
    if isinstance(carpeta_o_lista, str):
        archivos = sorted(glob.glob(os.path.join(carpeta_o_lista, '*.mp3')))
    else:
        archivos = carpeta_o_lista
    resultados = []  # lista de (archivo, [logits_por_chunk])
    for a in archivos:
        try:
            sig, sr = librosa.load(a, sr=48000, mono=True, res_type='kaiser_fast')
        except Exception:
            continue
        if len(sig) < 48000 * 1.0:
            continue
        chunks_logits = []
        for start in range(0, len(sig), 48000*3):
            chunk = sig[start:start+48000*3]
            if len(chunk) < 48000 * 1.0:
                continue
            if len(chunk) < 48000*3:
                tmp = np.zeros(48000*3, dtype=np.float32); tmp[:len(chunk)] = chunk; chunk = tmp
            interp.set_tensor(inp[0]['index'], np.array(chunk, dtype='float32')[np.newaxis, :])
            interp.invoke()
            logits = interp.get_tensor(out[0]['index'])[0].copy()
            chunks_logits.append(logits)
        if chunks_logits:
            resultados.append((a, chunks_logits))
    return resultados

print('Cargando logits crudos de todos los sets (una sola vez)...')
sets = {
    'Campo: Rufous_Hornero (76)': (cargar_logits(os.path.expanduser('~/Desktop/Tector/Datasets_prueba/Audios confusion/Rufous_Hornero')), IDX_H, IDX_K, 'Hornero', 'Kestrel'),
    'Campo: American_Kestrel=Hornero real (179)': (cargar_logits(os.path.expanduser('~/Desktop/Tector/Datasets_prueba/Audios confusion/American_Kestrel')), IDX_H, IDX_K, 'Hornero', 'Kestrel'),
    'Xeno-canto AR: Hornero (63)': (cargar_logits(os.path.expanduser('~/Desktop/Tector/Datasets_prueba/xc_hornero_ar')), IDX_H, IDX_K, 'Hornero', 'Kestrel'),
    'Xeno-canto AR: Kestrel (18)': (cargar_logits(os.path.expanduser('~/Desktop/Tector/Datasets_prueba/xc_kestrel_ar')), IDX_K, IDX_H, 'Kestrel', 'Hornero'),
}
print('Listo.\n')

for sens_conf in [0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4]:
    sens = sens_from_conf(sens_conf)
    print(f'===== SENSITIVITY={sens_conf} (escala sigmoide={sens:.3f}) =====')
    for nombre, (resultados, idx_propio, idx_otro, np_, no_) in sets.items():
        total_archivos = len(resultados)
        n_con_deteccion = 0
        n_falso_positivo_otro = 0
        n_segmentos_total = 0
        n_segmentos_propio_cruza = 0
        n_over_trigger = 0  # archivos con >=5 especies distintas cruzando umbral en algun chunk (alarma de sobre-disparo)
        for archivo, chunks in resultados:
            detecto = False
            for logits in chunks:
                scores = 1/(1+np.exp(-sens*logits))
                n_segmentos_total += 1
                if scores[idx_propio] >= UMBRAL:
                    n_segmentos_propio_cruza += 1
                    detecto = True
                if scores[idx_otro] >= UMBRAL:
                    n_falso_positivo_otro += 1
                n_cruzan = int((scores >= UMBRAL).sum())
                if n_cruzan >= 5:
                    n_over_trigger += 1
            if detecto:
                n_con_deteccion += 1
        print(f'  {nombre}: archivos con deteccion {n_con_deteccion}/{total_archivos} ({100*n_con_deteccion/max(1,total_archivos):.1f}%)  '
              f'segmentos {np_} cruza={n_segmentos_propio_cruza}/{n_segmentos_total} ({100*n_segmentos_propio_cruza/max(1,n_segmentos_total):.1f}%)  '
              f'falsos-{no_}={n_falso_positivo_otro}  sobre-disparo(>=5 sp)={n_over_trigger}')
    print()
