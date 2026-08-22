#!/usr/bin/env python3
import os, glob, subprocess, csv, sys, tempfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import soundfile as sf
import librosa

CARPETA = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(CARPETA, 'resultados_merlin.csv')

archivos = sorted(glob.glob(os.path.join(CARPETA, '*.mp3')))

ya_hechos = {}
if os.path.exists(SALIDA):
    with open(SALIDA) as f:
        for r in csv.DictReader(f):
            ya_hechos[r['archivo']] = r['especie_merlin']

def reproducir(mp3_path):
    sig, sr = librosa.load(mp3_path, sr=None, mono=True)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, sig, sr)
        subprocess.run(['paplay', tmp.name])
    os.unlink(tmp.name)

filas = []
pendientes = [a for a in archivos if os.path.basename(a) not in ya_hechos]
print(f'{len(archivos)} archivos totales, {len(pendientes)} pendientes.\n')

for i, path in enumerate(pendientes, 1):
    nombre = os.path.basename(path)
    partes = nombre.replace('.mp3', '').split('__vs__')
    victima = partes[0]
    resto = partes[1].split('__confBNviejo')
    ladron, conf = resto[0], resto[1]
    print(f'\n[{i}/{len(pendientes)}] BirdNET viejo dijo: {victima} ({conf}%)  |  v10 dice: {ladron}')
    while True:
        reproducir(path)
        resp = input("  Especie segun Merlin (o 'r' repetir, 'skip' saltar): ").strip()
        if resp.lower() == 'r':
            continue
        if resp.lower() == 'skip':
            resp = ''
        break
    filas.append({'archivo': nombre, 'especie_original_birdnet_viejo': victima,
                   'conf_original': conf, 'especie_v10': ladron, 'especie_merlin': resp})

    # Guardar progresivamente por si se corta a la mitad
    escribir_header = not os.path.exists(SALIDA)
    with open(SALIDA, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['archivo', 'especie_original_birdnet_viejo', 'conf_original', 'especie_v10', 'especie_merlin'])
        if escribir_header:
            w.writeheader()
        w.writerow(filas[-1])

print(f'\nListo. Resultados en {SALIDA}')
