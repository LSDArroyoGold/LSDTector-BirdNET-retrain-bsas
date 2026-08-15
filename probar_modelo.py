# -*- coding: utf-8 -*-
"""
Prueba rapida: corre el clasificador LSD-Tector v2 sobre uno o mas
archivos de audio, para confirmar que el modelo reentrenado funciona
correctamente en la Raspberry Pi real (no solo en la maquina donde se
entreno).

Uso:
    python3 probar_modelo.py ruta/a/audio.wav
    python3 probar_modelo.py ruta/a/carpeta_con_wavs/
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(CARPETA_SCRIPT, "modelo", "LSDTector_Classifier_v2.tflite")
RESULTADOS_DIR = os.path.join(CARPETA_SCRIPT, "resultados_prueba")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 probar_modelo.py <archivo_o_carpeta_de_audio>")
        sys.exit(1)

    audio_input = sys.argv[1]

    if not os.path.exists(CLASSIFIER_PATH):
        print(f"No se encontro el modelo en: {CLASSIFIER_PATH}")
        sys.exit(1)

    from birdnet_analyzer import analyze

    print(f"Analizando '{audio_input}' con el clasificador LSD-Tector v2...")
    analyze(
        audio_input=audio_input,
        output=RESULTADOS_DIR,
        classifier=CLASSIFIER_PATH,
        min_conf=0.1,
        rtype="csv",
    )
    print(f"\nListo. Resultados en: {RESULTADOS_DIR}")
    print("Revisar los CSV generados: cada fila es una deteccion, con la")
    print("especie (nombre cientifico), la confianza, y el rango horario")
    print("dentro del audio.")
