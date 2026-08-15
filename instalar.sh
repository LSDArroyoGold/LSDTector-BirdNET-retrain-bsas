#!/bin/bash
# ============================================================
# Instalacion del clasificador BirdNET reentrenado (LSD-Tector v2)
# en una Raspberry Pi con Raspberry Pi OS (Bookworm o posterior).
#
# No toca ningun otro software que ya corra en el dispositivo: crea
# un entorno virtual de Python propio y aislado (~/birdnet-v2-env).
# ============================================================
set -e

echo "=== Paso 1/3: dependencias del sistema ==="
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libsndfile1 ffmpeg

echo "=== Paso 2/3: entorno virtual dedicado (~/birdnet-v2-env) ==="
python3 -m venv ~/birdnet-v2-env
source ~/birdnet-v2-env/bin/activate
pip install --upgrade pip

echo "=== Paso 3/3: birdnet-analyzer (misma version usada para entrenar y validar el modelo) ==="
pip install birdnet-analyzer==2.4.0

echo ""
echo "=== Instalacion completa ==="
echo "Para probar que el modelo reentrenado funciona:"
echo ""
echo "    source ~/birdnet-v2-env/bin/activate"
echo "    python3 probar_modelo.py ruta/a/un/audio.wav"
echo ""
echo "Nota: si 'pip install' falla en encontrar un wheel de tensorflow"
echo "para esta arquitectura, Raspberry Pi OS suele traer configurado"
echo "piwheels.org como indice adicional de paquetes precompilados para"
echo "ARM; si no esta activo, ver https://www.piwheels.org/ para sumarlo."
