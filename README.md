# LSDTector — clasificador BirdNET v2

Clasificador de BirdNET reentrenado para las 193 especies de aves de la
región del campo de prueba (Buenos Aires, código eBird AR-B), listo para
instalar en la Raspberry Pi del LSD-Tector 2.0.

El modelo (`modelo/LSDTector_Classifier_v2.tflite`) reemplaza por completo
la capa de decisión final de BirdNET (modo *Replace*): sólo reconoce estas
193 especies locales, con mejor accuracy top-1 que el modelo global de
BirdNET para esta región (63.6% → 85.7% global, 62.6% → 86.0% macro sobre
el conjunto de validación). El detalle metodológico completo —por qué se
reentrenó, qué falló en el primer intento, y por qué se terminó
entrenando con regresión logística en vez del pipeline propio de
BirdNET-Analyzer— está documentado en el informe del proyecto.

El archivo `.tflite` es autocontenido: incluye tanto el extractor de
características original de BirdNET (sin modificar) como la capa de
decisión nueva, en un único grafo. No hace falta ningún otro archivo del
BirdNET original.

## Instalación

En la Raspberry Pi, con este repositorio ya clonado:

```bash
git clone https://github.com/<ORG>/LSDTector-BirdNET-v2.git
cd LSDTector-BirdNET-v2
bash instalar.sh
```

Esto instala un entorno virtual de Python dedicado (`~/birdnet-v2-env`)
con `birdnet-analyzer` en la misma versión usada para entrenar y validar
el modelo (2.4.0), sin tocar ningún otro software que ya corra en el
dispositivo.

## Probar que funciona

```bash
source ~/birdnet-v2-env/bin/activate
python3 probar_modelo.py ruta/a/un/audio.wav
```

También acepta una carpeta entera de archivos de audio. Corre el modelo
sobre el audio indicado y guarda los resultados en `resultados_prueba/`
(un CSV por archivo, con especie, confianza y rango horario de cada
detección). Sirve para confirmar que el modelo reentrenado corre
correctamente en el hardware real antes de integrarlo a cualquier
pipeline más grande.

## Actualizar más adelante

Si se sube una versión nueva del modelo a este repositorio, en la Pi
alcanza con:

```bash
cd LSDTector-BirdNET-v2
git pull
```

(no hace falta repetir `instalar.sh`, salvo que cambie la versión de
`birdnet-analyzer` fijada ahí).

## Qué NO incluye este repositorio (todavía)

Este repo resuelve únicamente "correr el modelo reentrenado
correctamente" en la Raspberry Pi. Deliberadamente no incluye la
integración con el resto del pipeline del LSD-Tector (grabación
automática programada, base de datos, envío a BirdWeather, manejo de
energía, watchdog de batería, etc.) — eso se decide en una etapa
posterior, una vez confirmado que el modelo funciona bien sobre el
dispositivo real.

En particular: **BirdNET-Pi (Nachtzuster/BirdNET-Pi), tal como está hoy,
no soporta clasificadores personalizados** — su selector de modelo tiene
una lista fija de 4 modelos posibles, sin forma de apuntar a un `.tflite`
propio. Cuando llegue el momento de integrar esto con la interfaz web,
la base de datos y BirdWeather, hay que decidir entre parchear
`scripts/utils/models.py` de BirdNET-Pi para reconocer este modelo como
una quinta opción, o armar un pipeline propio más liviano. Ese es un
problema aparte del que resuelve este repositorio.
