# LSDTector: clasificador BirdNET v2

> ✅ **Colapso de sensibilidad en audio real de campo, corregido y
> validado (22/8).** El problema no era solo Hornero/Kestrel: al
> extender el chequeo a más especies con volumen suficiente de
> detecciones reales, el modelo reentrenado perdía sensibilidad fuerte
> en varias de ellas (Hornero, Kiskadee, Chingolo y otras), aunque el
> catálogo original de BirdNET las detectaba perfecto en esos mismos
> clips. Se investigó a fondo (banda, ruido, reverb, compresión,
> distorsión, pipeline, sobreajuste, todo probado y descartado como
> causa única) hasta encontrar que el audio real de campo corre el
> embedding interno del modelo en una dirección muy específica que
> ninguna degradación sintética de Xeno-canto reproduce. Se reentrenaron
> 14 neuronas puntuales con audio real de campo (nunca con Xeno-canto
> degradado): 8 que habían colapsado del todo, más 5 que estaban
> "bien" pero limítrofes, más Kestrel con refuerzo de negativos duros
> reales. Validado contra un holdout real nunca visto en entrenamiento:
> las 13 especies con positivos nuevos mejoraron, ninguna empeoró, y
> las otras 179 especies quedaron byte-idénticas, sin tocar. Ver
> [`INFORME_FIX_AUDIO_REAL.md`](INFORME_FIX_AUDIO_REAL.md) para el
> detalle completo.
>
> ✅ **Falla de calibración corregida y validada (21/8).** La versión
> anterior tenía un desajuste entre cómo se entrenaba (por ranking
> relativo entre las 193 clases) y cómo se usa en producción (umbral
> absoluto e independiente por clase), que causaba decenas de detecciones
> simultáneas falsas. Corregido reentrenando cada una de las 193 clases
> como un clasificador binario independiente, con negativos reales de
> ~1000 especies ajenas y sonido no-ave (ver
> [`INFORME_FALLAS_V2.md`](INFORME_FALLAS_V2.md) para el detalle completo,
> incluida la sección de resolución al principio).
>
> ⚠️ **Corregido (21/8, más tarde): la "confusión Hornero/Kestrel"
> reportada antes medía la neurona equivocada** (la del catálogo global
> original, sin tocar por este proyecto: el índice real de las 193
> especies reentrenadas no lleva `_NombreComún`). Con la neurona
> correcta no hay evidencia de confusión fuerte entre estas dos especies;
> lo que sí parece real es baja sensibilidad de la neurona reentrenada de
> Hornero en ciertas grabaciones de campo puntuales. Ver la sección
> "CORRECCIÓN" al principio de
> [`INFORME_CONFUSION_HORNERO_KESTREL.md`](INFORME_CONFUSION_HORNERO_KESTREL.md).

Clasificador de BirdNET reentrenado para las 193 especies de aves de la
región del campo de prueba (Buenos Aires, código eBird AR-B), listo para
instalar en la Raspberry Pi del LSD-Tector 2.0.

El modelo (`modelo/LSDTector_Classifier_v2.tflite`) agrega al catálogo
global de BirdNET (modo *Append*) una capa de decisión nueva especializada
en estas 193 especies locales. Evaluado con la metodología real de
producción (umbral fijo de 0.7 aplicado de forma independiente a cada una
de las 6715 clases, agrupado por segmento de 3 segundos, no top-1/argmax):
87.3% de accuracy top-1 y 85.4% de recall por segmento sobre las 193
especies locales (contra 63.0%/71.3% del modelo original sin reentrenar),
con una tasa de detecciones falsas simultáneas ("especies espurias por
segmento") igual o mejor que la del modelo original: 44.9% de segmentos
con alguna detección espuria contra 48.3% del original. El detalle
metodológico completo está en el informe del proyecto y en
`INFORME_FALLAS_V2.md`.

El archivo `.tflite` es autocontenido: incluye el extractor de
características original de BirdNET (sin modificar), la capa de decisión
original (sin modificar), y la capa de decisión nueva, todo en un único
grafo con 6715 salidas en total. No hace falta ningún otro archivo del
BirdNET original.

## Instalación

En la Raspberry Pi, con este repositorio ya clonado:

```bash
git clone https://github.com/LSDArroyoGold/LSDTector-BirdNET-retrain-bsas.git
cd LSDTector-BirdNET-retrain-bsas
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
detección).

Esto sirve para una primera verificación rápida de que el modelo corre en
el hardware real, pero **no reemplaza la prueba real necesaria antes de
instalar en campo**: hay que correrlo con el umbral fijo de producción
(0.7, no el `min_conf=0.1` que usa este script por defecto) sobre clips de
sonido ambiente típico del sitio de despliegue, no solo cantos limpios de
una especie, y contar cuántas detecciones da por clip. Esta prueba con
`min_conf` bajo fue justamente la que no mostró la falla de calibración de
la v2 con la claridad suficiente (ver el punto 4 de "Qué se necesita para
reentrenar bien" en `INFORME_FALLAS_V2.md`).

## Cómo se actualiza el modelo en la Raspberry Pi

Este repositorio no se clona ni se actualiza directamente en el
dispositivo de campo con `git pull`. El mecanismo real es
`scripts/actualizar_modelo.sh`, en el repositorio `LSD-Tector2.0`: compara
el SHA del último commit de este repo (vía la API de GitHub) contra el
último aplicado, y si cambió, descarga `modelo/LSDTector_Classifier_v2.tflite`
y su archivo de labels directamente por `raw.githubusercontent.com`, para
pisar los archivos que BirdNET-Pi ya tiene en su propia carpeta de
modelo. Corre solo, en cada ventana de grabación, sin intervención
manual. Esto es justamente lo que motivó sacar el `.tflite` roto de acá:
mientras estuviera presente, este mecanismo lo iba a reinstalar solo en
cualquier dispositivo que actualizara, sin que nadie lo pidiera.

`instalar.sh` y `probar_modelo.py`, en cambio, son para correr el modelo
de forma aislada y manual, sin depender de BirdNET-Pi ni del resto del
pipeline del LSD-Tector, y siguen sirviendo para eso.

## Ajuste regional por frecuencia: se mudó de repo

El ajuste de sesgo por frecuencia regional (favorecer especies
localmente comunes cuando el sonido es ambiguo) ya no vive acá. Quedó
en [`LSDTector-BirdNET-custom-v1`](https://github.com/LSDArroyoGold/LSDTector-BirdNET-custom-v1),
separado a propósito: ese repo empaqueta el catálogo stock de BirdNET
(sin las 193 clases de este proyecto) con el ajuste regional aplicado
directamente encima. Mientras este repo siga en desarrollo activo del
reentreno en sí, conviene mantener las dos cosas desacopladas — cuándo
y cómo se combinan (reentreno + ajuste regional, los dos juntos) es una
decisión aparte, todavía sin tomar.

## Audio de las detecciones y espectrograma a espectro completo

`patches/reporting.py` es una versión parcheada de
`scripts/utils/reporting.py` de BirdNET-Pi: sube el bitrate del mp3 de
cada detección a 320kbps (por defecto, el encoder LAME le aplica un
filtro pasa-bajos que recorta el audio guardado a ~16kHz aunque el
archivo diga 48000 Hz — confirmado con FFT sobre una detección real) y
saca el remuestreo a 24kHz que traía la generación del espectrograma
(limitaba el eje de frecuencia visible a 12kHz). Ninguno de los dos
cambios afecta la detección en sí — el modelo analiza el `.wav` crudo a
48kHz antes de este paso — solo mejora la fidelidad de lo que se
guarda/muestra después. Para aplicarlo: copiar
`patches/reporting.py` sobre `~/BirdNET-Pi/scripts/utils/reporting.py`
y reiniciar `birdnet_analysis.service`.

## Qué NO incluye este repositorio (todavía)

Este repo resuelve únicamente el modelo en sí: entrenarlo, empaquetarlo,
y poder probarlo de forma aislada. No incluye base de datos, envío a
BirdWeather, manejo de energía, ni watchdog de batería; todo eso lo
resuelve BirdNET-Pi y el propio `LSD-Tector 2.0`.

La integración con BirdNET-Pi (que en algún momento pareció un problema
abierto, porque su selector de modelo tiene una lista fija de opciones
sin forma aparente de apuntar a un `.tflite` propio) ya está resuelta,
sin necesidad de parchear nada: al exportar en modo Append, el modelo
produce logits en el mismo formato que el modelo original, así que
BirdNET-Pi lo carga sin enterarse de la diferencia. Lo que falta ahora no
es esa integración, sino una versión del modelo que esté bien calibrada
para usarse con un umbral fijo por clase (ver `INFORME_FALLAS_V2.md`).
