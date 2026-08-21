# LSDTector: clasificador BirdNET v2

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
> ⚠️ **Pero: confusión Hornero/Kestrel confirmada, sin resolver (21/8).**
> Primera prueba real de campo con v2 corregido: con audio real de
> *Furnarius rufus*, el modelo nunca cruza el umbral de producción para
> Hornero (0/35 segmentos en ~4 min) y da dos falsos positivos confiados
> de *Falco sparverius* (0.79 y 0.92). Contradice las métricas de
> validación reportadas para al menos este par de especies. Ver
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
