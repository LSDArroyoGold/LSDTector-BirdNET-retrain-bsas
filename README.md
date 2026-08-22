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

## Ajuste regional por frecuencia (opcional)

Además del modelo universal, este repositorio permite generar una versión
ajustada a la región donde se instala cada dispositivo: a cada una de las
193 especies locales se le suma un sesgo proporcional a su frecuencia
real de observación en esa región, para que una especie localmente común
le gane más fácil a una localmente rara cuando el sonido es ambiguo. El
ajuste está acotado y nunca descarta ninguna especie por completo, solo
reordena el margen de confianza.

**Fuente preferida: archivos ya descargados a mano, sin conexión a eBird
desde el dispositivo.** La API pública de eBird no tiene un endpoint de
frecuencia (esa estadística, el "bar chart" del sitio, requiere sesión
logueada, y sus términos de uso restringen el uso comercial sin un
acuerdo de licencia aparte). Para no atar el proyecto a esos términos, la
frecuencia de cada región se descarga preferentemente **a mano, una vez,
desde una cuenta de eBird propia** (el sitio de eBird, no la API), y se
versiona en este repositorio bajo `frecuencias/<código-de-región>.txt`
(formato de exportación de bar chart, sin modificar). El dispositivo en
el campo hace reverse geocoding (lat/lon → código de región, vía
Nominatim/OpenStreetMap, datos abiertos, sin restricción de uso
comercial) y busca si ya existe el archivo de esa región, primero
localmente y si no lo tiene, lo descarga de este mismo repositorio por
`raw.githubusercontent.com` (no de eBird).

Para agregar una región nueva: entrar a `ebird.org/barchart` logueado,
elegir la región (código tipo ISO 3166-2, ej. `AR-B` para Buenos Aires),
descargar el archivo, y subirlo a `frecuencias/<código>.txt` en este
repositorio. `AR-B` ya está cargado como ejemplo.

**Respaldo opcional: API pública de eBird.** Si todavía no se cargó el
archivo de una región, y se configuró una API key de eBird (gratis,
[`ebird.org/api/keygen`](https://ebird.org/api/keygen), variable
`EBIRD_API_KEY` en `config_general.txt` de `LSD-Tector2.0`), el
dispositivo la usa como respaldo: observaciones recientes (últimos 30
días, el máximo que permite la API) como proxy de frecuencia. Es una
muestra bastante más chica y ruidosa que el bar chart histórico (se
prefiere el archivo siempre que exista), pero sirve para cubrir una
región nueva sin depender de que alguien la haya descargado a mano
todavía. **Importante si el dispositivo llega a venderse como producto**:
la API pública de eBird está sujeta a los términos de uso de eBird/Cornell
Lab, que restringen el uso comercial sin un acuerdo de licencia aparte
(ver la sección "Solicite un acuerdo de licencia" en el alta de la API
key). Este respaldo se agregó para uso académico/de investigación del
proyecto tal como está hoy; si el uso comercial se vuelve una posibilidad
real, conviene revisar esto con eBird antes de seguir dependiendo de la
API en producción, o directamente no configurar `EBIRD_API_KEY` y
depender solo de los archivos ya descargados a mano (que no tienen esta
restricción, porque nunca llaman a la API desde el dispositivo). Sin
archivo de región y sin API key configurada, el dispositivo sigue con el
modelo universal sin ajustar, sin error ni bloqueo en ningún caso.

En el dispositivo, esto corre automáticamente vía
`scripts/aplicar_ajuste_regional.sh` (en `LSD-Tector2.0`), después de
`actualizar_modelo.sh`, pero necesita el entorno `~/birdnet-v2-env` de
este mismo repositorio (`bash instalar.sh`, ver arriba) para poder
reexportar el `.tflite`. Sin ese entorno, o sin archivo de región
todavía, el dispositivo sigue con el modelo universal, que es siempre el
comportamiento por defecto.

Para generarlo a mano (por ejemplo para revisar el resultado antes de
confiar en el automatismo):

```bash
source ~/birdnet-v2-env/bin/activate
python3 generar_modelo_regional.py --lat -34.92 --lon -57.95
```

Guarda el `.tflite`, las labels, y un `ajuste_regional_meta.json` con el
detalle especie por especie (frecuencia usada y ajuste aplicado) en
`modelo_regional/`, para poder auditar exactamente qué se ajustó.

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
