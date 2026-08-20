# Informe de fallas — LSDTector_Classifier_v2

Fecha: 2026-08-20
Contexto: primera instalación real en hardware de campo (tector2, LSD-Tector 2.0),
integrado al pipeline de producción de BirdNET-Pi (no al harness aislado de
`probar_modelo.py`).

## Resumen

El modelo `LSDTector_Classifier_v2.tflite` (modo *Append*, 6715 salidas =
6522 del catálogo global de BirdNET + 193 especies locales) dispara
sistemáticamente **entre 15 y 25 especies distintas con confianza >95%
simultáneamente, en el mismo segmento de 3 segundos**, incluyendo
combinaciones biológicamente imposibles en el sitio real de despliegue
(GBA). No es un problema de configuración, red, ni del pipeline de
integración de BirdNET-Pi: se confirmó aislando el modelo y comparándolo
contra el modelo original.

**Esto invalida cualquier detección producida por este modelo hasta que
se corrija.** No es una falla intermitente ni de bajo impacto: en la
prueba real, prácticamente todos los clips analizados dispararon docenas
de falsos positivos, y el volumen de trabajo extra (extracción a mp3 +
POST a BirdWeather por cada falso positivo) generó una cola de análisis
que crecía más rápido de lo que se vaciaba, dejando el dispositivo sin
capacidad real de detección en tiempo real.

## Evidencia concreta

### 1. Logits crudos, antes de la sigmoide

Sobre un clip real de 3 segundos grabado en el sitio de despliegue:

```
logits: min=-21.295  max=6.927  mean=-12.258  std=3.059
90 clases con logit > 0   (de 6715)
43 clases con logit > 2
12 clases con logit > 5
59 clases con sigmoide > 0.7   (umbral CONFIDENCE de producción)
22 clases con sigmoide > 0.95
```

Top de especies detectadas simultáneamente en ese único clip (todas >97%
de confianza):

| Especie | Confianza | Por qué es imposible/absurdo |
|---|---|---|
| Black-browed Albatross (*Thalassarche melanophris*) | 99.3% | Ave pelágica oceánica. Nunca aparece en un patio de GBA. |
| Crested Myna (*Acridotheres cristatellus*) | 98.6% | Especie asiática, no presente en Sudamérica. |
| Rock Pigeon, American Kestrel, Masked Gnatcatcher, Picazuro Pigeon, House Sparrow, Great Egret, Burrowing Owl, Neotropic Cormorant, White-throated Hummingbird, Screaming Cowbird... | 97–99% | Especies reales de la región, pero es imposible que canten 10+ simultáneamente en el mismo segmento de 3s. |

Un BirdNET bien calibrado, en un clip de campo real, típicamente da 0-3
detecciones por segmento de 3s. Esto es un patrón de "dice que sí a casi
todo", no una lectura real del contenido acústico.

### 2. Comparación directa: modelo reentrenado vs. modelo original, mismo hardware, mismo pipeline

Se restauró el `.tflite` y `_Labels.txt` originales de BirdNET (6522
especies, recuperados directamente del historial de git de BirdNET-Pi, sin
tocar nada más del pipeline) y se corrió sobre la misma cola de clips
pendientes, en el mismo dispositivo:

| | Modelo reentrenado (Append, 6715 clases) | Modelo original BirdNET (6522 clases) |
|---|---|---|
| Ritmo de análisis | ~1 clip de 15s cada ~60s reales (más lento que tiempo real → cola creciendo sin parar) | ~1 clip cada ~1-2s reales (mucho más rápido que tiempo real) |
| Detecciones por clip | 15-25 especies simultáneas, >95% confianza | 0-2 especies por clip, umbral 0.7 |
| Cola de análisis pendiente | Creciendo sin control (231 → 240 → 267 clips en ~20 min) | Bajando de forma sostenida (267 → 146 clips en pocos minutos) |

Mismo hardware, mismo `birdnet.conf`, mismo `CONFIDENCE=0.7` y
`SENSITIVITY=1.25`, mismo código de BirdNET-Pi sin modificar. La única
variable que cambió fue el archivo `.tflite`/labels. Esto aísla la falla
al modelo en sí, no a la integración ni al dispositivo.

### 3. Se descartó una hipótesis: no es un problema de activación (sigmoid vs softmax)

Se sospechó inicialmente que la capa "Append" pudiera estar pensada para
softmax y BirdNET-Pi le estuviera aplicando sigmoide por error. Se
confirmó revisando el código fuente de `birdnet_analyzer`
(`train/custom_models.py`): el modo *Append* built-in oficialmente
concatena logits y aplica `tf.sigmoid()` — exactamente lo mismo que hace
`scripts/utils/models.py` de BirdNET-Pi en producción
(`1/(1+exp(-sensitivity*logit))`). **La activación es la correcta.** La
falla está en la calibración de los logits que produce la capa de
decisión nueva, no en cómo se los procesa después.

## Hipótesis de causa raíz (a confirmar por quien reentrene)

El propio README de este repo dice textualmente:

> "...por qué falló el primer intento, y por qué se terminó entrenando
> con regresión logística en vez del pipeline propio de BirdNET-Analyzer..."

Esto es la pista más fuerte. El pipeline oficial de `birdnet_analyzer`
(`train/utils.py`) no solo entrena la capa nueva: además calcula, por
cada clase, un **umbral óptimo calibrado** (`macro_precision_opt`,
`macro_recall_opt`, `macro_f1_opt`, buscado por clase) además del umbral
por defecto, precisamente porque un único umbral global (como el
`CONFIDENCE=0.7` que usa BirdNET-Pi en producción, aplicado por igual a
las 6715 clases) no funciona bien sin esa calibración por clase.

Si el reentreno con regresión logística (probablemente scikit-learn,
sobre embeddings extraídos) se hizo sin ese paso de calibración por
clase, y las métricas reportadas en el README (~63-86% "accuracy") se
midieron con una metodología de tipo top-1/argmax (una sola predicción
por clip, la de mayor score) en vez de con el umbral fijo de producción
aplicado independientemente a cada una de las 6715 clases (que es como
realmente se usa el modelo en BirdNET-Pi), es exactamente el tipo de
mismatch que explica todo lo observado:

- En validación top-1, con LSDTector_Classifier_v2 devolviendo *muchas*
  clases con score alto, el argmax puede seguir siendo mayormente
  correcto → buena "accuracy" reportada.
- En producción, sin argmax, se listan *todas* las clases que superan
  0.7 → docenas de falsos positivos por clip.

Esto también explicaría por qué la prueba original (`probar_modelo.py`,
`min_conf=0.1`, vía `birdnet_analyzer.analyze()`) no mostró este problema
tan claramente: conviene revisar si esa vía de análisis limita de algún
modo la cantidad de detecciones reportadas por segmento, a diferencia del
loop de `birdnet_analysis.py` de BirdNET-Pi que no tiene ese límite.

## Qué se necesita para reentrenar bien

1. **Evaluar con la misma metodología que producción usa de verdad**:
   umbral fijo (0.7) aplicado de forma independiente a cada una de las
   6715 clases (multi-label), no top-1/argmax. Reportar precision/recall
   por clase a ese umbral, no solo "accuracy" global.
2. **Calibrar el umbral por clase**, o al menos verificar qué rango de
   logits/sigmoide produce la capa nueva vs. la capa original, y si son
   comparables en escala. Si no lo son, un único `CONFIDENCE` global en
   `birdnet.conf` nunca va a funcionar bien para ambas partes del modelo
   combinado.
3. Si se sigue usando regresión logística externa en vez del pipeline de
   `birdnet_analyzer.train`, revisar la regularización (`C` en
   scikit-learn) y la cantidad/diversidad de ejemplos negativos usados
   por clase — con pocos negativos por clase es fácil que el clasificador
   aprenda "esto no es silencio" en vez de "esto es esta especie
   puntual", lo que empuja muchas clases a la vez hacia score alto ante
   cualquier audio con contenido acústico real.
4. Antes de instalar en un dispositivo de campo, correr una prueba que sí
   reproduzca el comportamiento real: tomar 5-10 clips de 15s reales de
   sonido ambiente típico del sitio de despliegue (no solo un canto
   limpio de una especie) y contar cuántas detecciones por clip da el
   modelo al umbral 0.7 real — si da más de 2-3 por clip de forma
   consistente, no está listo.

## Estado actual de tector2 (mientras tanto)

Se revirtió el modelo al original de BirdNET (recuperado con
`git checkout` desde el propio historial de `~/BirdNET-Pi`, sin necesidad
de descargar nada externo) para que el dispositivo siga dando detecciones
utilizables mientras se corrige el reentreno. El dispositivo pierde el
enfoque regional de las 193 especies locales hasta que haya una versión 3
del clasificador, pero vuelve a funcionar de forma confiable con el
catálogo global de BirdNET.
