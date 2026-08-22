# Informe: colapso de sensibilidad en audio real de campo, causa y fix

Fecha: 2026-08-21/22 (trabajo nocturno autónomo, sin supervisión directa
del usuario durante buena parte del proceso; validado paso a paso antes
de subir nada).

## 1. El problema no era solo Hornero/Kestrel

Al extender el chequeo de campo real (comparar la neurona original de
BirdNET contra la reentrenada, en el mismo clip real, sin depender de
que la carpeta esté bien etiquetada) a las especies con volumen
suficiente de detecciones reales, apareció que el colapso de
sensibilidad era generalizado, no un caso puntual:

| Especie | Clips reales confiables | Cruza umbral, original | Cruza umbral, reentrenada (v8) |
|---|---|---|---|
| Pitangus sulphuratus (Kiskadee) | 147 | 147/147 | 18/147 (12%) |
| Furnarius rufus (Hornero) | 67 | 67/67 | 8/67 (12%) |
| Zonotrichia capensis (Chingolo) | 35 | 35/35 | 8/35 (23%) |
| Agelaioides badius | 43 | 43/43 | 21/43 (49%) |
| Patagioenas picazuro | 47 | 47/47 | 24/47 (51%) |
| Sicalis flaveola | 8 | 8/8 | 0/8 (0%) |
| Colaptes campestris | 5 | 5/5 | 1/5 (20%) |
| Columbina picui | 3 | 3/3 | 0/3 (0%) |
| Nycticorax nycticorax | 5 | 5/5 | 1/5 (20%) |

## 2. Investigación de causa raíz

Se probaron, degradando audio limpio de Xeno-canto y midiendo si
reproducía el colapso: corte de banda a 16.4kHz (coincide con el
filtro típico de mp3 128kbps y se confirmó que TODOS los audios reales
de campo lo tienen), ruido de fondo real al mismo SNR medido en campo,
reverb de sala, resonancia de carcasa/campo cercano, clipping,
atenuación atmosférica por distancia, EQ de micrófono económico, AGC
dinámico, ganancia absoluta, jitter de tono, downmix estéreo, y
combinaciones. **Ninguna reprodujo el colapso.**

Se investigó también: mismatch de pipeline entre entrenamiento e
inferencia (descartado, procesamiento byte-idéntico verificado
empíricamente), estructura temporal/duetos (descartado, sin diferencia
estadística entre campo y Xeno-canto), sobreajuste por poca diversidad
del set de entrenamiento (contribuye una fracción modesta: 13% de
fallas incluso en Xeno-canto limpio no visto, muy por debajo del 88%
de falla en campo).

**Hallazgo más fuerte**: comparando los embeddings internos de BirdNET
(1024-dim, antes de cualquier clasificador) entre audio de
entrenamiento, Xeno-canto limpio, y audio real de campo, se encontró
que el audio de campo no está disperso al azar en el espacio de
embeddings (la distancia genérica al centroide es igual a la del audio
limpio), sino que se corre de forma sistemática y precisa exactamente
en la dirección que el clasificador usa para decidir cada especie,
incluso invirtiendo el signo. Esto explica por qué ninguna degradación
sintética genérica lo reproduce.

**Contribuyente identificado y medido**: interferencia de OTRAS
especies de ave vocalizando simultáneamente (coro real, no ruido
genérico) sí tumba el score de forma severa cuando la especie
interferente domina en volumen, y hay correlación real (aunque
moderada, r=-0.266) entre cantidad de especies co-vocalizando en un
clip real y la confianza de detección.

## 3. Fix aplicado: reentrenar con audio real de campo

Como ninguna simulación reproduce el problema, la única vía con
evidencia de funcionar es entrenar directamente con embeddings de
audio real capturado por el dispositivo (bajado de Drive vía rclone,
986 clips de 63 especies), no con Xeno-canto ni con Xeno-canto
degradado.

Se identificaron 8 especies con evidencia clara de fallo en campo real
(retención de detecciones < 50% respecto de la neurona original, con
n≥3 clips confiables): Furnarius rufus, Pitangus sulphuratus,
Zonotrichia capensis, Agelaioides badius, Sicalis flaveola, Colaptes
campestris, Columbina picui, Nycticorax nycticorax. Se reentrenó
**solo esas 8 neuronas** (más Falco sparverius, con refuerzo de
negativos duros, ver abajo), dejando las otras 185 especies
**byte-idénticas a la versión anterior** (verificado programáticamente
tras el reentreno). Riesgo de regresión en el resto del modelo: nulo
por construcción.

Metodología de reentreno: igual a la usada para corregir la falla de
calibración original (mismo pool de negativos: las otras 192 especies
+ ~1014 especies ajenas reales + ESC-50 no-evento; misma búsqueda de
hiperparámetro C; mismo umbral real de producción), pero agregando
como positivos adicionales los embeddings reales de campo, con un
split 70/30 a nivel de **clip completo** (no de ventana individual,
para no filtrar información de train a validación). El 30% de holdout
nunca se usó para entrenar nada, solo para medir la mejora real al
final.

### Caso especial: Falco sparverius (Kestrel)

La carpeta "Audios Kestrel (falsos)" (180 clips que el dispositivo
etiquetó como Kestrel en su momento) fue verificada a oído por el
usuario del proyecto: **son todos Hornero real, no Kestrel**. Estos
clips se usaron con doble propósito: como positivo adicional de
Furnarius rufus, y como **negativo duro real** de Falco sparverius (la
neurona de Kestrel se entrenó explícitamente para NO disparar ante
este audio). Esto ataca directamente la confusión Hornero/Kestrel
documentada desde el principio del proyecto, esta vez con datos reales
confirmados en vez de audio sintético.

No hay audio de campo con ground-truth confirmado de Kestrel real
disponible todavía, así que la neurona de Kestrel no tiene una
validación de recall sobre campo real propia; el chequeo existente
(71% de retención) puede estar parcialmente contaminado por el mismo
problema de origen (posible Hornero mal identificado por el propio
baseline) y se toma con cautela.

## 4. Resultado (holdout real, nunca visto en entrenamiento)

| Especie | Antes (v8) | Ahora | Mejora |
|---|---|---|---|
| Furnarius rufus | 1/154 (0.6%) | 154/154 (100%) | evidente |
| Pitangus sulphuratus | 7/136 (5%) | 134/136 (98.5%) | evidente |
| Zonotrichia capensis | 2/72 (3%) | 71/72 (98.6%) | evidente |
| Agelaioides badius | 7/60 (12%) | 56/60 (93.3%) | evidente |
| Sicalis flaveola | 0/10 (0%) | 9/10 (90%) | evidente |
| Colaptes campestris | 0/6 (0%) | 5/6 (83.3%) | evidente |
| Columbina picui | 2/6 (33%) | 6/6 (100%) | evidente |
| Nycticorax nycticorax | 0/8 (0%) | 4/8 (50%) | mejora, muestra chica |

Las 8 especies mejoraron, ninguna empeoró. Verificado también con el
pipeline completo de producción (`analyze()`, no solo álgebra de
pesos): 20 audios reales de Hornero, 20/20 detectados (media 0.951);
20 audios de "Kestrel falsos", 20/20 detectados como Hornero (media
0.949) y solo 2/20 disparan Kestrel (antes disparaba en la mayoría).

## 5. Ronda 2: reforzar también las especies "keep" pero limítrofes

Con el pipeline ya armado y validado, se extendió el mismo tratamiento
a 5 especies que habían quedado del lado "keep_custom" en la decisión
original pero con retención limítrofe (50-67%, no un colapso total
pero tampoco un resultado sólido): Patagioenas picazuro, Guira guira,
Vanellus chilensis, Certhiaxis cinnamomeus, Turdus amaurochalinus.
Mismo método: embeddings reales de campo como positivos adicionales,
split 70/30 por clip, partiendo esta vez de los pesos ya corregidos en
la sección anterior (no de v8), para no perder las mejoras ya hechas.

Resultado en holdout real (nunca visto en entrenamiento):

| Especie | Antes | Ahora |
|---|---|---|
| Patagioenas picazuro | 9/40 (22.5%) | 37/40 (92.5%) |
| Guira guira | 5/12 (41.7%) | 10/12 (83.3%) |
| Vanellus chilensis | 3/10 (30%) | 7/10 (70%) |
| Certhiaxis cinnamomeus | 1/8 (12.5%) | 6/8 (75%) |
| Turdus amaurochalinus | 2/6 (33.3%) | 5/6 (83.3%) |

Total de especies tocadas en toda la sesión: 14 (las 8 de la sección 3,
más Falco sparverius con refuerzo de negativos, más estas 5). Las
otras 179 especies quedan byte-idénticas a la versión previa a esta
sesión (verificado programáticamente contra `pesos_v8_ovr.npz`
original, no solo contra el paso intermedio).

## 6. Validación final independiente, sobre las 986 detecciones reales completas

Después de la ronda 2, se repitió la comparación baseline vs
reentrenado (misma metodología de la sección 1: neurona original
contra la nueva, en el mismo clip real, sin depender de etiquetas de
carpeta) sobre el archivo completo de 986 detecciones reales, no solo
sobre el 30% de holdout. Esta corrida es independiente del proceso de
entrenamiento en sí (usa el pipeline real `analyze()`, no álgebra de
pesos) y sirve como confirmación final.

| Especie | Antes (v8) | Ahora (v10) |
|---|---|---|
| Furnarius rufus | 8/67 (12%) | 67/67 (100%) |
| Pitangus sulphuratus | 18/147 (12%) | 147/147 (100%) |
| Zonotrichia capensis | 8/35 (23%) | 35/35 (100%) |
| Agelaioides badius | 21/43 (49%) | 43/43 (100%) |
| Sicalis flaveola | 0/8 (0%) | 8/8 (100%) |
| Colaptes campestris | 1/5 (20%) | 5/5 (100%) |
| Columbina picui | 0/3 (0%) | 3/3 (100%) |
| Nycticorax nycticorax | 1/5 (20%) | 5/5 (100%) |
| Patagioenas picazuro | 24/47 (51%) | 47/47 (100%) |
| Guira guira | 8/14 (57%) | 14/14 (100%) |
| Vanellus chilensis | 5/8 (62%) | 8/8 (100%) |
| Certhiaxis cinnamomeus | 1/2 (50%) | 2/2 (100%) |
| Turdus amaurochalinus | 2/3 (67%) | 3/3 (100%) |
| Falco sparverius | 90/127 (71%) | 15/127 (12%) |

Las 13 especies con positivos nuevos llegan a 100% (esperable: esta
medición incluye clips que sí se usaron en entrenamiento, a diferencia
del holdout de la sección 4; el valor real de generalización es el de
esa sección). Las especies no tocadas dieron el mismo resultado que
antes, punto por punto (verificación adicional de que nada se rompió).

**Sobre la caída de Falco sparverius (71% → 12%): es el resultado
esperado y correcto, no una regresión.** Esas 127 detecciones
"confiadas" del catálogo original probablemente son, en su mayoría, el
mismo audio de Hornero mal identificado como Kestrel por el propio
baseline (la razón por la que la carpeta "Kestrel (falsos)" existe).
La neurona de Kestrel reentrenada ahora rechaza correctamente ese
audio, que es exactamente lo que se le pidió con los negativos duros
de la sección 3. Sigue sin existir una medición limpia del recall de
Kestrel sobre Kestrel real, porque no hay audio de campo con esa
especie confirmada todavía.

## 7. Qué falta

- Validación en hardware real (Raspberry Pi), como toda entrega de
  este proyecto.
- Conseguir audio de campo con Kestrel real confirmado para poder
  medir su recall en campo, no solo su tasa de falsos positivos.
- Nycticorax nycticorax tiene muestra de validación muy chica (8
  clips), seguir monitoreando en campo.
- El hallazgo de interferencia por coro de otras especies quedó
  identificado pero no resuelto de raíz; si el problema reaparece en
  otras especies a futuro, ese es el ángulo a seguir.
- Myiopsitta monachus, Megaceryle torquata, Colaptes melanochloros y
  Tringa flavipes quedaron marcadas "revisar" (muy poca data real, 1-2
  clips, 0 aciertos) pero sin tocar: no hay evidencia suficiente para
  reentrenar con confianza aunque el patrón sea sospechoso. Si aparece
  más audio de campo de estas especies, revisar de nuevo.
