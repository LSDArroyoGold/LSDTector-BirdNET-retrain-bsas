# Validación independiente del modelo v10 y decisión de no desplegarlo (todavía)

Fecha: 21-22/8. Todo lo de este informe se hizo **localmente**, sin tocar el
Tector 2 (ni el dispositivo ni sus repos), para poder probar a fondo el
modelo v10 (`modelo/LSDTector_Classifier_v2.tflite`, el de
`INFORME_FIX_AUDIO_REAL.md`) antes de decidir si instalarlo en campo.
Los scripts usados están en [`validacion/scripts/`](validacion/scripts/) y
los resultados crudos en [`validacion/resultados/`](validacion/resultados/).

**Resultado final: no se instala v10 en tector2 por ahora.** Se sigue
usando el modelo original de BirdNET + el filtro de geolocalización
(`LSDTector-BirdNET-custom-v1`) + el fix de ancho de banda de audio (este
mismo repo). El motivo no es que v10 esté "mal" — mejora mucho el
problema de Hornero/Kestrel — sino que la validación encontró un problema
de fondo en 3 de las 14 neuronas reentrenadas que conviene resolver antes
de reemplazar el modelo en producción. Está todo detallado abajo.

## 0. Entorno de prueba

Venv local con `tensorflow`, `librosa`, `numpy`, `resampy`, `soundfile`,
corriendo el `.tflite` con el intérprete de TFLite directamente (mismas
`SENSITIVITY`/`CONFIDENCE` que produccion: 1.1 / 0.6, salvo donde se indica
lo contrario). Un detalle importante que costó descubrir y que hay que
tener siempre presente al tocar este modelo: el archivo de labels mezcla
dos formatos —

- Las primeras 6522 entradas son el catálogo original de BirdNET, sin
  tocar, formato `NombreCientifico_NombreComún` (ej.
  `Furnarius rufus_Rufous Hornero`).
- Las últimas 193 son las clases reentrenadas de este proyecto, formato
  bare, **solo** nombre científico (ej. `Furnarius rufus`, sin el
  `_NombreComún`).

Usar el índice equivocado (buscar `Furnarius rufus_Rufous Hornero` cuando
se quiere medir la neurona reentrenada) invalida cualquier resultado.
Todos los tests de este informe usan el índice bare correcto.

## 1. Hornero/Kestrel con audio real de campo (script 1)

255 clips reales de tector1 (`Audios confusion/`: 76 Rufous_Hornero + 179
American_Kestrel — esta segunda carpeta está mal etiquetada por el
BirdNET viejo, son todos Hornero reales, confirmado a oído por el
usuario). Resultado: la neurona reentrenada de Hornero cruza 0.6 en 98.7%
/ 100% de los clips (contra 26.3%/2.2% de la neurona original — esto
explica la mala clasificación original). Hornero le gana a Kestrel en el
100% de los casos, en ambas carpetas, con SENSITIVITY 1.1 y 1.25.

## 2 y 3. Verificación con Xeno-canto (scripts 2 y 3)

Con la API key del usuario se bajaron 63 grabaciones de Hornero + 18 de
Kestrel, ambas de Argentina, calidad Xeno-canto A (script `2`). Sobre 548
segmentos totales: **cero falsos positivos cruzados en ninguna dirección**
(ni un segmento de Hornero disparó Kestrel por encima del umbral, ni
viceversa). La sensibilidad absoluta en este audio externo es más baja que
en el de campo (52-55% de los archivos con al menos una detección) — mismo
patrón descrito en `INFORME_FIX_AUDIO_REAL.md`: la neurona quedó
especializada en la firma acústica real de campo, no en "Hornero en
general", pero sin generar confusión cruzada.

## 4. Barrido de SENSITIVITY (script 3)

Se precalcularon los logits crudos de los 4 datasets (campo Hornero, campo
Kestrel/Hornero-mal-etiquetado, Xeno-canto Hornero, Xeno-canto Kestrel) y
se barrió `SENSITIVITY` de 0.9 a 1.4. El audio de campo se mantiene
prácticamente plano (98.7-100%) en todo el rango. El audio de Xeno-canto sí
es sensible (55.6% de detección a 0.9, bajando a 46.0% a 1.4). Un contador
de "sobre-disparo" (≥5 especies cruzando umbral en el mismo segmento)
aumenta a medida que baja SENSITIVITY. **Conclusión y decisión, con el
usuario dándole más peso al audio de campo real que a Xeno-canto:
mantener CONFIDENCE=0.6 y SENSITIVITY=1.1**, ambos ya así en producción,
sin cambios. 1.1 y 1.2 dan resultados prácticamente idénticos en toda
métrica medida.

## 5. Prueba amplia sobre las 986 detecciones de tector1, 37 especies (script 4)

Objetivo: buscar confusiones nuevas más allá de Hornero/Kestrel. Se
descartó la idea original de usar una corrida fresca del BirdNET original
como "referencia confiable" — el usuario señaló correctamente que sería
circular, porque el propio BirdNET original (corriendo en tector1) es el
que generó esas etiquetas en primer lugar. En cambio: se usó la etiqueta
de nombre de archivo (que ya es la opinión del BirdNET viejo) filtrada por
confianza original ≥50%, como mejor proxy disponible sin acceso a Merlin,
y se comparó contra el top1 de v10 en todo el catálogo de 6715 clases.

Resultado (382 clips, 25 especies): la mayoría de las especies con
volumen suficiente salió bien (Grayish_Baywing, Rufous-collared_Sparrow,
Rufous_Hornero, Guira_Cuckoo: 100% de coincidencia). Pero aparecieron 3
neuronas que le "ganaban" el top1 a muchas especies **distintas y no
relacionadas entre sí**: `Zonotrichia capensis` (Chingolo), `Guira guira`
(Pirincho) y `Agelaioides badius` (Tordo músico) — las tres, justo, parte
de las 14 neuronas reentrenadas en v10. Afectaba sobre todo a
Great_Kiskadee (n=137, la muestra más sólida del dataset) y en menor
medida a Southern_Lapwing, Picazuro_Pigeon, Saffron_Finch,
Yellow-chinned_Spinetail, entre otras.

Detalle completo en
[`validacion/resultados/registros_test_amplio_37_especies.csv`](validacion/resultados/registros_test_amplio_37_especies.csv).

## 6. Verificación manual con Merlin (scripts 4 y 7)

Para no quedarse solo con la comparación top1 (potencialmente engañosa,
ver corrección en el punto 7), se armó un pool de 28 clips reales —
mayor y menor confianza original de cada combinación víctima/especie
"ladrona" — y el usuario los pasó uno por uno por Merlin (app de Cornell)
con [`validacion/scripts/7_revisar_manual_merlin.py`](validacion/scripts/7_revisar_manual_merlin.py),
un script interactivo que reproduce cada clip (decodificando el mp3 con
librosa/soundfile y reproduciendo con `paplay`, ya que el equipo no tenía
`ffmpeg`/`mpv`/`ffplay` instalados), permite repetir cuantas veces haga
falta, y guarda todo progresivamente en un CSV.

Resultado completo en
[`validacion/resultados/verificacion_manual_merlin.csv`](validacion/resultados/verificacion_manual_merlin.csv).
Resumen:

| Víctima (label BirdNET viejo) | n verificado | Merlin confirma la víctima | Merlin confirma la "ladrona" (v10) | Otra cosa / ninguna |
|---|---|---|---|---|
| Great_Kiskadee | 6 | **6/6** | 0 | 0 |
| Southern_Lapwing | 2 | **2/2** | 0 | 0 |
| Picazuro_Pigeon | 2 | 1 (+1 ambiguo, menciona ambas) | 0 | 0 |
| Saffron_Finch | 2 | 1 | 1 | 0 |
| Buff-browed_Foliage-gleaner | 1 | 0 | 1 | 0 |
| Creamy-bellied_Thrush | 1 | 1 (ambiguo, menciona ambas) | 0 | 0 |
| Gray-cowled_Wood-Rail | 2 | 1 ambiguo (menciona ambas) | 0 | 1 skip |
| American_Kestrel (folder ya sabido mal etiquetado) | 6 | 0 | 0 | 5 Hornero, 1 skip |
| Yellow-chinned_Spinetail | 3 | 0 | 0 | 2 Hornero, 1 skip |
| Black-crowned_Night-Heron | 1 | 0 | 0 | 1 skip |
| Solitary_Sandpiper | 2 | 0 | 0 | 2 skip |

## 7. Corrección importante: esto no es "v10 le roba la detección a Kiskadee"

Al revisar el detalle por archivo (columna `propio_score` en el CSV del
punto 5), en los 16 casos de Kiskadee la neurona propia de Kiskadee
**también** cruza el umbral con confianza altísima (0.88 a 0.996). En
producción real, BirdNET-Pi aplica umbral independiente por clase (no
"gana el más alto") — así que Kiskadee se sigue detectando bien. Lo que
pasa es que Chingolo/Pirincho/Tordo músico **también** cruzan el umbral en
el mismo segmento, generando una detección falsa extra al lado de la
correcta. Es un problema real (contamina el log de detecciones,
infla las estadísticas de esas 3 especies) pero más acotado de lo que
parecía al principio.

## 8. Se intentó calibrar un umbral por especie — no alcanza (script 5)

Idea: en vez de un único CONFIDENCE global para las 193 especies, usar un
umbral más alto específico para las 3 neuronas problemáticas (recomendación
estándar de la literatura de BirdNET, ver punto 10). Usando el propio
audio real de tector1 de cada "ladrona" como positivo y el de sus
"víctimas" como negativo:

| Especie | Umbral necesario para falsos positivos ≈0% | Recall propio a ese umbral |
|---|---|---|
| Guira guira | 0.99 | 100% (sigue perfecto) |
| Agelaioides badius | 0.995-0.999 | 92-100% |
| Zonotrichia capensis | **ni con 0.999 baja de 0.6%** | 26.9% (se pierde 73% del recall real) |

El usuario, correctamente, rechazó este approach: necesitar un umbral de
0.99 para separar es señal de un problema de fondo, no algo para tapar con
un número. **No se aplicó ningún cambio de umbral.**

## 9. Diagnóstico de la causa raíz: atajo de dispositivo/sitio (script 6)

Hipótesis: las 14 neuronas reentrenadas de v10 se entrenaron **solo** con
audio real positivo de tector1 (mismo micrófono, mismo sitio). Si los
negativos usados durante el reentreno no vinieron también grabados por
tector1 en las mismas condiciones, la red pudo aprender a reconocer "esto
lo grabó tector1" en vez del canto real de la especie — un atajo espurio
casi tan predictivo como el canto en el propio set de entrenamiento, pero
que no generaliza.

Prueba: correr las 3 neuronas problemáticas sobre las 81 grabaciones de
Xeno-canto de Hornero/Kestrel de Argentina (mismas especies, pero grabadas
con otro equipo, en otro sitio — nada que ver con tector1).

| Especie | Falsos positivos en audio de tector1 (otras especies) | Falsos positivos en audio externo (Xeno-canto) |
|---|---|---|
| Zonotrichia capensis | 41.4% | **0.0%** |
| Guira guira | 29.1% | **0.0%** |
| Agelaioides badius | 35.9% | **3.7%** (contra ~1.2% a umbral 0.9) |

La diferencia es enorme y en la dirección exacta que predice la hipótesis
del atajo. **Diagnóstico: las neuronas de Chingolo, Pirincho y Tordo
músico del v10 no aprendieron el canto de la especie — aprendieron algo
del micrófono/ambiente acústico propio de tector1.** El fix no es de
umbral ni de calibración: es de datos de entrenamiento. Esas 3 neuronas
necesitan negativos duros que también sean grabaciones reales de tector1
(de otras especies, o de silencio/ambiente del sitio), para que la red no
pueda usar la firma del dispositivo como atajo.

## 10. Research: por qué Merlin no tiene este problema

Investigación (con fuentes) sobre por qué Merlin Sound ID (Cornell) generaliza
mucho mejor que BirdNET a audio de campo real, y cómo hace para detectar
dos especies cantando en el mismo clip:

- **Datos de entrenamiento con etiquetas temporalmente precisas.** Merlin
  usa 140h de audio con anotación experta del segundo exacto en que canta
  cada especie (no "este clip de 30s es fulano"), curado exclusivamente
  por Macaulay Library/eBird. BirdNET usa etiquetas débiles (clip completo
  = una especie), sin garantía de que no haya otra cantando de fondo — el
  mismo problema que probablemente causó el atajo de dispositivo del punto 9.
  [Behind the Scenes of Sound ID in Merlin](https://www.macaulaylibrary.org/2021/06/22/behind-the-scenes-of-sound-id-in-merlin/)
- **"Focal vs soundscape domain shift"**: es el nombre técnico, documentado
  y activamente investigado en la literatura de bioacústica, de exactamente
  el problema que describe `INFORME_FIX_AUDIO_REAL.md` (el audio de campo
  corre el embedding en una dirección que el audio limpio de archivo no
  reproduce). No es un problema exclusivo nuestro.
  [BirdNET: A deep learning solution for avian diversity monitoring (Kahl et al. 2021)](https://connormwood.com/wp-content/uploads/2021/02/kahl.etal-2021-birdnet-a-deep-learning-solution-for-avian-diversity-monitoring.pdf),
  [BirdNET: applications, performance, pitfalls and future opportunities (Pérez-Granados 2023)](https://onlinelibrary.wiley.com/doi/full/10.1111/ibi.13193)
- **Precedente directo: HawkEars** (2025), clasificador regional para
  Canadá (314 especies, mismo espíritu que este proyecto). Recall 2-4x
  mejor que BirdNET a igual precisión, usando etiquetas "fuertes" generadas
  por búsqueda de embeddings (filtran del set de entrenamiento los clips
  contaminados con otras especies de fondo) y catálogo regional acotado.
  [HawkEars: A regional, high-performance avian acoustic classifier](https://ftp-public.abmi.ca/home/publications/documents/656_Huus_etal_2025_HawkEarsAcousticClassifier.pdf)
- **Umbral por especie, no global**: guía oficial del propio equipo de
  BirdNET — los scores no son probabilidades calibradas y el umbral óptimo
  varía mucho por especie. Es la idea que se probó en el punto 8 (no
  alcanzó para Chingolo, pero confirma que el enfoque en sí es válido y
  reconocido).
  [Guidelines for appropriate use of BirdNET scores (Wood & Kahl 2024)](https://connormwood.com/wp-content/uploads/2024/02/wood-kahl-2024-guidelines-for-birdnet-scores.pdf)

## 11. Decisión y plan a futuro

**Por ahora**: seguir en producción con el modelo original de BirdNET +
filtro de geolocalización (`LSDTector-BirdNET-custom-v1`, validado, mejora
el ratio Hornero/Kestrel de 10% a 50% sin excluir nada de forma dura) +
fix de ancho de banda de audio (este repo). v10 **no** se instala en
tector2 todavía.

**Plan para resolver la causa raíz**: el dispositivo "Tector Home"
(enchufado, grabando y detectando de forma continua, no por ventanas)
puede ser la vía para juntar los datos que hacen falta: además de
detecciones, programarlo para grabar ~10 minutos diarios de ruido
ambiente en horas de poca actividad aviar, específicamente para tener
negativos reales grabados con el mismo equipo — la pieza que le faltó a
las 3 neuronas problemáticas del punto 9.

**El problema de fondo pendiente**: sin importar cuántos datos nuevos se
junten, las detecciones de un dispositivo no supervisado las etiqueta la
propia red — no hay referencia certera automática de qué especie era en
realidad. Enfoque propuesto (sin implementar todavía, para cuando llegue
el momento): no verificar manualmente todo el volumen, sino usar el
**desacuerdo entre modelos independientes** (BirdNET original vs el
retrain, eventualmente vs Merlin) como disparador de revisión — donde
coinciden, alta confianza sin revisar nada; donde discrepan, ahí es donde
vale la pena una verificación manual (como se hizo en el punto 6). Así se
prioriza esfuerzo humano en los casos dudosos en vez de revisar todo el
volumen.
