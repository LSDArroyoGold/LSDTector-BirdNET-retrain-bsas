# Informe: confusión Furnarius rufus / Falco sparverius en v2 (post-fix)

## CORRECCIÓN IMPORTANTE (21/8, más tarde el mismo día)

**Todo lo que sigue midió la neurona equivocada.** El archivo de labels
tiene 6715 entradas: las primeras 6522 (formato
`NombreCientífico_NombreComún`) son el catálogo global original de
BirdNET, **sin modificar por este proyecto**. Las 193 especies
realmente reentrenadas están en los últimos 193 índices (6522-6714),
con formato de **solo nombre científico, sin guión bajo** (ej.
`Furnarius rufus`, no `Furnarius rufus_Rufous Hornero`).

Este informe usó `labels.index('Furnarius rufus_Rufous Hornero')` y
`labels.index('Falco sparverius_American Kestrel')` — es decir, midió
la neurona **original de BirdNET, intacta**, no la que este proyecto
reentrena y calibra. Todo lo que dice más abajo sobre "la falla de
calibración de v2" y "la confusión sigue sin resolver" es sobre esa capa
original, que nunca fue tocada por v2 ni por el ajuste regional.

Confirmado directamente: comparando logits de dos exports del mismo
modelo base con distinto `--alpha` (0.6 vs 1.2) sobre el mismo input
sintético, la neurona `Furnarius rufus_Rufous Hornero` (índice
combinado) tiene diferencia **exactamente 0.0** entre ambos exports —
prueba de que nunca recibió el ajuste. La neurona `Furnarius rufus`
(índice bare, la real) sí difiere (+1.68 en logit).

**Con la neurona correcta**, midiendo sobre las mismas grabaciones de
campo reales de Hornero: los scores son bastante más bajos en general
(típicamente 0.00-0.2, rara vez más) que lo que este informe reportó
usando la neurona equivocada. El ajuste regional (`--alpha`) sí mueve la
aguja cuando ya hay algo de señal real (en un caso, 0.13 → 0.41 al
duplicar alpha), pero no genera señal donde el extractor casi no ve nada
de la especie — en esos segmentos gana Kestrel por default, con valores
igual de bajos (0.01-0.07), sin ser una confusión fuerte, simplemente
ausencia de señal de ambas clases.

**Conclusión revisada**: no hay evidencia de una confusión fuerte y
sistemática Hornero/Kestrel en la capa realmente reentrenada (a
diferencia de lo que este informe concluía originalmente sobre la capa
original). Lo que sí parece real: la sensibilidad de la neurona
reentrenada de Hornero es baja para varias grabaciones de campo
puntuales -- vale la pena revisar si hace falta más/mejores ejemplos de
entrenamiento para esa especie específica, no necesariamente un problema
de confusión entre clases.

El contenido original de este informe queda abajo, sin editar, como
referencia de lo que se midió (con la salvedad de que fue sobre la
neurona equivocada).

---

Fecha: 2026-08-21
Contexto: primera prueba real en campo del modelo v2 corregido (commit
`b9d5e77`, el que resuelve la falla de calibración documentada en
`INFORME_FALLAS_V2.md`), instalado y corriendo en producción en tector2
(LSD-Tector 2.0). Esta prueba es posterior y separada de esa corrección:
la calibración general está resuelta, pero apareció un problema nuevo,
específico de un par de especies.

## Resumen

Con audios de campo reales de Hornero (*Furnarius rufus*) — los mismos
que el modelo v1 (antes del reentreno) confundía con confianza alta como
American Kestrel (*Falco sparverius*) — **la confusión sigue presente en
v2**, cuantificada con la misma metodología exacta de producción (umbral
fijo 0.7, sigmoide independiente por clase, sin top-1/argmax). Esto
contradice el 85.4% de recall y 87.3% de accuracy reportados en la
validación de v2, al menos para este par de especies puntual.

## Lo que SÍ funciona bien en v2 (para que quede balanceado)

- **La falla de calibración masiva está resuelta.** Antes v2 disparaba
  15-25 especies simultáneas por clip con confianza >95% (ver
  `INFORME_FALLAS_V2.md`). Eso no volvió a pasar en ninguna prueba de
  esta sesión.
- **Great Kiskadee (Pitangus sulphuratus) se reconoce correctamente y de
  forma limpia.** Con audio real de Kiskadee reproducido en el sitio de
  despliegue: 16 detecciones en producción, confianza 73%-96%, **sin
  ninguna especie espuria acompañante**, correctamente registradas en la
  base de datos local y subidas a BirdWeather (POST 201 confirmado).
- **El pipeline de producción está sano**: procesa a ritmo real (~1s de
  análisis por clip de 15s), sin acumular cola, sin el consumo de CPU/red
  desmedido que causaba la v2 rota original.
- La integración con BirdNET-Pi (modo Append, activación sigmoid, parseo
  de nombre científico/común desde las etiquetas combinadas
  `SciName_CommonName`) funciona correctamente — no hay ningún bug de
  integración de por medio en nada de lo que sigue.

## Lo que NO funciona: confusión Hornero/Kestrel

### Metodología

Se reprodujeron audios de campo reales de Hornero (los mismos que
antes se confundían con Kestrel) cerca del micrófono de tector2,
capturando los archivos `.wav` completos de 15s tal cual los graba
`birdnet_recording.service` (no un harness aislado). Se corrió el modelo
actualmente instalado en producción (`b9d5e77`) directamente sobre cada
clip, calculando el sigmoide con la misma fórmula y `SENSITIVITY` que usa
BirdNET-Pi en producción (`1/(1+exp(-0.75*logit))`, `SENSITIVITY=1.25` en
`birdnet.conf`), por segmento de 3s, comparando específicamente el score
de `Furnarius rufus_Rufous Hornero` contra `Falco sparverius_American
Kestrel` (las dos clases del bloque de 193 especies locales).

### Resultado, 35 segmentos de 3s (~4 minutos de audio real)

| Métrica | Valor |
|---|---|
| Score promedio Hornero | 0.068 |
| Score promedio Kestrel | **0.138** (el doble que Hornero, siendo Hornero lo que sonaba) |
| Segmentos donde Kestrel > Hornero | **24/35 (68.6%)** |
| Segmentos donde Hornero cruza el umbral de producción (0.7) | **0/35** |
| Segmentos donde Kestrel cruza el umbral (0.7) — falso positivo | **2/35** |

Los dos falsos positivos de Kestrel:

| Archivo | Segmento | Score Kestrel | Score Hornero (mismo segmento) |
|---|---|---|---|
| `2026-08-21-birdnet-11:37:52.wav` | 0-3s | **0.7899** | 0.0012 |
| `2026-08-21-birdnet-11:40:37.wav` | 12-15s | **0.9239** | 0.0074 |

Tabla completa de los 35 segmentos (score Hornero vs Kestrel, quién gana en cada uno):

```
archivo                          seg   hornero   kestrel       gana
2026-08-21-birdnet-11:37:52.wav     0    0.0012    0.7899    kestrel [K>=0.7]
2026-08-21-birdnet-11:37:52.wav     3    0.0060    0.3065    kestrel
2026-08-21-birdnet-11:37:52.wav     6    0.0046    0.0280    kestrel
2026-08-21-birdnet-11:37:52.wav     9    0.0109    0.0478    kestrel
2026-08-21-birdnet-11:37:52.wav    12    0.0247    0.2566    kestrel
2026-08-21-birdnet-11:40:37.wav     0    0.3134    0.1224    HORNERO
2026-08-21-birdnet-11:40:37.wav     3    0.0049    0.0040    HORNERO
2026-08-21-birdnet-11:40:37.wav     6    0.1002    0.0920    HORNERO
2026-08-21-birdnet-11:40:37.wav     9    0.0291    0.0869    kestrel
2026-08-21-birdnet-11:40:37.wav    12    0.0074    0.9239    kestrel [K>=0.7]
2026-08-21-birdnet-11:40:52.wav     0    0.1386    0.0428    HORNERO
2026-08-21-birdnet-11:40:52.wav     3    0.0279    0.1535    kestrel
2026-08-21-birdnet-11:40:52.wav     6    0.3717    0.0156    HORNERO
2026-08-21-birdnet-11:40:52.wav     9    0.0233    0.1586    kestrel
2026-08-21-birdnet-11:40:52.wav    12    0.0332    0.4384    kestrel
2026-08-21-birdnet-11:41:07.wav     0    0.0569    0.1006    kestrel
2026-08-21-birdnet-11:41:07.wav     3    0.0543    0.0219    HORNERO
2026-08-21-birdnet-11:41:07.wav     6    0.0004    0.0057    kestrel
2026-08-21-birdnet-11:41:07.wav     9    0.2618    0.0176    HORNERO
2026-08-21-birdnet-11:41:07.wav    12    0.0006    0.0326    kestrel
2026-08-21-birdnet-11:41:22.wav     0    0.1719    0.0151    HORNERO
2026-08-21-birdnet-11:41:22.wav     3    0.0002    0.1327    kestrel
2026-08-21-birdnet-11:41:22.wav     6    0.0003    0.0068    kestrel
2026-08-21-birdnet-11:41:22.wav     9    0.0010    0.0144    kestrel
2026-08-21-birdnet-11:41:22.wav    12    0.0005    0.0058    kestrel
2026-08-21-birdnet-11:41:37.wav     0    0.0021    0.3514    kestrel
2026-08-21-birdnet-11:41:37.wav     3    0.1686    0.0331    HORNERO
2026-08-21-birdnet-11:41:37.wav     6    0.0002    0.0435    kestrel
2026-08-21-birdnet-11:41:37.wav     9    0.1765    0.1630    HORNERO
2026-08-21-birdnet-11:41:37.wav    12    0.0003    0.0114    kestrel
2026-08-21-birdnet-11:41:52.wav     0    0.1837    0.0254    HORNERO
2026-08-21-birdnet-11:41:52.wav     3    0.0003    0.0142    kestrel
2026-08-21-birdnet-11:41:52.wav     6    0.0151    0.1258    kestrel
2026-08-21-birdnet-11:41:52.wav     9    0.1816    0.2374    kestrel
2026-08-21-birdnet-11:41:52.wav    12    0.0002    0.0055    kestrel
```

(Nota: hay chunks con Hornero real sonando donde el score de Hornero
sube por encima del ruido de fondo — hasta 0.37 — lo cual indica que el
modelo sí capta *algo* de señal real de la especie. El problema es que
ni una sola vez llega al umbral de producción, y en la mayoría de los
segmentos Kestrel puntúa más alto que la especie que realmente estaba
sonando.)

### Por qué esto es grave y no un detalle menor

1. En producción real, este resultado significa: el dispositivo **nunca
   va a reportar un Hornero** (0/35 cruces de umbral en casi 4 minutos de
   audio real e inequívoco de la especie), pero sí va a reportar
   **falsos positivos confiados de Kestrel** con la misma frecuencia con
   la que canta un Hornero cerca del micrófono — una especie que
   probablemente ni siquiera está presente en el sitio de despliegue.
2. Esto contradice directamente las métricas de validación reportadas en
   `INFORME_FALLAS_V2.md` (87.3% accuracy, 85.4% recall por segmento,
   medidas con la metodología real de producción). O el par
   Hornero/Kestrel está subrepresentado en el set de validación (y el
   promedio macro sobre 193 especies lo esconde), o hay embeddings
   compartidos entre estas dos especies puntuales que la re-calibración
   por clase (193 clasificadores binarios independientes) no llegó a
   separar.
3. Esto no fue visto durante el diagnóstico original ni durante la
   validación de la corrección — es un hallazgo posterior, en la primera
   prueba real de campo después de instalar v2 corregido. Vale la pena
   revisar si hay otros pares de especies con el mismo patrón que
   simplemente no se probaron todavía.

## Qué se necesita para investigar esto

1. Revisar específicamente el desempeño de estas dos clases en el set de
   validación (no el promedio macro): precision/recall/confusion matrix
   para `Furnarius rufus` vs `Falco sparverius` en particular.
2. Revisar cuántos ejemplos de entrenamiento tuvo cada una — si Hornero
   tuvo pocos ejemplos reales (o de mala calidad/grabados en condiciones
   muy distintas a esta prueba), el clasificador binario "Hornero vs
   todo" pudo haber aprendido un límite de decisión débil,
   independientemente de que ya no comparta escala con las demás 192
   clases.
3. Si el embedding compartido (extractor de características original de
   BirdNET, sin modificar) efectivamente no separa bien estas dos
   especies, revisar si hace falta más datos negativos específicos de
   Kestrel en el entrenamiento de la clase Hornero (hard negative
   mining), no solo negativos genéricos.
4. Antes de dar por buena cualquier futura versión, probar puntualmente
   con audio real de los pares de especies que ya se sabe que fueron
   problemáticos (este informe deja el método y el script listos para
   repetir la prueba).
