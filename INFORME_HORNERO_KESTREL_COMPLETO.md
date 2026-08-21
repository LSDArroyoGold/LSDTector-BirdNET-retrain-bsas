# Informe completo: confusión Furnarius rufus / Falco sparverius

Fecha: 2026-08-21. Documento único que junta todas las pruebas del día sobre
este problema puntual, en orden cronológico, con los datos crudos de cada
una. El problema **no está resuelto** — hay una discrepancia real y sin
explicar entre dos formas de medir, documentada en detalle en la sección
final. No se debe interpretar nada de este informe como "no hay
problema": el problema es real, tiene impacto directo en la fiabilidad
del dispositivo en el campo, y necesita más trabajo para cerrarse.

## 1. Contexto: modelo v2, calibración corregida

El modelo instalado en producción durante la mayor parte de este día fue
`b9d5e77` (ver `INFORME_FALLAS_V2.md`): modo Append, 6522 especies
originales de BirdNET + 193 locales reentrenadas como clasificadores
binarios independientes. Esa corrección resolvió una falla de
calibración masiva previa (docenas de detecciones espurias simultáneas).
La confusión Hornero/Kestrel que sigue este informe apareció **después**
de esa corrección, en la primera prueba de campo real del modelo ya
corregido.

## 2. Primera detección del problema (índice equivocado, corregido después)

La primera ronda de pruebas (35 segmentos de audio real de hornero,
reproducido por parlante) midió por error la neurona del catálogo
**original** de BirdNET (`Furnarius rufus_Rufous Hornero`, con guión
bajo) en vez de la neurona **reentrenada real** (`Furnarius rufus`,
sin guión bajo). Ese error de índice fue identificado y corregido en la
misma sesión (documentado en `INFORME_CONFUSION_HORNERO_KESTREL.md`).

**Con el índice correcto**, sobre las mismas grabaciones de campo
(reproducidas por parlante), usando el modelo v2 en producción
(SENSITIVITY entonces en 1.1, luego subido a distintos valores en
pruebas posteriores):

- Kiskadee (control positivo): 0.60 → 0.998, funciona bien.
- Kestrel: nunca superó 0.09 en esa tanda puntual.
- Hornero: señal real pero débil, máximo ~0.13-0.22, sin cruzar el
  umbral de producción.

## 3. Experimento "FULL": sesgo regional también sobre el catálogo original

Se prototipeó (sin pushear al repo compartido, solo en tector2) una
extensión que aplica el mismo sesgo por frecuencia regional también a
las 6522 especies del catálogo original de BirdNET, no solo a las 193
locales.

**Resultado real, con audio real de hornero reproducido por parlante,
modelo FULL instalado en producción:**

```
Segmento donde SÍ sonaba hornero (3-6s): Kestrel orig=0.8669, Hornero orig=0.9790 (ambos altos)
Segmentos donde NO sonaba nada de esto:
  0-3s:   Kestrel orig=0.9810
  6-9s:   Kestrel orig=0.9893
  9-12s:  Kestrel orig=0.9556
  12-15s: Kestrel orig=0.9265
```

Kestrel del catálogo original se disparó casi constante (87%-99%),
independientemente de si sonaba algo relacionado. Causa identificada:
el sesgo se centraba contra el promedio de las 6522 especies, la
mayoría sin dato real (piso), lo que saturaba a cualquier especie con
dato real (incluida Kestrel) al mismo tope que Hornero.

## 4. Experimento "stock + regional corregido"

Se corrigió el cálculo del sesgo (centrado solo contra las ~470-477
especies con dato real, no contra las 6522) y se generó un modelo
**BirdNET stock puro (sin las 193 locales) + este sesgo corregido**,
alpha=0.4, elegido para dar a Hornero un refuerzo real (+2.01) sin
saturar a Kestrel (+0.87) como en el experimento anterior.

**Resultado real, con 4 clips de audio de campo (20 segmentos), hornero
reproducido por parlante, modelo instalado en producción:**

```
archivo 16:40:15  0-3s:  hornero=0.0004  kestrel=0.0149  (kestrel)
archivo 16:40:15  9-12s: hornero=0.0071  kestrel=0.7425  (kestrel)
archivo 16:40:30  6-9s:  hornero=0.0013  kestrel=0.5112  (kestrel)
archivo 16:41:01  9-12s: hornero=0.0090  kestrel=0.8657  (kestrel)
```

**18 de 20 segmentos ganó Kestrel**, incluyendo valores altos y
aislados (0.74, 0.87, 0.51) sin razón acústica aparente. Con un sesgo
bien calculado y no degenerado, la confusión persistió igual.

## 5. Se descartaron dos hipótesis de "versión distinta"

A pedido explícito de revisar si había una discrepancia de
versión/pipeline entre el entrenamiento/validación y lo que corre en
producción:

- **Checkpoint base**: comparado directamente, MD5 idéntico
  (`b1c981fe261910b473b9b7eec9ebcd4e`) entre el archivo que usa
  `birdnet_analyzer` (entorno de entrenamiento/validación) y el archivo
  committeado en el propio repositorio de BirdNET-Pi (producción).
  Es el mismo archivo, byte por byte.
- **Preprocesamiento de audio**: comparado el código fuente de ambos
  lados. Ambos usan exactamente `librosa.load(path, sr=48000,
  mono=True, res_type='kaiser_fast')`. Idéntico.

Estas dos hipótesis quedan descartadas como explicación.

## 6. Prueba con archivo limpio (Xeno-canto), vía script manual

Se descargó un archivo real de Xeno-canto
(`Furnarius_rufus_-_Rufous_Hornero_XC298518.mp3`, vía Wikimedia
Commons) y se lo pasó directamente al modelo (sin parlante, sin
micrófono), usando un script de inspección manual:

```
0-3s:   hornero=0.9758  kestrel=0.0000
3-6s:   hornero=0.9610  kestrel=0.0000
6-9s:   hornero=0.9890  kestrel=0.0000
9-12s:  hornero=0.9318  kestrel=0.0000
12-15s: hornero=0.9832  kestrel=0.0000
15-18s: hornero=0.9685  kestrel=0.0000
18-21s: hornero=0.9690  kestrel=0.0000
21-24s: hornero=0.8019  kestrel=0.0000
24-27s: hornero=0.9968  kestrel=0.0000
27-30s: hornero=0.1537  kestrel=0.0016
```

9 de 10 segmentos con Hornero >80%, Kestrel exactamente 0.0000 en los
10.

## 7. Prueba con archivo limpio, vía pipeline real de producción (no un script)

Para eliminar cualquier diferencia entre el script de inspección manual
y el código real, se convirtió el mismo archivo al formato exacto de
grabación (WAV 48kHz estéreo 16-bit, mismo patrón de nombre de archivo)
y se lo insertó directamente en `~/BirdSongs/StreamData/`, la carpeta
donde `birdnet_analysis.service` busca grabaciones nuevas — el mismo
código, mismo umbral (`CONFIDENCE=0.6`), sin ningún script intermedio.

**Resultado real de producción (`journalctl -u birdnet_analysis.service`):**

```
17:22:31  Furnarius rufus;Rufous Hornero;0.9759
17:22:34  Furnarius rufus;Rufous Hornero;0.961
17:22:37  Furnarius rufus;Rufous Hornero;0.989
17:22:40  Furnarius rufus;Rufous Hornero;0.9317
17:22:43  Furnarius rufus;Rufous Hornero;0.9832
17:22:32  Furnarius rufus;Rufous Hornero;0.9684
17:22:35  Furnarius rufus;Rufous Hornero;0.969
17:22:38  Furnarius rufus;Rufous Hornero;0.8013
17:22:41  Furnarius rufus;Rufous Hornero;0.9968
```

9 de 9 segmentos correctos, subidos a BirdWeather sin problema (POST
201 x9), cero falsos positivos de Kestrel.

## 8. El hallazgo real y sin resolver: la discrepancia en sí misma

Con el mismo modelo, el mismo código, el mismo umbral, en el mismo
dispositivo:

| Fuente de audio | Resultado |
|---|---|
| Archivo Xeno-canto limpio, directo al pipeline real | 9/9 correcto, Kestrel=0 siempre |
| Grabaciones de campo reproducidas por parlante de celular | Kestrel gana 18-24 de cada ~20-35 segmentos, con valores altos aislados |

**Esto es el hallazgo central, y no está explicado todavía.** Dos
lecturas posibles, ninguna confirmada:

1. El parlante del celular introduce suficiente distorsión/coloración
   de frecuencia como para que el audio reproducido deje de
   representar un hornero real de forma fiel — en cuyo caso el
   problema es de metodología de prueba, no del dispositivo en el
   campo real (donde un hornero canta al aire libre, no a través de un
   parlante).
2. El dispositivo real, en el campo, con un pájaro real cantando a
   cierta distancia del micrófono (no en estudio, con ruido ambiente
   real de por medio), podría comportarse más parecido a la prueba por
   parlante que a la prueba con archivo limpio — en cuyo caso el
   problema es real y grave para el uso previsto del LSD-Tector.

**Ninguna de las dos hipótesis está confirmada.** No hay todavía una
prueba con un hornero real cantando al aire libre, a la distancia y
condiciones típicas de despliegue de campo, sin pasar por un parlante.
Esa es la prueba que falta y que se necesita para cerrar esto de
verdad.

## 9. Qué hace falta para cerrar esto

1. Grabar (o conseguir) audio de un hornero real cantando al aire
   libre, a distancia realista del micrófono del dispositivo, sin
   pasar por ningún parlante intermedio, y correrlo por el mismo
   pipeline real de producción (como se hizo en la Sección 7).
2. Si ese audio real de campo también falla (Kestrel gana), el
   problema es real y confirmado en condiciones de despliegue reales
   — hace falta reentrenar la neurona de Kestrel con negativos duros
   reales de Hornero, con datos de esta clase de condición acústica
   (no solo grabaciones limpias tipo Xeno-canto).
3. Si ese audio real de campo funciona bien (como el archivo Xeno-canto
   limpio), la conclusión sería que el método de prueba por parlante no
   es representativo, y habría que re-evaluar todos los hallazgos de
   "confusión" de hoy bajo esa luz — pero esto todavía no está
   confirmado, es una hipótesis pendiente de prueba.
4. Independientemente de lo anterior, vale la pena revisar la robustez
   del modelo ante audio degradado/con ruido en general (no solo este
   par de especies), ya que el dispositivo va a operar en condiciones
   de campo reales con ruido ambiente, no en condiciones de laboratorio.
