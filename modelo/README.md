# Sin modelo instalable por ahora

El clasificador v2 (Append, 193 especies AR-B) tenía una falla de
calibración confirmada en hardware real: dispara docenas de especies
simultáneas con confianza alta en el mismo clip, incluyendo especies
imposibles para la región. Ver [`../INFORME_FALLAS_V2.md`](../INFORME_FALLAS_V2.md)
para el detalle completo, la evidencia, y la hipótesis de causa raíz.

Se sacaron el `.tflite` y el archivo de labels de este repositorio hasta
que exista una versión 3 corregida. Esto es deliberado: `LSD-Tector2.0`
mantiene el modelo actualizado en cada dispositivo de forma automática
(`scripts/actualizar_modelo.sh`), comparando el SHA del último commit de
este repositorio y descargando lo que encuentre en esta carpeta. Mientras
el `.tflite` roto siguiera acá, ese mecanismo lo iba a reinstalar solo, sin
intervención humana, en cualquier dispositivo que hiciera una
actualización (incluido tector2, ya revertido a mano al modelo original de
BirdNET). Con la carpeta vacía, ese script simplemente falla la descarga y
no toca el modelo que ya esté instalado.

Cuando haya una v3 corregida, este archivo se reemplaza junto con el
`.tflite` y las labels nuevas.
