# Scripts y resultados de la validación independiente de v10

Ver [`../INFORME_VALIDACION_INDEPENDIENTE_V10.md`](../INFORME_VALIDACION_INDEPENDIENTE_V10.md)
para el informe completo con el análisis y las conclusiones. Esta carpeta
tiene el material de soporte:

- `scripts/1` a `6`: corridos localmente contra `modelo/LSDTector_Classifier_v2.tflite`,
  sin depender de BirdNET-Pi ni de ningún dispositivo. Requieren un venv de
  Python con `tensorflow`, `librosa`, `numpy`, `resampy`, `soundfile`. Las
  rutas de audio de entrada (`~/Desktop/Tector/Datasets_prueba/Audios confusion`,
  `~/Desktop/Tector/Datasets_prueba/BirdNET_Detecciones`, `~/Desktop/Tector/Datasets_prueba/xc_hornero_ar`,
  `~/Desktop/Tector/Datasets_prueba/xc_kestrel_ar`) son locales de la máquina donde se
  corrió esto, no están en el repo por tamaño — hay que ajustar las rutas
  si se quiere repetir en otra máquina.
- `scripts/7_revisar_manual_merlin.py`: script interactivo para verificar
  clips a mano contra Merlin — reproduce audio (decodifica con
  librosa/soundfile, reproduce con `paplay`), permite repetir, y guarda
  progresivamente la respuesta en un CSV. Se corre apuntándolo a una
  carpeta de mp3s con el patrón de nombre
  `<especie_original>__vs__<especie_v10>__confBNviejo<NN>.mp3`.
- `resultados/registros_test_amplio_37_especies.csv`: salida completa del
  script 4 (382 clips, 25 especies con datos suficientes).
- `resultados/verificacion_manual_merlin.csv`: las 28 respuestas reales del
  usuario verificando con Merlin (columna `especie_merlin`; vacío = Merlin
  no detectó nada; dos especies separadas por `/` = Merlin dio dos
  posibles o dos aves cantando en el mismo clip).
