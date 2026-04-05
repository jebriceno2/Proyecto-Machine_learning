# Proyecto Final - Competencia de Clasificacion por Decada

Implementacion inicial para la parte 1 del proyecto de `Aprendizaje de Maquina 2026-10`.

## Objetivo

Construir un clasificador de texto con `scikit-learn` que:

- use `train.csv` para entrenamiento y validacion local,
- use `eval.csv` solo para prediccion final,
- genere archivos de submission con formato `id,answer`,
- guarde el mejor modelo con `joblib`,
- deje listas al menos 5 variantes de envio para Kaggle.

## Estructura

- `data/train.csv`: datos etiquetados con columnas `text` y `decade`.
- `data/eval.csv`: datos sin etiqueta con columnas `id` y `text`.
- `data/submissions/`: archivos listos para subir a Kaggle.
- `models/`: modelo final serializado con `joblib`.
- `notebooks/proyecto_final.ipynb`: notebook principal del proyecto.
- `scripts/run_stage1_pipeline.py`: ejecucion reproducible por consola.
- `src/stage1_pipeline.py`: utilidades y modelos clasicos.

## Flujo recomendado

1. Validar candidatos con un split estratificado sobre `train.csv`.
2. Escoger el modelo con mejor desempeno local.
3. Reentrenarlo con todo `train.csv`.
4. Guardarlo en `models/` usando `joblib`.
5. Predecir `eval.csv` y exportar el submission a `data/submissions/`.

## Modelos iniciales

La primera version prioriza modelos clasicos robustos para texto historico con OCR y ortografia variable:

- `TF-IDF` de caracteres + `LinearSVC`
- variantes con `char` y `char_wb`
- una variante combinada de palabras + caracteres

## Ejecucion

Generar solo el mejor modelo y su submission:

```bash
python scripts/run_stage1_pipeline.py
```

Generar cinco submissions distintos para cubrir la participacion minima en Kaggle:

```bash
python scripts/run_stage1_pipeline.py --generate-five-submissions
```

## Notas importantes

- `eval.csv` no se usa para entrenar ni para ajustar hiperparametros.
- La metrica oficial debe confirmarse en la pagina de la competencia de Kaggle; localmente se reportan `accuracy` y `macro_f1`.
- Para Bloque Neon, el entregable final de la parte 1 debe incluir el notebook y el modelo `scikit-learn` guardado con `pickle` o `joblib`.
