# Ortelana Textil — clasificación de `Format_Reporte`

Trabajo práctico integrador de IADS5. Proyecto de aprendizaje automático de punta a punta sobre
`ventas_historicas_ortelana_argentina.csv` (3285 registros sucursal × año × tela, 2016-2025).

## Problema
- **Principal (clasificación multiclase):** predecir `Format_Reporte` (B, C, M, S), es decir con qué formato
  de períodos fue cargado un registro, a partir de `Sucursal_ID`, `Año`, `Tela_ID`, `Metros1` y `Metros2`.
- **Secundario (regresión):** `Metros_Total`, metros vendidos en el año (suma de `Metros1..Metros12`),
  a partir de año, sucursal y tela.

## Decisiones principales
- Se excluyen `Metros3..Metros12`: su presencia o ausencia revela el formato (B tiene 6 períodos, M 12, C y S 2).
- Features derivadas fila a fila: `log_M1` y `log_ratio_M2_M1` = log(Metros2/Metros1).
- `Sucursal_ID` y `Tela_ID` son identificadores: con one-hot el modelo memoriza el formato de cada par
  sucursal-tela (cada par tiene siempre el mismo formato), así que el modelo final **no** los usa.
- Partición única 80/20 estratificada (`random_state=42`), persistida en `data/split_indices.csv`.
- Selección de modelos e hiperparámetros solo con validación cruzada sobre train; test se usó una vez
  para el modelo final (las métricas de test de actividades previas son históricas y no guiaron decisiones).

## Resultados (F1 macro)
| Modelo                                 |   F1_CV10_media |   F1_CV10_std |   F1_CV_grupos |   F1_test_histórico |
|:---------------------------------------|----------------:|--------------:|---------------:|--------------------:|
| Random Forest + IDs one-hot (pipeline) |           0.982 |         0.008 |          0.917 |               0.988 |
| XGBoost (100, lr=0.1, depth=3)         |           0.947 |         0.007 |          0.937 |               0.952 |
| Random Forest (100 árboles)            |           0.943 |         0.008 |          0.934 |               0.943 |
| Árbol gini (max_depth=4)               |           0.939 |         0.007 |          0.931 |               0.936 |
| Árbol entropy (max_depth=4)            |           0.938 |         0.006 |          0.929 |               0.936 |
| k-NN k=7 escalado                      |           0.932 |         0.013 |          0.926 |               0.922 |
| LogReg (Año, log_M1, log_ratio)        |           0.76  |         0.021 |          0.758 |               0.769 |
| Baseline (más frecuente)               |           0.11  |         0     |          0.11  |               0.11  |

**Modelo final:** XGBoost `{'learning_rate': 0.3, 'max_depth': 2, 'n_estimators': 100}` dentro de un `Pipeline`
(features → imputación + escalado → modelo).
- F1 macro CV10 (train): 0.951
- **F1 macro test: 0.949** (IC 95% bootstrap [0.932, 0.965]), accuracy 0.948
- Baseline (clase más frecuente) en test: 0.110

## Estructura
```
ventas_historicas_ortelana_argentina.csv   dataset
IADS5_TP_Integrador_Ortelana_Cascallar-Kanesky-Luciani-Soriano.ipynb         notebook con las 16 actividades
data/split_indices.csv                     partición train/test
models/modelo_productivo.joblib            pipeline final serializado
src/ortelana_ml.py                         features y envoltorio de etiquetas (usados por el pipeline)
src/predecir.py                            script de predicción
outputs/                                   tablas y predicciones generadas
requirements.txt                           dependencias con versiones
```

## Reproducción
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace IADS5_TP_Integrador_Ortelana_Cascallar-Kanesky-Luciani-Soriano.ipynb
python src/predecir.py --entrada outputs/ejemplo_entrada.csv --salida outputs/ejemplo_predicciones.csv
```
El CSV de entrada necesita las columnas ['Sucursal_ID', 'Año', 'Tela_ID', 'Metros1', 'Metros2'].

## Limitaciones
- La partición aleatoria mezcla años y repite pares sucursal-tela entre train y test; el desempeño sobre pares
  nuevos se estimó aparte con validación agrupada (F1 macro 0.938 para el modelo final).
- La confusión principal es entre C y S (ambos formatos de 2 períodos); el significado de cada código no está
  documentado en el dataset.
- No hay validación temporal (entrenar con años viejos y evaluar en años nuevos); es la próxima mejora sugerida.
