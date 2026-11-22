"""Predice Format_Reporte para registros de Ortelana Textil con el pipeline productivo.

Uso (desde la raíz del proyecto):
    python src/predecir.py --entrada nuevos_registros.csv --salida predicciones.csv

El CSV de entrada debe tener las columnas Sucursal_ID, Año, Tela_ID, Metros1 y Metros2
(columnas extra se ignoran; faltantes en las celdas se imputan dentro del pipeline).
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))  # el pickle referencia src.ortelana_ml

from src.ortelana_ml import COLUMNAS_ENTRADA  # noqa: E402

MODELO_POR_DEFECTO = RAIZ / 'models' / 'modelo_productivo.joblib'


def predecir(df, ruta_modelo=MODELO_POR_DEFECTO):
    faltantes = [c for c in COLUMNAS_ENTRADA if c not in df.columns]
    if faltantes:
        raise ValueError(f'Faltan columnas requeridas: {faltantes}')
    modelo = joblib.load(ruta_modelo)
    X = df[COLUMNAS_ENTRADA]
    salida = df[COLUMNAS_ENTRADA].copy()
    salida['Format_Reporte_pred'] = modelo.predict(X)
    proba = modelo.predict_proba(X)
    for i, clase in enumerate(modelo.classes_):
        salida[f'proba_{clase}'] = proba[:, i].round(4)
    return salida


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--entrada', required=True, help='CSV con los registros a clasificar')
    parser.add_argument('--salida', default='predicciones.csv', help='CSV de salida')
    parser.add_argument('--modelo', default=str(MODELO_POR_DEFECTO), help='ruta del .joblib')
    args = parser.parse_args()

    df = pd.read_csv(args.entrada)
    salida = predecir(df, args.modelo)
    salida.to_csv(args.salida, index=False)
    print(f'{len(salida)} predicciones guardadas en {args.salida}')
    print(salida['Format_Reporte_pred'].value_counts().to_string())


if __name__ == '__main__':
    main()
