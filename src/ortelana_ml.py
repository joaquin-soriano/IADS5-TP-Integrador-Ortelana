"""Componentes reutilizables del proyecto Ortelana Textil.

Viven en un módulo (y no en el notebook) para que el pipeline serializado con
joblib pueda recargarse desde cualquier script, por ejemplo src/predecir.py.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder

# Columnas crudas que necesita el pipeline productivo. Metros3..Metros12 se
# excluyen a propósito: su presencia/ausencia revela el formato (proxy del target).
COLUMNAS_ENTRADA = ['Sucursal_ID', 'Año', 'Tela_ID', 'Metros1', 'Metros2']


def agregar_features(X):
    """Agrega features derivadas fila a fila (no aprende nada de los datos).

    - log_M1: escala del primer período reportado.
    - log_ratio_M2_M1: log(Metros2 / Metros1), forma del reporte entre períodos.
    Los faltantes se propagan como NaN para que los imputa el pipeline.
    """
    X = pd.DataFrame(X).copy()
    m1 = pd.to_numeric(X['Metros1'], errors='coerce').clip(lower=0)
    m2 = pd.to_numeric(X['Metros2'], errors='coerce').clip(lower=0)
    X['log_M1'] = np.log1p(m1)
    X['log_ratio_M2_M1'] = np.log1p(m2) - np.log1p(m1)
    return X


class ClasificadorEtiquetado(BaseEstimator, ClassifierMixin):
    """Envuelve un clasificador que requiere etiquetas enteras (p. ej. XGBoost)
    para que el pipeline reciba y devuelva las etiquetas originales ('B', 'C', ...)."""

    def __init__(self, estimador):
        self.estimador = estimador

    def fit(self, X, y):
        self.codificador_ = LabelEncoder().fit(y)
        self.classes_ = self.codificador_.classes_
        self.estimador_ = clone(self.estimador).fit(X, self.codificador_.transform(y))
        return self

    def predict(self, X):
        return self.codificador_.inverse_transform(self.estimador_.predict(X))

    def predict_proba(self, X):
        return self.estimador_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimador_.feature_importances_
