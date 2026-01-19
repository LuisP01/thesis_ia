import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def analizar_serie(df):
    """
    Análisis básico de la serie temporal para justificación metodológica.
    Retorna características que ayudan a elegir el modelo.
    """
    y = df['y'].values
    
    X = np.arange(len(y)).reshape(-1, 1)
    modelo_lineal = LinearRegression().fit(X, y)
    r2 = modelo_lineal.score(X, y)
    
    cv = np.std(y) / np.mean(y) if np.mean(y) > 0 else 1
    
    cambios = np.abs(np.diff(y) / (y[:-1] + 1e-10))
    cambio_mediano = np.median(cambios)
    
    estacionalidad = 0
    if len(y) >= 13:
        difs_anuales = []
        for i in range(12, len(y)):
            if y[i-12] > 0:
                dif_rel = abs(y[i] - y[i-12]) / y[i-12]
                difs_anuales.append(dif_rel)
        if difs_anuales:
            estacionalidad = 1 - np.median(difs_anuales)
    
    print("\nANÁLISIS DE LA SERIE TEMPORAL:")
    print(f"   • Tendencia (R²): {r2:.3f}")
    print(f"   • Variabilidad (CV): {cv:.3f}")
    print(f"   • Cambio mensual mediano: {cambio_mediano:.3f}")
    print(f"   • Estacionalidad aproximada: {estacionalidad:.3f}")
    
    return {
        'r2': r2,
        'cv': cv,
        'cambio_mediano': cambio_mediano,
        'estacionalidad': estacionalidad,
        'n_meses': len(y)
    }
