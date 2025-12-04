import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

def detectar_changepoint_scale(df):
    y = df['y'].values
    media = np.mean(y)
    std = np.std(y)

    # Desviación estándar relativa
    std_rel = std / media

    # Cambios mes a mes
    cambios = np.abs(np.diff(y) / y[:-1])
    cambios_prom = np.mean(cambios)

    # Rango relativo
    rango_rel = (np.max(y) - np.min(y)) / media

    print("\n=== ANÁLISIS DE VARIABILIDAD ===")
    print(f"Desviación relativa: {std_rel:.3f}")
    print(f"Promedio de cambios mensuales: {cambios_prom:.3f}")
    print(f"Rango relativo: {rango_rel:.3f}")

    # Decisión automática
    if std_rel < 0.10 and cambios_prom < 0.10:
        print("Serie MUY SUAVE → usando CPS = 0.01")
        return 0.01

    if std_rel < 0.25:
        print("Serie NORMAL → usando CPS = 0.05")
        return 0.05

    if std_rel < 0.50:
        print("Serie VARIABLE → usando CPS = 0.1")
        return 0.1

    print("Serie MUY VARIABLE → usando CPS = 0.2")
    return 0.2

#Cargar datos
df = pd.read_csv('data.csv', sep=';', decimal=',')
df = df[['ds', 'y']]

print("=== DIAGNÓSTICO INICIAL ===")
print("Primeras filas (ORIGINAL):")
print(df.head())
print("\nÚltimas filas (ORIGINAL):")
print(df.tail())

#Estadisticas
print("\n Estadisticas de consumo 'y':")
print(f"Minimo: {df['y'].min()}")
print(f"Maximo: {df['y'].max()}")
print(f"Media: {df['y'].mean()}")

#Datos historicos
plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'], marker='o', linewidth=2, markersize=6, color='blue')
plt.title("Consumo Mensual Historico", fontsize=14, fontweight='bold')
plt.xlabel("Fecha")
plt.ylabel("Consumo")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Crear el modelo
#growth tiene dos opciones: 'linear' y 'logistic'. Linear si es que no hay limites y logistic si hay limites.

def evaluar_modelo(df, cps, sps, growth_type):
    """
    Entrena un Prophet, calcula MAPE en los últimos 5 meses.
    Retorna MAPE y el modelo entrenado.
    """

    try:
        m = Prophet(
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            growth=growth_type
        )

        # Entrenar
        m.fit(df)

        # Crear forecast SOLO del histórico
        future = m.make_future_dataframe(periods=0, freq='ME')
        if 'cap' in df.columns:
            future['cap'] = df['cap'].max()
        forecast = m.predict(future)

        # Comparación real vs yhat
        real = df['y'].tail(5).values
        pred = forecast['yhat'].tail(5).values

        mape = np.mean(np.abs((real - pred) / real)) * 100

        return mape, m

    except Exception as e:
        print(f"⚠️ Error con cps={cps}, sps={sps}, growth={growth_type}: {e}")
        return np.inf, None


# === CANDIDATOS AUTOMÁTICOS ===

cps_auto = detectar_changepoint_scale(df)

cps_list = [0.005, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
sps_list = [1, 2, 5, 10, 15, 20]
growth_list = ['linear', 'logistic']

# Reglas para logistic (máximo histórico × 1.3)
cap_max = df['y'].max() * 1.3
df_logistic = df.copy()
df_logistic['cap'] = cap_max

mejor_mape = np.inf
mejor_modelo = None
mejores_params = None

print("\n=== ENTRENANDO MODELOS AUTOMÁTICOS ===")

for cps in cps_list:
    for sps in sps_list:
        for growth in growth_list:

            df_model = df_logistic if growth == 'logistic' else df

            mape, modelo = evaluar_modelo(df_model, cps, sps, growth)

            print(f"Modelo cps={cps}, sps={sps}, growth={growth} → MAPE={mape:.2f}%")

            if mape < mejor_mape:
                mejor_mape = mape
                mejor_modelo = modelo
                mejores_params = (cps, sps, growth)

print("\n=== MEJOR MODELO ENCONTRADO ===")
print(f"Changepoint Scale: {mejores_params[0]}")
print(f"Seasonality Scale: {mejores_params[1]}")
print(f"Crecimiento: {mejores_params[2]}")
print(f"MEJOR MAPE: {mejor_mape:.2f}%")

m = mejor_modelo

# freq='M': Significa que cada período debe ser de un mes (Month End).
#Creará solo los registros necesarios
future = m.make_future_dataframe(periods=1, freq='ME')
if 'cap' in df.columns:
    future['cap'] = df['cap'].max()
forecast = m.predict(future)
#Para predecir valores dentro de los campos previamente creados
forecast = m.predict(future)
print("\n PREDICCIÓN DEL PROXIMO MES:")
next_month = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1).copy()

print(f"Fecha de predicción: {next_month['ds'].dt.strftime('%Y-%m-%d').values[0]}")
print(f"Predicción: {next_month['yhat'].values[0]:.2f}")
print(f"Intervalo inferior: {next_month['yhat_lower'].values[0]:.2f}")
print(f"Intervalo superior: {next_month['yhat_upper'].values[0]:.2f}")

#Plot graficos
fig1 = m.plot(forecast)
plt.title("Predicción del Consumo Mensual", fontsize=14, fontweight='bold')
plt.xlabel("Fecha")
plt.ylabel("Consumo")
plt.grid(True, alpha=0.3)
plt.show()

#Plot componentes
fig2 = m.plot_components(forecast)
plt.tight_layout()
plt.show()

#Resumen de tendencia
primer_valor = df['y'].iloc[0]
ultimo_valor = df['y'].iloc[-1]

tendencia = "CRECIENTE" if ultimo_valor > primer_valor else "DECRECIENTE"
crecimiento = ultimo_valor - primer_valor

print(f"Primer valor: {primer_valor:.2f} (fecha: {df['ds'].iloc[0]})")
print(f"Último valor: {ultimo_valor:.2f} (fecha: {df['ds'].iloc[-1]})")
print(f"Tendencia general: {tendencia}")
print(f"Crecimiento total: {crecimiento:.2f}")
print(f"Predicción para el proximo mes: {max(next_month['yhat'].values[0], 0):.2f}")

# Comparación últimas predicciones vs reales
print("\n🔍 COMPARACIÓN ENTRE REALES Y PREDICCIÓN:")

comparison = forecast[['ds', 'yhat']].tail(6).copy()   # últimos 6 del forecast
comparison = comparison.head(5)                        # solo los 5 reales conocidos
comparison['y_real'] = df['y'].tail(5).values          # últimos 5 valores reales

print(comparison[['ds', 'y_real', 'yhat']].round(2))

#Porcentajes de error

comparison['error_abs'] = abs(comparison['y_real'] - comparison['yhat'])
comparison['error_pct'] = comparison['error_abs'] / comparison['y_real'] * 100

# MAPE = promedio de los errores porcentuales
mape = comparison['error_pct'].mean()

print("Porcentaje de acierto:" f" {100 - mape:.2f}%")
