import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np

# 1. Cargar CSV y limpiar columnas innecesarias
df = pd.read_csv('data_luz.csv', sep=';', decimal=',')
df = df[['ds', 'y']]

print("=== DIAGNÓSTICO INICIAL ===")
print(f"Primeras filas (ORIGINAL):")
print(df.head())
print(f"\nÚltimas filas (ORIGINAL):")
print(df.tail())

# 2. CONVERSIÓN CORRECTA Y ORDENAMIENTO
df['y'] = df['y'].astype(str).str.replace(',', '.').str.replace(' ', '')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df['ds'] = pd.to_datetime(df['ds'], format='%d/%m/%y')

# 3. ORDENAR POR FECHA DE FORMA ASCENDANTE (MÁS IMPORTANTE)
df = df.sort_values('ds', ascending=True).reset_index(drop=True)

print(f"\n=== DESPUÉS DE ORDENAR ===")
print(f"Primeras filas (ORDENADO):")
print(df.head())
print(f"\nÚltimas filas (ORDENADO):")
print(df.tail())

print(f"\nEstadísticas de 'y':")
print(f"  Mínimo: {df['y'].min()}")
print(f"  Máximo: {df['y'].max()}")
print(f"  Promedio: {df['y'].mean():.2f}")

# 4. Graficar datos históricos CORRECTAMENTE ORDENADOS
plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'], marker='o', linewidth=2, markersize=6, color='blue')
plt.title("Consumo Mensual Histórico (Ordenado Correctamente)", fontsize=14, fontweight='bold')
plt.xlabel("Fecha")
plt.ylabel("Consumo")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Crear modelo Prophet con ajustes para series crecientes
m = Prophet(
    growth='linear',
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0
)

# 6. Entrenar modelo
print("\n🔧 Entrenando modelo Prophet...")
m.fit(df)

# 7. Crear dataframe para predecir
future = m.make_future_dataframe(periods=1, freq='ME')

# 8. Hacer predicción
forecast = m.predict(future)

# 9. Mostrar resultados
print("\n📊 PREDICCIÓN DEL PRÓXIMO MES:")
next_month = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1).copy()

print(f"Fecha de predicción: {next_month['ds'].dt.strftime('%Y-%m-%d').values[0]}")
print(f"Predicción: {next_month['yhat'].values[0]:.2f}")
print(f"Intervalo inferior: {next_month['yhat_lower'].values[0]:.2f}")
print(f"Intervalo superior: {next_month['yhat_upper'].values[0]:.2f}")

# 10. Graficar pronóstico completo
fig1 = m.plot(forecast)
plt.title("Predicción del Consumo Mensual", fontsize=14, fontweight='bold')
plt.xlabel("Fecha")
plt.ylabel("Consumo")
plt.grid(True, alpha=0.3)
plt.show()

# 11. Graficar componentes
fig2 = m.plot_components(forecast)
plt.tight_layout()
plt.show()

# 12. Análisis de tendencia CORREGIDO
print(f"\n📊 RESUMEN CORREGIDO:")
print(f"Período de datos: {len(df)} meses")
print(f"Primer valor: {df['y'].iloc[0]:.2f} (fecha: {df['ds'].iloc[0].strftime('%Y-%m')})")
print(f"Último valor: {df['y'].iloc[-1]:.2f} (fecha: {df['ds'].iloc[-1].strftime('%Y-%m')})")

tendencia = "CRECIENTE" if df['y'].iloc[-1] > df['y'].iloc[0] else "DECRECIENTE"
crecimiento = df['y'].iloc[-1] - df['y'].iloc[0]

print(f"Tendencia general: {tendencia}")
print(f"Crecimiento total: {crecimiento:.2f}")
print(f"Predicción para próximo mes: {max(next_month['yhat'].values[0], 0):.2f}")

# 13. Mostrar las últimas predicciones vs reales
print(f"\n🔍 COMPARACIÓN RECIENTE:")
comparison = forecast[['ds', 'yhat']].tail(6).copy()
comparison = comparison.head(5)  # Últimos 5 meses conocidos
comparison['y_real'] = df['y'].tail(5).values

print(comparison[['ds', 'y_real', 'yhat']].round(2))