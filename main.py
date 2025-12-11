import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import numpy as np

def detectar_changepoint_scale(df):
    """
    Combina tu nuevo enfoque (R²) con el antiguo (CV)
    para mayor robustez.
    """
    y = df['y'].values
    
    # MÉTRICA 1: Tendencia lineal (tu contribución)
    x = np.arange(len(y)).reshape(-1, 1)
    modelo = LinearRegression().fit(x, y)
    r2 = modelo.score(x, y)
    
    # MÉTRICA 2: Variabilidad (contribución original)
    cv = np.std(y) / np.mean(y)
    
    # MÉTRICA 3: Cambios relativos
    cambios = np.abs(np.diff(y) / (y[:-1] + 1e-10))
    cambios_prom = np.median(cambios)  # Mediana más robusta
    
    print("\n=== ANÁLISIS HÍBRIDO ===")
    print(f"Tendencia (R²): {r2:.3f}")
    print(f"Variabilidad (CV): {cv:.3f}")
    print(f"Cambio mensual mediano: {cambios_prom:.3f}")
    
    # PUNTUACIÓN COMBINADA
    # Da peso a ambas métricas
    score_tendencia = r2 * 0.7  # Peso 70% a tendencia
    score_variabilidad = max(0, 1 - cv) * 0.3  # Peso 30% a estabilidad
    
    score_total = score_tendencia + score_variabilidad
    
    print(f"Puntuación combinada: {score_total:.3f}")
    
    # DECISIÓN BASADA EN PUNTUACIÓN
    if score_total > 0.8:
        print("Serie ALTAMENTE PREDECIBLE (CPS = 0.01)")
        return 0.01
    elif score_total > 0.6:
        print("Serie PREDECIBLE (CPS = 0.03)")
        return 0.03
    elif score_total > 0.4:
        print("Serie MODERADA (CPS = 0.05)")
        return 0.05
    elif score_total > 0.2:
        print("Serie VARIABLE (CPS = 0.1)")
        return 0.1
    else:
        print("Serie MUY VARIABLE (CPS = 0.2)")
        return 0.2

#Cargar datos
df = pd.read_csv('data3.csv', sep=';', decimal=',')
df['ds'] = pd.to_datetime(df['ds'])
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

# ======================================================
# MODELO ALTERNATIVO PARA SERIES CON POCOS DATOS
# ======================================================
if len(df) < 20:  
    print("\nDataset pequeño (<20 filas). Usando modelo alternativo de regresión lineal.\n")

    y = df["y"].values
    X = np.arange(len(y)).reshape(-1, 1)

    # Modelo lineal (sin overfitting)
    modelo_lr = LinearRegression()
    modelo_lr.fit(X, y)

    # Predicción del próximo mes
    next_index = np.array([[len(y)]])
    pred = modelo_lr.predict(next_index)[0]

    # Leave-One-Out MAPE
    errores = []
    for i in range(1, len(y)):
        X_train = np.arange(i).reshape(-1, 1)
        y_train = y[:i]
        X_test = np.array([[i]])
        y_test = y[i]

        modelo_temp = LinearRegression()
        modelo_temp.fit(X_train, y_train)
        pred_i = modelo_temp.predict(X_test)[0]

        errores.append(abs((y_test - pred_i) / y_test) * 100)

    mape_lr = np.mean(errores)
    print(f"Predicción del próximo mes: {pred:.2f}")
    print(f"Porcentaje de acierto estimado: {100 - mape_lr:.2f}%")

    # ======================================================
    # GRAFICAR RESULTADOS
    # ======================================================
    plt.figure(figsize=(10, 6))
    plt.plot(df["ds"], df["y"], marker='o', label="Datos reales", linewidth=2)
    plt.plot(df["ds"].tolist() + [df["ds"].iloc[-1] + pd.DateOffset(months=1)],
             modelo_lr.predict(np.arange(len(y)+1).reshape(-1, 1)),
             linestyle='--', marker='o', label="Tendencia modelo LR")

    plt.title("Modelo Lineal — Pronóstico con pocos datos", fontsize=14)
    plt.xlabel("Fecha")
    plt.ylabel("Consumo")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ======================================================
    # 📎 INTERVALO DE CONFIANZA SIMPLE
    # ======================================================
    residuals = y - modelo_lr.predict(X)
    std_err = np.std(residuals)

    lower = pred - 1.96 * std_err
    upper = pred + 1.96 * std_err

    print("\n🔎 Intervalo de confianza del pronóstico:")
    print(f"  y_hat: {pred:.2f}")
    print(f"  Lower: {lower:.2f}")
    print(f"  Upper: {upper:.2f}")

    # DETENER PARA NO USAR PROPHET
    quit()


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
cap_max = np.percentile(df['y'], 95) * 1.2
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
