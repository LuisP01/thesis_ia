import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import sys
from sklearn.metrics import mean_absolute_error

def analizar_serie(df):
    """
    Análisis básico de la serie temporal para justificación metodológica.
    Retorna características que ayudan a elegir el modelo.
    """
    y = df['y'].values
    
    # 1. Tendencia (R² de regresión lineal)
    X = np.arange(len(y)).reshape(-1, 1)
    modelo_lineal = LinearRegression().fit(X, y)
    r2 = modelo_lineal.score(X, y)
    
    # 2. Variabilidad (Coeficiente de Variación)
    cv = np.std(y) / np.mean(y) if np.mean(y) > 0 else 1
    
    # 3. Cambios mensuales medianos
    cambios = np.abs(np.diff(y) / (y[:-1] + 1e-10))
    cambio_mediano = np.median(cambios)
    
    # 4. Estacionalidad aproximada (diferencias año a año)
    estacionalidad = 0
    if len(y) >= 13:
        # Comparar cada mes con el mismo mes del año anterior
        difs_anuales = []
        for i in range(12, len(y)):
            if y[i-12] > 0:
                dif_rel = abs(y[i] - y[i-12]) / y[i-12]
                difs_anuales.append(dif_rel)
        if difs_anuales:
            estacionalidad = 1 - np.median(difs_anuales)
    
    print("\n📊 ANÁLISIS DE LA SERIE TEMPORAL:")
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

def walk_forward_validation(df, modelo_tipo='prophet', parametros=None, verbose=False):

    """
    Validación temporal walk-forward para series temporales.
    Solo usa datos pasados para predecir futuros.
    """
    errores = []
    predicciones = []
    reales = []
    
    # Mínimo 12 meses para entrenar, dejar 12 para validar
    n_total = len(df)
    if n_total < 24:
        print("    Pocos datos para validación walk-forward completa")
        return np.inf, ([], [])
    
    # Para cada punto desde el mes 24 hasta el final
    for i in range(12, n_total - 1):
        # Entrenar con datos hasta i
        df_train = df.iloc[:i+1].copy()
        
        if modelo_tipo == 'prophet':
            # Configurar Prophet con parámetros
            if parametros is None:
                parametros = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10}
            
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=parametros.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=parametros.get('seasonality_prior_scale', 10),
                interval_width=0.95
            )
            m.fit(df_train)
            
            # Predecir próximo mes
            future = m.make_future_dataframe(periods=1, freq='ME')
            forecast = m.predict(future)
            pred = forecast.iloc[-1]['yhat']
            
        elif modelo_tipo == 'lineal':
            # Regresión lineal
            X_train = np.arange(len(df_train)).reshape(-1, 1)
            y_train = df_train['y'].values
            
            modelo = LinearRegression()
            modelo.fit(X_train, y_train)
            
            # Predecir próximo mes
            pred = modelo.predict([[len(df_train)]])[0]
        
        # Valor real (siguiente mes)
        real = df.iloc[i+1]['y']
        
        # Calcular error (protegido contra divisiones por cero)
        error_rel = abs(real - pred) / max(abs(real), 1e-6)

        if verbose:
            train_start = df_train['ds'].iloc[0].strftime('%Y-%m')
            train_end = df_train['ds'].iloc[-1].strftime('%Y-%m')
            test_date = df.iloc[i+1]['ds'].strftime('%Y-%m')
            
            print(
                f"🧪 [{modelo_tipo.upper()}] "
                f"Train: {train_start} → {train_end} | "
                f"Test: {test_date} | "
                f"Real: {real:.2f} | Pred: {pred:.2f} | "
                f"Error: {error_rel*100:.2f}%"
            )

        
        errores.append(error_rel)
        predicciones.append(pred)
        reales.append(real)
    
    mape = np.mean(errores) * 100 if errores else np.inf
    
    return mape, (reales, predicciones)

def evaluar_modelos_prophet(df, caracteristicas):
    """
    Evalúa diferentes configuraciones de Prophet basadas en las características.
    Versión simplificada para tesis.
    """
    resultados = []
    
    # Configuraciones basadas en el análisis
    if caracteristicas['r2'] > 0.7:
        # Serie con buena tendencia
        cps_list = [0.01, 0.05]
    elif caracteristicas['cv'] > 0.5:
        # Serie muy variable
        cps_list = [0.1, 0.2]
    else:
        # Serie moderada
        cps_list = [0.05, 0.1]
    
    # Seasonality scale basado en estacionalidad
    if caracteristicas['estacionalidad'] > 0.3:
        sps_list = [5, 10]
    else:
        sps_list = [10, 20]
    
    print(f"\n🔍 Evaluando Prophet: {len(cps_list)}×{len(sps_list)} = {len(cps_list)*len(sps_list)} combinaciones")
    
    for cps in cps_list:
        for sps in sps_list:
            print(f"  • Probando cps={cps}, sps={sps}")
            
            mape, _ = walk_forward_validation(
                df, 
                modelo_tipo='prophet',
                parametros={'changepoint_prior_scale': cps, 'seasonality_prior_scale': sps},
                verbose= True
            )
            
            if mape < np.inf:
                resultados.append({
                    'mape': mape,
                    'cps': cps,
                    'sps': sps
                })
    
    # Ordenar por MAPE
    resultados.sort(key=lambda x: x['mape'])
    
    return resultados

def predecir_proximo_mes(df, mejor_modelo, parametros=None):
    """
    Predice el próximo mes usando el mejor modelo encontrado.
    """
    if mejor_modelo == 'lineal':
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['y'].values
        
        modelo = LinearRegression()
        modelo.fit(X, y)
        
        proximo_idx = len(df)
        pred = modelo.predict([[proximo_idx]])[0]
        
        residuos = y - modelo.predict(X)
        std_residuos = np.std(residuos)
        intervalo = [pred - 2*std_residuos, pred + 2*std_residuos]
        
        return pred, intervalo, modelo
        
    elif mejor_modelo == 'prophet':
        if parametros is None:
            parametros = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10}
        
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=parametros['changepoint_prior_scale'],
            seasonality_prior_scale=parametros['seasonality_prior_scale'],
            interval_width=0.95
        )
        
        m.fit(df)
        
        future = m.make_future_dataframe(periods=1, freq='ME')
        forecast = m.predict(future)
        
        pred = forecast.iloc[-1]['yhat']
        intervalo = [forecast.iloc[-1]['yhat_lower'], forecast.iloc[-1]['yhat_upper']]
        
        return pred, intervalo, m

def main():
    print("=" * 60)
    print("SISTEMA DE PREDICCIÓN - VERSIÓN TESIS OPTIMIZADA")
    print("=" * 60)
    
    cedula = input("Ingrese el número de cédula: ").strip()
    if not cedula:
        print("❌ Error: La cédula no puede estar vacía")
        sys.exit(1)
    
    tipo = input("Ingrese el tipo de servicio (agua/luz): ").strip().lower()
    if tipo not in ['agua', 'luz']:
        print("❌ Error: El tipo debe ser 'agua' o 'luz'")
        sys.exit(1)
    
    nombre_archivo = f"{cedula}_{tipo}.csv"
    
    if not os.path.exists(nombre_archivo):
        print(f"❌ Error: No se encontró el archivo '{nombre_archivo}'")
        sys.exit(1)
    
    try:
        df = pd.read_csv(nombre_archivo, sep=';', decimal=',')
        df['ds'] = pd.to_datetime(df['ds'])
        
        df['ds'] = df['ds'].dt.to_period('M').dt.to_timestamp('M')
        
        df = df[['ds', 'y']].sort_values('ds').reset_index(drop=True)
        
        print(f"\n✅ Datos cargados: {len(df)} meses")
        print(f"📅 Desde: {df['ds'].iloc[0].strftime('%Y-%m')}")
        print(f"📅 Hasta: {df['ds'].iloc[-1].strftime('%Y-%m')}")
        
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("ANÁLISIS EXPLORATORIO")
    print("=" * 60)
    
    caracteristicas = analizar_serie(df)
    
    print("\n" + "=" * 60)
    print("EVALUACIÓN DE MODELOS (WALK-FORWARD VALIDATION)")
    print("=" * 60)
    
    print("\n📈 Evaluando Regresión Lineal...")
    mape_lineal, resultados_lineal = walk_forward_validation(df, modelo_tipo='lineal')
    
    mape_prophet = np.inf
    mejores_parametros_prophet = None
    resultados_prophet = []
    
    if caracteristicas['n_meses'] >= 24:
        print("\n📈 Evaluando Prophet...")
        resultados_prophet = evaluar_modelos_prophet(df, caracteristicas)
        
        if resultados_prophet:
            mape_prophet = resultados_prophet[0]['mape']
            mejores_parametros_prophet = {
                'changepoint_prior_scale': resultados_prophet[0]['cps'],
                'seasonality_prior_scale': resultados_prophet[0]['sps']
            }
    else:
        print(f"\n  Datos insuficientes para Prophet (se requieren ≥24 meses, hay {caracteristicas['n_meses']})!!!")
    
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 60)
    
    print(f"\n📊 RESULTADOS DE VALIDACIÓN:")
    print(f"   • Regresión Lineal: MAPE = {mape_lineal:.2f}%")
    
    if mape_prophet < np.inf:
        print(f"   • Prophet (mejor configuración): MAPE = {mape_prophet:.2f}%")
    
    if mape_prophet < mape_lineal:
        mejor_modelo = 'prophet'
        diferencia = mape_lineal - mape_prophet
        print(f"\n✅ MEJOR MODELO: Prophet (mejor por {diferencia:.2f}% MAPE)")
        print(f"   • Parámetros óptimos: CPS={mejores_parametros_prophet['changepoint_prior_scale']}, "
              f"SPS={mejores_parametros_prophet['seasonality_prior_scale']}")
    else:
        mejor_modelo = 'lineal'
        if mape_prophet < np.inf:
            diferencia = mape_prophet - mape_lineal
            print(f"\n✅ MEJOR MODELO: Regresión Lineal (mejor por {diferencia:.2f}% MAPE)")
        else:
            print(f"\n✅ MEJOR MODELO: Regresión Lineal (único modelo evaluable)")
    
    print(f"\n🎓 JUSTIFICACIÓN ACADÉMICA:")
    
    if mejor_modelo == 'lineal':
        if caracteristicas['r2'] > 0.7:
            print(f"   • Serie con fuerte tendencia lineal (R²={caracteristicas['r2']:.3f})")
        if caracteristicas['estacionalidad'] < 0.2:
            print(f"   • Baja estacionalidad detectada ({caracteristicas['estacionalidad']:.3f})")
        if caracteristicas['n_meses'] < 24:
            print(f"   • Datos insuficientes para modelos complejos (n={caracteristicas['n_meses']})")
        print(f"   • Modelo parsimonioso más robusto para series cortas/lineales")
    
    elif mejor_modelo == 'prophet':
        print(f"   • Prophet captura mejor la estacionalidad ({caracteristicas['estacionalidad']:.3f})")
        print(f"   • Suficientes datos para modelo complejo (n={caracteristicas['n_meses']})")
        print(f"   • Intervalos de confianza probabilísticos")
    
    print("\n" + "=" * 60)
    print("PREDICCIÓN DEL PRÓXIMO MES")
    print("=" * 60)
    
    if mejor_modelo == 'prophet' and mejores_parametros_prophet:
        pred, intervalo, modelo_obj = predecir_proximo_mes(
            df, mejor_modelo, mejores_parametros_prophet
        )
    else:
        pred, intervalo, modelo_obj = predecir_proximo_mes(df, mejor_modelo)
    
    ultima_fecha = df['ds'].iloc[-1]
    if pd.isna(ultima_fecha):
        proxima_fecha = "Desconocida"
    else:
        ultima_fecha = pd.Timestamp(ultima_fecha)
        proxima_fecha = (ultima_fecha + pd.DateOffset(months=1)).strftime('%Y-%m')
    
    print(f"\nPróximo mes a predecir: {proxima_fecha}")
    print(f"Valor predicho: {pred:.2f}")
    print(f"Intervalo de confianza 95%: [{intervalo[0]:.2f}, {intervalo[1]:.2f}]")
    print(f"Modelo utilizado: {mejor_modelo.upper()}")
    
    print("\n📈 Generando gráfico de resultados...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['ds'], df['y'], 'b-', label='Datos históricos', linewidth=2)
    
    if mejor_modelo == 'prophet':
        fig2 = modelo_obj.plot(modelo_obj.predict(
            modelo_obj.make_future_dataframe(periods=1, freq='ME')
        ))
        plt.title(f"Predicción Prophet - {tipo.capitalize()} (Cédula: {cedula})", fontsize=14)
        plt.xlabel("Fecha")
        plt.ylabel("Consumo")
        plt.grid(True, alpha=0.3)
    else:
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['y'].values
        
        ax.plot(df['ds'], modelo_obj.predict(X), 'r--', label='Tendencia lineal', linewidth=2)
        
        ax.axvline(x=pd.Timestamp(proxima_fecha + '-01'), color='g', linestyle=':', 
                  label='Mes a predecir', linewidth=2)
        ax.plot(pd.Timestamp(proxima_fecha + '-01'), pred, 'go', markersize=10, 
               label=f'Predicción: {pred:.1f}')
        
        ax.fill_between([pd.Timestamp(proxima_fecha + '-01')], 
                       intervalo[0], intervalo[1], 
                       color='green', alpha=0.2, label='Intervalo 95%')
        
        ax.set_title(f"Predicción Lineal - {tipo.capitalize()} (Cédula: {cedula})", fontsize=14)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Consumo")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("RESUMEN PARA DOCUMENTACIÓN DE TESIS")
    print("=" * 60)
    
    print(f"\nMETODOLOGÍA APLICADA:")
    print(f"   1. Análisis exploratorio de la serie temporal")
    print(f"   2. Validación walk-forward (temporalmente correcta)")
    print(f"   3. Comparación de dos modelos: Lineal vs Prophet")
    print(f"   4. Selección basada en MAPE y características de la serie")
    print(f"   5. Predicción del próximo mes con intervalo de confianza")
    
    print(f"\nRESULTADOS OBTENIDOS:")
    print(f"   • Modelo seleccionado: {mejor_modelo.upper()}")
    print(f"   • MAPE modelo lineal: {mape_lineal:.2f}%")
    if mape_prophet < np.inf:
        print(f"   • MAPE mejor Prophet: {mape_prophet:.2f}%")
    print(f"   • Predicción próximo mes: {pred:.2f}")
    print(f"   • Intervalo 95%: [{intervalo[0]:.2f}, {intervalo[1]:.2f}]")
    
    print(f"\nCONCLUSIONES:")
    print(f"   • {'Modelo simple (lineal) suficiente para esta serie' if mejor_modelo == 'lineal' else 'Modelo complejo (Prophet) justificado por características estacionales'}")
    print(f"   • Metodología evita sobreajuste mediante validación temporal")
    print(f"   • Resultados reproducibles y justificables académicamente")
    
    print("\n" + "=" * 60)
    print(f"✅ PROCESO COMPLETADO - Cédula: {cedula}, Tipo: {tipo}")
    print("=" * 60)

if __name__ == "__main__":
    main()