import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import sys
from sklearn.metrics import mean_absolute_error

from src.models.prophet import evaluar_modelos_prophet, walk_forward_validation
from src.services.forecastService import guardar_forecast
from src.services.payment import calcular_pago
from src.utils.firebaseStorage import descargar_csv_firebase
from src.utils.monthlyForecast import predecir_proximo_mes
from src.utils.seriesAnalytics import analizar_serie


def ejecutar_sistema_completo(id: int, cedula: str, tipo: str, agua_data, luz_data):
    print("=" * 60)
    print("SISTEMA DE PREDICCIÓN - VERSIÓN TESIS OPTIMIZADA")
    print("=" * 60)
    
    ruta_local = descargar_csv_firebase(cedula, tipo)
    if ruta_local is None:
        print(f"Se omite {cedula} ({tipo}) por falta de archivo")
        return

    if tipo == "agua" and not agua_data:
        print(f"Se omite {cedula} (agua) — no tiene datos")
        return

    if tipo == "luz" and not luz_data:
        print(f"Se omite {cedula} (luz) — no tiene datos")
        return


    print("Ruta descargada:", ruta_local)

    if ruta_local and os.path.exists(ruta_local):
        print("Tamaño archivo (bytes):", os.path.getsize(ruta_local))


        
    try:
        df = pd.read_csv(ruta_local, sep=';', decimal=',')
        df['ds'] = pd.to_datetime(df['ds'])
        
        df['ds'] = df['ds'].dt.to_period('M').dt.to_timestamp('M')
        
        df = df[['ds', 'y']].sort_values('ds').reset_index(drop=True)
        
        print(f"\nDatos cargados: {len(df)} meses")
        print(f"Desde: {df['ds'].iloc[0].strftime('%Y-%m')}")
        print(f"Hasta: {df['ds'].iloc[-1].strftime('%Y-%m')}")
        
    except Exception as e:
        raise Exception(f"Error cargando datos {cedula}: {e}")
    
    print("\n" + "=" * 60)
    print("ANÁLISIS EXPLORATORIO")
    print("=" * 60)
    
    caracteristicas = analizar_serie(df)
    
    print("\n" + "=" * 60)
    print("EVALUACIÓN DE MODELOS (WALK-FORWARD VALIDATION)")
    print("=" * 60)
    
    print("\nEvaluando Regresión Lineal...")
    mape_lineal, resultados_lineal = walk_forward_validation(df, modelo_tipo='lineal')
    
    mape_prophet = np.inf
    mejores_parametros_prophet = None
    resultados_prophet = []
    
    if caracteristicas['n_meses'] >= 24:
        print("\nEvaluando Prophet...")
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
    
    print(f"\nRESULTADOS DE VALIDACIÓN:")
    print(f"   • Regresión Lineal: MAPE = {mape_lineal:.2f}%")
    
    if mape_prophet < np.inf:
        print(f"   • Prophet (mejor configuración): MAPE = {mape_prophet:.2f}%")
    
    if mape_prophet < mape_lineal:
        mejor_modelo = 'prophet'
        diferencia = mape_lineal - mape_prophet
        print(f"\nMEJOR MODELO: Prophet (mejor por {diferencia:.2f}% MAPE)")
        print(f"   • Parámetros óptimos: CPS={mejores_parametros_prophet['changepoint_prior_scale']}, "
              f"SPS={mejores_parametros_prophet['seasonality_prior_scale']}")
    else:
        mejor_modelo = 'lineal'
        if mape_prophet < np.inf:
            diferencia = mape_prophet - mape_lineal
            print(f"\nMEJOR MODELO: Regresión Lineal (mejor por {diferencia:.2f}% MAPE)")
        else:
            print(f"\nMEJOR MODELO: Regresión Lineal (único modelo evaluable)")
    
    print(f"\n JUSTIFICACIÓN ACADÉMICA:")
    
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
        
        intervalo_estadistico = intervalo.copy()

        intervalo_logico = [
            max(0, intervalo[0]),
            intervalo[1]
        ]

        print(f"Intervalo estadístico 95%: [{intervalo_estadistico[0]:.2f}, {intervalo_estadistico[1]:.2f}]")
        print(f"Intervalo aplicado al negocio (truncado en 0): [{intervalo_logico[0]:.2f}, {intervalo_logico[1]:.2f}]")

    else:
        pred, intervalo, modelo_obj = predecir_proximo_mes(df, mejor_modelo)
        intervalo_estadistico = intervalo.copy()

        intervalo_logico = [
            max(0, intervalo[0]),
            intervalo[1]
        ]

        print(f"Intervalo estadístico 95%: [{intervalo_estadistico[0]:.2f}, {intervalo_estadistico[1]:.2f}]")
        print(f"Intervalo aplicado al negocio (truncado en 0): [{intervalo_logico[0]:.2f}, {intervalo_logico[1]:.2f}]")
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
    
    mostrar_analisis_precision(df, mape_lineal, mape_prophet, intervalo, pred)
    
    y = df['y'].values
    n_ceros = np.sum(y == 0)

    if mape_prophet < np.inf:
        error_usado = min(mape_lineal, mape_prophet)
    else:
        error_usado = mape_lineal
    
    print(f"Error usado para BD: {error_usado:.2f}%")

    payment = calcular_pago(
        tipo=tipo,
        consumo=pred,
        water_data=agua_data if tipo == "agua" else None,
        electricity_data=luz_data if tipo == "luz" else None
    )

            
    guardar_forecast(
        tipo=tipo,
        periodo_yyyy_mm=proxima_fecha,
        pred=pred,
        payment=payment,
        intervalo=intervalo,
        cedula=id,
        predict_percentage=error_usado
    )

    print("Forecast guardado en base de datos")
        
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
        print(f"PROCESO COMPLETADO - Cédula: {cedula}, Tipo: {tipo}")
        print("=" * 60)

def mostrar_analisis_precision(df, mape_lineal, mape_prophet, intervalo, pred):
    """
    Muestra análisis completo de precisión para tesis.
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS DE PRECISIÓN - NIVEL TESIS")
    print("=" * 60)
    
    y = df['y'].values
    n_total = len(y)
    n_ceros = np.sum(y == 0)
    
    if n_ceros > 0:
        metrica_nombre = "SMAPE"
        razon = f"La serie contiene {n_ceros} valores cero → usando SMAPE (robusto a ceros)"
    else:
        metrica_nombre = "MAPE"
        razon = "Serie sin valores cero → usando MAPE"
    
    print(f"\n METODOLOGÍA DE VALIDACIÓN:")
    print(f"   • Datos: {n_total} meses históricos")
    print(f"   • Métrica: {metrica_nombre}")
    print(f"   • {razon}")
    
    if n_total < 12:
        print(f"   • Técnica: Leave-One-Out Cross Validation (LOOCV)")
    elif n_total < 24:
        print(f"   • Técnica: Validación parcial (70% train, 30% test)")
    else:
        print(f"   • Técnica: Walk-Forward Validation completa")
    
    print(f"\n RESULTADOS DE VALIDACIÓN:")
    print(f"   • Regresión Lineal: {mape_lineal:.2f}% {metrica_nombre}")
    
    if mape_prophet < np.inf:
        print(f"   • Prophet: {mape_prophet:.2f}% {metrica_nombre}")
    
    print(f"\n INTERPRETACIÓN ACADÉMICA:")
    
    error_usado = mape_lineal if mape_prophet == np.inf else min(mape_lineal, mape_prophet)
    
    if error_usado < 10:
        nivel = "EXCELENTE"
        confianza = "Alta (modelo muy confiable)"
    elif error_usado < 20:
        nivel = "BUENA" 
        confianza = "Media-Alta (modelo aceptable)"
    elif error_usado < 30:
        nivel = "MODERADA"
        confianza = "Media (usar con cautela)"
    elif error_usado < 50:
        nivel = "BAJA"
        confianza = "Baja (alta incertidumbre)"
    else:
        nivel = "MUY BAJA"
        confianza = "Muy baja (poco confiable)"
    
    print(f"   • Precisión {nivel}: {error_usado:.1f}% error")
    print(f"   • Nivel de confianza: {confianza}")
    
    precision_porcentual = max(0, 100 - error_usado)
    
    print(f"\n INFERENCIA SOBRE PREDICCIÓN FUTURA:")
    print(f"   Basado en la validación histórica:")
    print(f"   • Error esperado: ~{error_usado:.1f}%")
    print(f"   • Precisión estimada: {precision_porcentual:.1f}%")
    print(f"   • Intervalo de confianza 95%: [{intervalo[0]:.2f}, {intervalo[1]:.2f}]")
    
    if pred != 0:
        amplitud_rel = (intervalo[1] - intervalo[0]) / abs(pred) * 100
        print(f"   • Amplitud del intervalo: {amplitud_rel:.1f}% del valor predicho")
    
    print(f"\n JUSTIFICACIÓN:")
    print(f"   La métrica {metrica_nombre} de {error_usado:.1f}% indica que el modelo")
    print(f"   tiene una precisión {nivel.lower()}, lo que es {'aceptable' if error_usado < 30 else 'limitado'}")
    print(f"   para predicciones futuras en contextos reales.")