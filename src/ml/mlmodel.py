import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import sys
from sklearn.metrics import mean_absolute_error

from src.models.prophet import evaluar_modelos_prophet, walk_forward_validation
from src.utils.firebaseStorage import descargar_csv_firebase
from src.utils.monthlyForecast import predecir_proximo_mes
from src.utils.seriesAnalytics import analizar_serie


def ejecutar_sistema_completo(cedula: str, tipo: str):
    print("=" * 60)
    print("SISTEMA DE PREDICCIÓN - VERSIÓN TESIS OPTIMIZADA")
    print("=" * 60)
    
    ruta_local = descargar_csv_firebase(cedula, tipo)
    if ruta_local is None:
        print(f"Se omite {cedula} ({tipo}) por falta de archivo")
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
        print(f"Error al cargar datos: {e}")
        sys.exit(1)
    
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