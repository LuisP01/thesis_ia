import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression

def evaluar_modelos_prophet(df, caracteristicas):
    """
    Evalúa diferentes configuraciones de Prophet basadas en las características.
    Versión simplificada para tesis.
    """
    resultados = []
    
    if caracteristicas['r2'] > 0.7:
        cps_list = [0.01, 0.05]
    elif caracteristicas['cv'] > 0.5:
        cps_list = [0.1, 0.2]
    else:
        cps_list = [0.05, 0.1]
    
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
    
    resultados.sort(key=lambda x: x['mape'])
    
    return resultados


def walk_forward_validation(df, modelo_tipo='prophet', parametros=None, verbose=False):

    """
    Validación temporal walk-forward para series temporales.
    Solo usa datos pasados para predecir futuros.
    """
    errores = []
    predicciones = []
    reales = []
    
    n_total = len(df)
    if n_total < 24:
        print("    Pocos datos para validación walk-forward completa")
        return np.inf, ([], [])
    
    for i in range(12, n_total - 1):
        df_train = df.iloc[:i+1].copy()
        
        if modelo_tipo == 'prophet':
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
            
            future = m.make_future_dataframe(periods=1, freq='ME')
            forecast = m.predict(future)
            pred = forecast.iloc[-1]['yhat']
            
        elif modelo_tipo == 'lineal':
            X_train = np.arange(len(df_train)).reshape(-1, 1)
            y_train = df_train['y'].values
            
            modelo = LinearRegression()
            modelo.fit(X_train, y_train)
            
            pred = modelo.predict([[len(df_train)]])[0]
        
        real = df.iloc[i+1]['y']
        
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