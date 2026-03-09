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
    
    print(f"\n Evaluando Prophet: {len(cps_list)}×{len(sps_list)} = {len(cps_list)*len(sps_list)} combinaciones")
    
    for cps in cps_list:
        for sps in sps_list:
            print(f"  • Probando cps={cps}, sps={sps}")
            
            mape, _ = walk_forward_validation(
                df, 
                modelo_tipo='prophet',
                parametros={'changepoint_prior_scale': cps, 'seasonality_prior_scale': sps},
                verbose=False
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
    Versión TESIS - maneja series cortas y valores cero automáticamente.
    """
    y = df['y'].values
    n_total = len(y)
    
    n_ceros = np.sum(y == 0)
    usar_smape = n_ceros > 0
    
    errores = []
    predicciones = []
    reales = []
    
    print(f"    Serie de {n_total} meses", end="")
    if n_ceros > 0:
        print(f" ({n_ceros} valores cero → usando SMAPE)")
    else:
        print(" (sin ceros → usando MAPE)")
    
    if n_total < 8:
        print(f"     Serie muy corta (<8 meses) → predicción básica")
        if modelo_tipo == 'lineal' and n_total >= 3:
            X = np.arange(n_total).reshape(-1, 1)
            modelo = LinearRegression()
            modelo.fit(X, y)
            pred = modelo.predict([[n_total]])[0]
            
            residuos = y - modelo.predict(X)
            if len(residuos) > 1:
                error_estimado = np.std(residuos) / np.mean(np.abs(y)) * 100 if np.mean(np.abs(y)) > 0 else 30.0
            else:
                error_estimado = 25.0  
            
            metric = error_estimado
            print(f"    Error estimado: {metric:.1f}% (serie muy corta)")
            
        else:
            metric = 30.0  
        return metric, ([], [])
    
    elif n_total < 12:
        print(f"    Serie corta ({n_total} meses) → usando Leave-One-Out")
        
        for i in range(n_total):
            mask = np.ones(n_total, dtype=bool)
            mask[i] = False
            df_train = df[mask].copy()
            
            if len(df_train) < 4:  
                continue
                
            if modelo_tipo == 'lineal':
                X_train = np.arange(len(df_train)).reshape(-1, 1)
                y_train = df_train['y'].values
                modelo = LinearRegression()
                modelo.fit(X_train, y_train)
                pred = modelo.predict([[len(df_train)]])[0]
                
            elif modelo_tipo == 'prophet':
                if parametros is None:
                    parametros = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10}
                m = Prophet(
                    yearly_seasonality=False,  
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
            
            real = y[i]
            
            if usar_smape or real == 0:
                error = 2 * abs(pred - real) / (abs(real) + abs(pred) + 1e-10)
            else:
                error = abs(pred - real) / max(abs(real), 1e-6)
            
            errores.append(error)
            predicciones.append(pred)
            reales.append(real)
    
    elif n_total < 24:
        print(f"    Serie media ({n_total} meses) → usando validación parcial")
        
        split_idx = int(n_total * 0.7)
        if split_idx < 6:  
            split_idx = max(6, n_total - 4)
        
        for i in range(split_idx, n_total):
            df_train = df.iloc[:i].copy()
            
            if len(df_train) < 6: 
                continue
                
            if modelo_tipo == 'lineal':
                X_train = np.arange(len(df_train)).reshape(-1, 1)
                y_train = df_train['y'].values
                modelo = LinearRegression()
                modelo.fit(X_train, y_train)
                pred = modelo.predict([[len(df_train)]])[0]
                
            elif modelo_tipo == 'prophet':
                if parametros is None:
                    parametros = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10}
                m = Prophet(
                    yearly_seasonality=False,
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
            
            real = y[i]
            
            if usar_smape or real == 0:
                error = 2 * abs(pred - real) / (abs(real) + abs(pred) + 1e-10)
            else:
                error = abs(pred - real) / max(abs(real), 1e-6)
            
            errores.append(error)
            predicciones.append(pred)
            reales.append(real)
    
    else:
        print(f"    Serie larga ({n_total} meses) → usando walk-forward completo")
        
        for i in range(12, n_total - 1):
            df_train = df.iloc[:i+1].copy()
            
            if modelo_tipo == 'lineal':
                X_train = np.arange(len(df_train)).reshape(-1, 1)
                y_train = df_train['y'].values
                modelo = LinearRegression()
                modelo.fit(X_train, y_train)
                pred = modelo.predict([[len(df_train)]])[0]
                
            elif modelo_tipo == 'prophet':
                if parametros is None:
                    parametros = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10}
                m = Prophet(
                    yearly_seasonality=False,
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
            
            real = y[i+1]
            
            if usar_smape or real == 0:
                error = 2 * abs(pred - real) / (abs(real) + abs(pred) + 1e-10)
            else:
                error = abs(pred - real) / max(abs(real), 1e-6)
            
            errores.append(error)
            predicciones.append(pred)
            reales.append(real)
    
    if errores:
        metric = np.mean(errores) * 100
        if usar_smape:
            print(f"    SMAPE: {metric:.2f}% (robusto con ceros)")
        else:
            print(f"    MAPE: {metric:.2f}%")
        return metric, (reales, predicciones)
    
    print(f"    No se pudo calcular error -> usando estimación: 25.0%")
    return 25.0, ([], [])