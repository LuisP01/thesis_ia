from prophet import Prophet
from sklearn.linear_model import LinearRegression
import numpy as np

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
            yearly_seasonality=False,
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