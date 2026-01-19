from datetime import datetime
from src.config.dbConfig import get_postgres_connection
from src.queries.queries import INSERT_FORECAST


def guardar_forecast(tipo, periodo_yyyy_mm, pred, payment, intervalo, cedula):
    period_date = datetime.strptime(periodo_yyyy_mm + "-01", "%Y-%m-%d")

    conn = get_postgres_connection()
    cur = conn.cursor()

    cur.execute(
        INSERT_FORECAST,
        (
            tipo,
            period_date,
            float(pred),
            float(payment),
            cedula,
            float(intervalo[1]), 
            float(intervalo[0])
        )
    )

    conn.commit()
    cur.close()
    conn.close()

    print("Forecast insertado en BD")
