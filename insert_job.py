import os
import redis
import psycopg2
from datetime import datetime

# 1. Conexión a Redis
redis_client = redis.Redis(
    host=os.environ["REDIS_HOST"],
    port=int(os.environ["REDIS_PORT"]),
    password=os.environ["REDIS_PASSWORD"],
    decode_responses=True
)

# 2. Conexión a PostgreSQL
conn = psycopg2.connect(
    host=os.environ["PG_HOST"],
    port=os.environ["PG_PORT"],
    user=os.environ["PG_USER"],
    password=os.environ["PG_PASSWORD"],
    database=os.environ["PG_DATABASE"]
)
cursor = conn.cursor()

# 3. Obtener valores del día
energia_kwh = redis_client.get("energia_dia_actual")
fecha_dia = redis_client.get("fecha_dia_actual")

if energia_kwh is None:
    print("No hay datos en Redis para hoy. Cron finalizado.")
    exit()

energia_kwh = float(energia_kwh)
fecha_dia = datetime.strptime(fecha_dia, "%Y-%m-%d").date()

print(f"📥 Insertando en BD | Fecha: {fecha_dia} | Energía: {energia_kwh:.4f} kWh")

# 4. Insertar en PostgreSQL
cursor.execute("""
    INSERT INTO consumo_diario (fecha, energia_kwh, temperatura)
    VALUES (%s, %s, %s)
""", (fecha_dia, energia_kwh))

conn.commit()

# 5. Resetear Redis
redis_client.set("energia_dia_actual", 0)
redis_client.set("fecha_dia_actual", datetime.now().strftime("%Y-%m-%d"))

print("Redis reseteado. Día cerrado correctamente.")

cursor.close()
conn.close()
