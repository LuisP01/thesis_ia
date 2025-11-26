import pandas as pd
import matplotlib.pyplot as plt

# 1. Asegúrate de usar el separador correcto (punto y coma)
# 2. Usa decimal=',' ya que tus temperaturas usan coma (26,8)
df = pd.read_csv('data.csv', sep=';', decimal=',')

# Una vez cargado correctamente, el código de conversión de fecha funcionará
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')

# 1. Temperatura vs tiempo
plt.plot(df['Fecha'], df['Temperatura'])
plt.title("Temperatura promedio")
plt.xlabel("Fecha")
plt.ylabel("Temperatura")
plt.show()

# 2. Consumo vs tiempo
plt.plot(df['Fecha'], df['Consumido'])
plt.title("Consumo promedio")
plt.xlabel("Fecha")
plt.ylabel("Consumido")
plt.show()

# 3. Scatter: temperatura vs consumo
plt.scatter(df['Temperatura'], df['Consumido'])
plt.title("Relación entre temperatura y Consumido")
plt.xlabel("Temperatura")
plt.ylabel("Consumido")
plt.show()
