import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

"""Datos"""
mu = 35
sigma = 2.5

"""Punto a: Gráficas de la función de densidad de probabilidad y la función de distribución acumulativa"""
# Rango de temperaturas
temperaturas = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

# PDF: Función de Densidad de Probabilidad
pdf = norm.pdf(temperaturas, mu, sigma)

# CDF: Función de Distribución Acumulada
cdf = norm.cdf(temperaturas, mu, sigma)

# Gráfica de la PDF
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(temperaturas, pdf, color='blue', linewidth=2)
plt.title('Función de Densidad de Probabilidad (PDF)')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Densidad de probabilidad')
plt.grid(True)

# Gráfica de la CDF
plt.subplot(1, 2, 2)
plt.plot(temperaturas, cdf, color='red', linewidth=2)
plt.title('Función de Distribución Acumulada (CDF)')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Probabilidad acumulada')
plt.grid(True)

# Ajustar y mostrar las gráficas
plt.tight_layout()
plt.show()

"""Punto b: Probabilidad de que se active el sistema de enfriamiento de un servidor al azar al superar los 40°C"""
print("Punto b:")

# Probabilidad P(X > 40) = 1 - P(X <= 40)
prob = 1 - norm.cdf(40, mu, sigma)
print(f"La probabilidad de que se active el sistema de enfriamiento gracias a que la temperatura sea mayor a 40°C es de {prob * 100:.2f}%")

print("--------------------------------------------------")

"""Punto c: Porcentaje de tiempo que un servidor opere entre los 30°C y 38°C (temperatura ideal)"""
print("Punto c:")

# Probabilidad P(30 <= X <= 38) = P(X <= 38) - P(X <= 30)
prob = norm.cdf(38, mu, sigma) - norm.cdf(30, mu, sigma)
print(f"Porcentaje de tiempo en que el servidor opera en el rango óptimo: {prob * 100:.2f}%")

print("--------------------------------------------------")

"""Punto d: Umbral de alerta temprana que active una advertencia en el 10% superior de las temperaturas (percentil 90)"""
print("Punto d:")

# Percentil 90
T = norm.ppf(0.90, mu, sigma)
print(f"El umbral de alerta temprana (percentil 90) es de {T:.2f} °C")

print("--------------------------------------------------")

"""Punto e: Promedio de temperaturas de 5 servidores que superen los 37°C"""
print("Punto e:")
n = 5
sigma_promedio = sigma / (n ** 0.5)

# Probabilidad P(promedio > 37)
prob = 1 - norm.cdf(37, mu, sigma_promedio)
print(f"La probabilidad de que el promedio de los 5 servidores sea mayor a 37°C es de {prob * 100:.2f}%")

print("--------------------------------------------------")

"""Punto f: Probabilidad de que de los 1.000 servidores más de 25 de ellos tengan una temperatura superior a 41°C"""
print("Punto f:")

# Probabilidad de que un servidor tenga una temperatura superior a 41°C
p = 1 - norm.cdf(41, loc=35, scale=2.5)
print(f"La probabilidad de que un servidor tenga una temperatura superior a 41°C es de {p * 100:.2f}%")

# Probabilidad de que más de 25 servidores tengan una temperatura superior a 41°C
mu_Y = 1000 * p
sigma_Y = (1000 * p * (1 - p)) ** 0.5
z = (25.5 - mu_Y) / sigma_Y
prob = 1 - norm.cdf(z)
print(f"La proabilidad de que más de 25 de los 1.000 servidores tengan una temperatura superior a 41°C es aproximadamente de {prob * 100:.10f}%")

print("--------------------------------------------------")

"""Punto g: Temperatura máxima que deberá soportar un sistema de enfriamiento que pueda manejar el 99.9% de todos los eventos de temperatura"""
print("Punto g:")

# Temperatura máxima que soporta el sistema de enfriamiento (percentil 99.9)
T_max = norm.ppf(0.999, loc=mu, scale=sigma)
print(f"La temperatura máxima a soportar (percentil 99.9) es de {T_max:.2f}°C")

print("--------------------------------------------------")

"""Punto h: Diseñar un sistema de enfriamiento variable basado en los cuartiles"""
print("Punto h:")

# Cuartiles
Q1 = norm.ppf(0.25, loc=mu, scale=sigma) 
Q2 = norm.ppf(0.50, loc=mu, scale=sigma) 
Q3 = norm.ppf(0.75, loc=mu, scale=sigma)

print(f"Cuartiles:")
print(f"Q1 = {Q1:.2f}°C")
print(f"Q2 = {Q2:.2f}°C")
print(f"Q3 = {Q3:.2f}°C")

# Conclusiones sobre el sistema de enfriamiento variable
print("Conclusión general:\n- El sistema de enfriamiento variable se puede ajustar para operar de manera óptima en función de los cuartiles.")
print("Eficiencia energética:\n- Reduce el consumo al usar solo la potencia necesaria en cada rango, por ejemplo, no activar refrigeración líquida si T ≤ 33.31°C.")
print("Prevención de sobrecargas:\n- El modo crítico (> 36.69°C) actúa antes de llegar a umbrales peligrosos.")
print("Adaptabilidad:\n- Se ajusta automáticamente a la distribución natural de las temperaturas.")

print("--------------------------------------------------")