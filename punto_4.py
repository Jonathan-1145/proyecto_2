import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

web_server_requests_g = pd.read_csv('Web_Server_Requests_G.csv')

web_server_requests_g_df = pd.DataFrame(web_server_requests_g)

"""Punto a: Identificar patrones de tráfico como horas pico de actividad y momentos de menor demanda"""
# Convertir Timestamp a datetime y extraer las horas y minutos
web_server_requests_g_df['Timestamp'] = pd.to_datetime(web_server_requests_g_df['Timestamp'])
web_server_requests_g_df['Hora'] = web_server_requests_g_df['Timestamp'].dt.hour
web_server_requests_g_df['Minuto'] = web_server_requests_g_df['Timestamp'].dt.minute

# 1. Analizar por horas (promedio de solicitudes por hora)
hourly_avg = web_server_requests_g_df.groupby('Hora')['ArrivalRate'].mean()

# 2. Identificar picos y valles
hora_pico = hourly_avg.idxmax()
valor_pico = hourly_avg.max()
hora_valle = hourly_avg.idxmin()
valor_valle = hourly_avg.min()

# 3. Visualizar detalladamente
plt.figure(figsize=(14, 7))

# Gráfico de línea para la tendencia horaria
plt.subplot(2, 1, 1)
hourly_avg.plot(kind='line', marker='o', color='b', linestyle='-', linewidth=2)
plt.title('Patrones de Tráfico Web - Tendencia Horaria', fontsize=14)
plt.xlabel('Hora del Día', fontsize=12)
plt.ylabel('Arrival Rate Promedio', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(x=hora_pico, color='r', linestyle='--', label=f'Pico: {hora_pico}:00 ({valor_pico:.1f} solicitudes)')
plt.axvline(x=hora_valle, color='g', linestyle='--', label=f'Valle: {hora_valle}:00 ({valor_valle:.1f} solicitudes)')
plt.legend()

# Gráfico de cajas y bigotes por hora para la distribución detallada
plt.subplot(2, 1, 2)
sns.boxplot(x='Hora', y='ArrivalRate', data=web_server_requests_g_df, palette='coolwarm', hue='Hora', legend=False)
plt.title('Distribución de Solicitudes por Hora', fontsize=14)
plt.xlabel('Hora del Día', fontsize=12)
plt.ylabel('Arrival Rate', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Resultados numéricos
print("\nAnálisis de Patrones de Tráfico:")
print(f"- Hora pico: {hora_pico}:00 hrs ({valor_pico:.2f} solicitudes/intervalo)")
print(f"- Hora valle: {hora_valle}:00 hrs ({valor_valle:.2f} solicitudes/intervalo)")
print(f"- Rango diario: {valor_pico-valor_valle:.2f} solicitudes/intervalo\n")
print("--------------------------------------------------\n")

"""Punto b: Distribución Gamma para los datos de 'ArrivalRate', histograma de sus datos y la función de densidad de probabilidad (PDF)"""
# Cargar los datos de ArrivalRate
arrival_rates = web_server_requests_g['ArrivalRate']

# Ajustar una distribución gamma a los datos
params = gamma.fit(arrival_rates)
shape, loc, scale = params

# Crear histograma y gráfico de densidad
plt.figure(figsize=(10, 6))
count, bins, ignored = plt.hist(arrival_rates, bins=30, density=True, alpha=0.6, color='g', label='Datos observados')

# Generar valores para la PDF teórica
x = np.linspace(min(arrival_rates), max(arrival_rates), 1000)
pdf = gamma.pdf(x, shape, loc=loc, scale=scale)

# Graficar la PDF ajustada
plt.plot(x, pdf, 'r-', lw=2, label=f'Gamma ajustada (α={shape:.2f}, β={1/scale:.2f})')

# Configuración del gráfico
plt.title('Ajuste de Distribución Gamma al Arrival Rate del Servidor Web')
plt.xlabel('Arrival Rate (solicitudes por intervalo)')
plt.ylabel('Densidad de probabilidad')
plt.legend()
plt.grid(True)

# Mostrar parámetros en el gráfico
params_text = f'Parámetros gamma:\nα (shape) = {shape:.2f}\nβ (rate) = {1/scale:.2f}\nθ (scale) = {scale:.2f}\nloc = {loc:.2f}'
plt.text(0.95, 0.95, params_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.show()

"""Punto c: Probabilidad de que el servidor reciba más de 8 solicitudes durante el próximo intervalo"""
# P(X > 8) = 1 - P(X <= 8)
prob_mas_8 = 1 - gamma.cdf(8, shape, loc=loc, scale=scale)
print(f"La probabilidad de recibir más de 8 solicitudes es de {(prob_mas_8 * 100):.2f}%\n")

print("--------------------------------------------------\n")

"""Punto d: Probabilidad de que la tasa de llegada supere un umbral crítico de, por ejemplo, 15 solicitudes por minuto, en el siguiente intervalo"""
shape, loc, scale = gamma.fit(arrival_rates)

shape = shape
loc = loc
scale = scale

# P(X > 15)
prob_mas_15 = 1 - gamma.cdf(15, shape, loc=loc, scale=scale)
print(f"La probabilidad de que se reciban más de 15 solicitudes por minuto en el siguiente intervalo es de {(prob_mas_15 * 100):.2f}%\n")

print("--------------------------------------------------\n")

"""Punto e: Probabilidad de superar la capacidad máxima de procesamiento del servidor en el siguiente intervalo si la capacidad máxima es de 20 solicitudes por intervalo"""
# P(X > 20)
prob_mas_20 = 1 - gamma.cdf(20, shape, loc=loc, scale=scale)
print(f"La probabilidad de obtener 20 solicitudes por intervalo y así superar la capacidad máxima del servidor es de {(prob_mas_20 * 100):.4f}%\n")

print("--------------------------------------------------")