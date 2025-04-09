import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt

transacciones_csv = pd.read_csv('transacciones.csv')

transacciones_df = pd.DataFrame(transacciones_csv)

transacciones_totales = transacciones_df.shape[0]
transacciones_sospechosas = (transacciones_df[transacciones_df['Estado'] == 'sospechosa']).shape[0]
transacciones_fraudulentas = (transacciones_df[transacciones_df['Estado'] == 'fraudulenta']).shape[0]

print("Transacciones totales:", transacciones_totales)
print("Transacciones sospechosas:", transacciones_sospechosas)
print("Transacciones fraudulentas:", transacciones_fraudulentas)

print("--------------------------------------------------")

"""Probabilidad de fraude"""

probabilidad_fraude = (transacciones_fraudulentas / transacciones_totales) * 100
print(f"La probabilidad de que un día determinado una de las transacciones sospechosas sea fraudulenta es de {probabilidad_fraude}%")

"""Cantidad más probable de transacciones fraudulentas por día"""
promedio_transacciones_día = transacciones_totales / 30
cantidad_sospechosas_día = transacciones_sospechosas / 30
cantidad_fraudulentas_día = transacciones_fraudulentas / 30

λ = 1
k_values = np.arange(0, 10)
probabilities = poisson.pmf(k_values, λ)

plt.figure(figsize=(10, 6))
plt.bar(k_values, probabilities * 100, color='skyblue')
plt.title('Distribución de probabilidad de fraudes por día', fontsize=14)
plt.xlabel('Número de transacciones fraudulentas por día', fontsize=12)
plt.ylabel('Probabilidad (%)', fontsize=12)
plt.xticks(k_values)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, prob in enumerate(probabilities * 100):
    plt.text(i, prob + 1, f'{prob:.1f}%', ha='center', fontsize=10)

plt.show()