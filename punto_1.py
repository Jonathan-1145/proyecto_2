import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, expon

"""Generación de muestras aleatorias de tamaño 150 de las siguientes distribuciones"""
# X ∼ b(x; n, p). Seleccione valores para n (entre 10 y 25) y p.
np.random.seed(123)
muestra_binomial = np.random.binomial(n=15, p=0.3, size=150)

# Y ∼ exp(x; λ). Seleccione un valor para λ.
muestra_exponencial = np.random.exponential(scale=1/0.5, size=150)

""""Punto a: Generación de histogramas y comparación con la función de probabilidad teórica"""
# Muestra binomial
counts, bins, _ = plt.hist(muestra_binomial, bins=range(0, 17), density=True, alpha=0.7, edgecolor='black', label='Muestra (Histograma)')

k = np.arange(0, 16)
pmf_teorica = binom.pmf(k, n=15, p=0.3)

plt.scatter(k, pmf_teorica, color='red', marker='o', label='PMF Teórica')

plt.title('Histograma de la Muestra Binomial y PMF Teórica')
plt.xlabel('Número de Éxitos (k)')
plt.ylabel('Probabilidad')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# Muestra exponencial
plt.hist(muestra_exponencial, bins=20, density=True, alpha=0.7, edgecolor='black', label='Muestra (Histograma)')

y = np.linspace(0, 10, 1000)
pdf_teorica = expon.pdf(y, scale=2)

plt.plot(y, pdf_teorica, color='red', linewidth=2, label='Teórica (PDF)')

plt.title('Exponencial(λ=0.5): Muestra vs Teórica')
plt.xlabel('Valor (y)')
plt.ylabel('Densidad de Probabilidad')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.xlim(0, 10)
plt.show()

"""Punto b: Promedio, cuartiles y desviación estandar de las muestras. Comparación con los valores teóricos"""
# Muestra binomial
media_binomial = np.mean(muestra_binomial)
desviacion_estandar_binomial = np.std(muestra_binomial, ddof=1)
cuartiles_binomial = np.percentile(muestra_binomial, [25, 50, 75])


n = 15
p = 0.3
media_teorica_binomial = n * p
desviacion_estandar_teorica_binomial = np.sqrt(n * p * (1 - p))
Q1_teorico = binom.ppf(0.25, n=15, p=0.3)
Q2_teorico = binom.ppf(0.50, n=15, p=0.3)
Q3_teorico = binom.ppf(0.75, n=15, p=0.3)

print("Muestra Binomial")
print("Media Muestra:", media_binomial)
print("Media Teórica:", media_teorica_binomial)
print("Desviación Estándar Muestra:", desviacion_estandar_binomial)
print("Desviación Estándar Teórica:", desviacion_estandar_teorica_binomial)
print("Cuartiles Muestra:", cuartiles_binomial)
print("Cuartiles Teóricos:", f"[{Q1_teorico:.0f}. {Q2_teorico:.0f}. {Q3_teorico:.0f}.]")

print("--------------------------------------------------")
# Muestra exponencial
media_exponencial = np.mean(muestra_exponencial)
desviacion_estandar_exponencial = np.std(muestra_exponencial, ddof=1)
cuartiles_exponencial = np.percentile(muestra_exponencial, [25, 50, 75])

λ = 0.5
media_teorica_exponencial = 1 / λ
desviacion_estandar_teorica_exponencial = 1 /  λ
Q1_teorico_exponencial = expon.ppf(0.25, scale=2)
Q2_teorico_exponencial = expon.ppf(0.50, scale=2)
Q3_teorico_exponencial = expon.ppf(0.75, scale=2)

print("Muestra Exponencial")
print("Media Muestra:", media_exponencial)
print("Media Teórica:", media_teorica_exponencial)
print("Desviación Estándar Muestra:", desviacion_estandar_exponencial)
print("Desviación Estándar Teórica:", desviacion_estandar_teorica_exponencial)
print("Cuartiles Muestra:", cuartiles_exponencial)
print("Cuartiles Teóricos:", f"[{Q1_teorico_exponencial:.8f}. {Q2_teorico_exponencial:.8f}. {Q3_teorico_exponencial:.8f}.]")