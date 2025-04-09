import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Función para las estadísticas descriptivas

def print_stats(name, data):
    print(f"\n📊 {name}:")
    print("-" * 40)
    print(f"  • Count: {len(data):>10}")
    print(f"  • Mean: {data.mean():>12.2f} USD")
    print(f"  • Std: {data.std():>13.2f} USD")
    print(f"  • Min: {data.min():>13.2f} USD")
    print(f"  • 25%: {data.quantile(0.25):>13.2f} USD")
    print(f"  • 50%: {data.median():>13.2f} USD")
    print(f"  • 75%: {data.quantile(0.75):>13.2f} USD")
    print(f"  • Max: {data.max():>13.2f} USD")

print("🔍 Estadísticas Descriptivas")
print("=" * 40)

"""Punto a: Distribución de los dos set de datos"""

# Carga, análisis y visualización de datos de det_fraudes_train.csv

# Cargar datos de entrenamiento
det_fraudes_train = pd.read_csv('det_fraudes_train.csv')

# Separar transacciones normales y fraudulentas (entrenamiento)
normal = det_fraudes_train[det_fraudes_train['categoría'] == 0]['monto (USD)']
fraud = det_fraudes_train[det_fraudes_train['categoría'] == 1]['monto (USD)']

# Estadísticas descriptivas
print_stats("Datos de Entrenamiento - Transacciones Normales", normal)
print_stats("Datos de Entrenamiento - Transacciones Fraudulentas", fraud)

# Conteo de categorías
count_normal = len(det_fraudes_train[det_fraudes_train['categoría'] == 0])
count_fraud = len(det_fraudes_train[det_fraudes_train['categoría'] == 1])

# Histograma de montos (entrenamiento)
plt.figure(figsize=(10, 6))
# Histograma de transacciones NORMALES (categoría 0)
sns.histplot(data=det_fraudes_train[det_fraudes_train['categoría'] == 0], x='monto (USD)', color='blue', kde=True, bins=50, label='Normales (categoría 0)')
# Histograma de transacciones FRAUDULENTAS (categoría 1)
sns.histplot(data=det_fraudes_train[det_fraudes_train['categoría'] == 1], x='monto (USD)', color='red', kde=True, bins=50, label='Fraudulentas (categoría 1)')
# Línea de la media global
plt.axvline(x=det_fraudes_train['monto (USD)'].mean(), color='black', linestyle='--', label=f'Media global: {det_fraudes_train["monto (USD)"].mean():.2f} USD')
plt.title('Distribución de Montos en el Conjunto de Entrenamiento')
plt.xlabel('Monto (USD)')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

# Carga, análisis y visualización de datos de det_fraudes_test.csv

# Cargar datos de prueba
det_fraudes_test = pd.read_csv('det_fraudes_test.csv')

# Estadísticas descriptivas
print_stats("Datos de Prueba - Todos los datos", det_fraudes_test['monto (USD)'])

# Histograma de montos (prueba)
plt.figure(figsize=(10, 6))
# Histograma + KDE de los montos de prueba (color único para evitar confusión)
sns.histplot(data=det_fraudes_test, x='monto (USD)', color='orange', kde=True, bins=30, label='Transacciones (Prueba)', alpha=0.6)
# Línea de la media global
mean_test = det_fraudes_test['monto (USD)'].mean()
plt.axvline(x=mean_test, color='black', linestyle='--', linewidth=2, label=f'Media: {mean_test:.2f} USD')
plt.title('Distribución de Montos en el Conjunto de Prueba', fontsize=14)
plt.xlabel('Monto (USD)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)  # Cuadrícula horizontal sutil
plt.show()

# Comparación de distribuciones entre entrenamiento y prueba
# Histograma de montos (entrenamiento versus prueba) 
plt.figure(figsize=(12, 6))
# 1. Entrenamiento (Normales)
sns.kdeplot(data=det_fraudes_train[det_fraudes_train['categoría'] == 0]['monto (USD)'], color='blue', label='Entrenamiento: Normales', fill=True, alpha=0.3)
# 2. Entrenamiento (Fraudulentas)
sns.kdeplot(data=det_fraudes_train[det_fraudes_train['categoría'] == 1]['monto (USD)'], color='red', label='Entrenamiento: Fraudulentas', fill=True, alpha=0.3)
# 3. Prueba
sns.kdeplot(data=det_fraudes_test['monto (USD)'], color='purple', label='Prueba: Sin etiqueta', fill=True, alpha=0.3)
# Ajustes visuales
plt.title('Comparación de Distribuciones: Normales vs Fraudulentas vs Prueba', fontsize=14)
plt.xlabel('Monto (USD)', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend(loc='upper right')  # Leyenda clara
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()

"""Punto b: Criterio para clasificar una transacción como 'normal' o 'fraudulenta' a partir de la distribución normal"""
print("\n🔍 Clasificación de Transacciones")
print("=" * 40)
# 1. Calcular media y desviación estándar de transacciones normales (categoría 0)
normales = det_fraudes_train[det_fraudes_train['categoría'] == 0]['monto (USD)']
mean = normales.mean()
std = normales.std()

# 2. Definir umbral de fraude (3 desviaciones estándar arriba de la media)
umbral_fraude = mean + 3 * std
print(f"Umbral de fraude: {umbral_fraude:.2f} USD")

# 3. Función para clasificar
def clasificar_transaccion(monto):
    return "Fraudulenta" if monto > umbral_fraude else "Normal"

# 4. Mostrar resultados
print("\nPrimeras clasificaciones en el set de prueba:")
print(det_fraudes_test.head())

# 5. Visualización
plt.figure(figsize=(10, 6))
sns.histplot(normales, kde=True, color='blue', label='Normales')
plt.axvline(umbral_fraude, color='red', linestyle='--', label=f'Umbral: {umbral_fraude:.2f} USD')
plt.title('Distribución de Montos Normales vs Umbral de Fraude')
plt.xlabel('Monto (USD)')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

"""Punto c: Función exactitud para comprobar si el clasiicador funciona de forma adecuada"""
# 1. Crear columna 'predicción'
det_fraudes_train['predicción'] = det_fraudes_train['monto (USD)'].apply(lambda x: 1 if x > umbral_fraude else 0)

# 2. Calcular exactitud
aciertos = sum(det_fraudes_train['predicción'] == det_fraudes_train['categoría'])
total_datos = len(det_fraudes_train)
exactitud = 100 * (aciertos / total_datos)

# 3. Mostrar resultados
print()
print(f"   📊 Evaluación del Clasificador (Entrenamiento)")
print("-" * 40)
print(f"- Umbral de fraude: {umbral_fraude:.2f} USD")
print(f"- Transacciones normales (0): {len(normales)}")
print(f"- Transacciones fraudulentas (1): {len(det_fraudes_train) - len(normales)}")
print(f"- Aciertos: {aciertos} / {total_datos}")
print(f"- Exactitud: {exactitud:.2f}%")

"""Punto d: Clasificación de los datos del set de prueba con el modelo entrenado por los pasos anteriores"""
print()
print("   🔍 CLASIFICACIÓN DEL SET DE PRUEBA")
print("-" * 40)

# 1. Aplicar el clasificador al set de prueba
det_fraudes_test['clasificación'] = det_fraudes_test['monto (USD)'].apply(clasificar_transaccion)

# 2. Mostrar estadísticas de clasificación
num_normales = sum(det_fraudes_test['clasificación'] == "Normal")
num_fraudes = sum(det_fraudes_test['clasificación'] == "Fraudulenta")

print(f"- Transacciones clasificadas como normales: {num_normales}")
print(f"- Transacciones clasificadas como fraudulentas: {num_fraudes}")
print(f"- Porcentaje de fraudes detectados: {num_fraudes / len(det_fraudes_test) * 100:.2f}%")

# 3. Mostrar ejemplos de transacciones fraudulentas detectadas
print("\nEjemplos de transacciones clasificadas como fraudulentas:")
print(det_fraudes_test[det_fraudes_test['clasificación'] == "Fraudulenta"].head())

# 4. Visualización
plt.figure(figsize=(12, 6))
# Histograma con colores diferenciados
ax = sns.histplot(data=det_fraudes_test, x='monto (USD)', hue='clasificación', palette={"Normal": "blue", "Fraudulenta": "red"}, kde=False, bins=30, alpha=0.6, edgecolor='white')
# Línea de umbral con anotación
plt.axvline(umbral_fraude, color='black', linestyle='--', linewidth=1.5, label=f'Umbral de fraude: {umbral_fraude:.2f} USD')
# Personalización básica
plt.title('Clasificación de Transacciones en el Set de Prueba', fontsize=14)
plt.xlabel('Monto (USD)', fontsize=12)
plt.ylabel('Número de Transacciones', fontsize=12)
# Leyenda explicativa (solo colores y umbral)
plt.legend(labels=['Transacciones Normales', 'Transacciones Fraudulentas', f'Umbral: {umbral_fraude:.2f} USD'], loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.3)  # Cuadrícula horizontal sutil
plt.tight_layout()
plt.show()