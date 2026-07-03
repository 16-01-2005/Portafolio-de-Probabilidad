# **Guía de Actividades Práctico-Experimentales Nro. 005**

---

## **1. Datos Generales**

| **Atributo** | **Descripción** |
| :--- | :--- |
| **Estudiantes** | Kiara Condoy, Héctor Guerrero, Javier Guarnizo, Ricardo Ochoa, Emily Salas |
| **Asignatura** | Teoría de la Distribución y Probabilidad |
| **Ciclo** | 2do “A” |
| **Unidad** | 1 |
| **Práctica Nro.** | 005 |
| **Título de la Práctica** | **Distribuciones Discretas Notables: Modelado y Simulación de Procesos de Bernoulli y Eventos Raros** |
| **Nombre del Docente** | Cristian Ramiro Narváez Guillén |
| **Fecha** | Martes 12 de abril 2026 |

---


# **Tarea 1: Modelado Computacional de la Distribución Binomial**
---


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parámetros del modelo Binomial
n_ensayos = 20
p_falla = 0.15

# Dominio de la variable aleatoria (0 a n)
x = np.arange(0, n_ensayos + 1)

# Cálculo de PMF (Función de Masa) y CDF (Función Acumulada)
pmf_binomial = binom.pmf(x, n_ensayos, p_falla)
cdf_binomial = binom.cdf(x, n_ensayos, p_falla)

# Visualización
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico PMF
ax[0].vlines(x, 0, pmf_binomial, colors='b', lw=5, alpha=0.5)
ax[0].plot(x, pmf_binomial, 'bo', ms=8)
ax[0].set_title(f'PMF - Binomial (n={n_ensayos}, p={p_falla})')
ax[0].set_xlabel('Número de fallos (x)')
ax[0].set_ylabel('P(X = x)')
ax[0].grid(True, alpha=0.3)

# Gráfico CDF
ax[1].step(x, cdf_binomial, where='post', color='r', lw=2)
ax[1].plot(x, cdf_binomial, 'ro', alpha=0.5)
ax[1].set_title('CDF - Distribución Acumulada')
ax[1].set_xlabel('Número de fallos (x)')
ax[1].set_ylabel('P(X <= x)')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Cálculo de probabilidad específica: P(X <= 3)
prob_max_3 = binom.cdf(3, n_ensayos, p_falla)
print(f"La probabilidad de tener 3 fallos o menos es: {prob_max_3:.4f}")
```


# **Tarea 2: Modelado de la Distribución de Poisson (Eventos Raros)**


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# 1. Parámetros del modelo de Poisson
# La imagen indica una tasa media (lambda) de 4.5
lam = 4.5

# 2. Dominio de la variable aleatoria
# La tarea pide evaluar desde x = 0 hasta x = 15
x = np.arange(0, 16)

# 3. Cálculo de la PMF (Probability Mass Function)
# Usamos poisson.pmf para obtener la probabilidad de cada punto
pmf_poisson = poisson.pmf(x, lam)

# 4. Visualización (Únicamente la PMF como pide la tarea)
plt.figure(figsize=(10, 6))

# Dibujamos las líneas verticales y los puntos
plt.vlines(x, 0, pmf_poisson, colors='b', lw=5, alpha=0.5)
plt.plot(x, pmf_poisson, 'bo', ms=8)

# Configuración de etiquetas y título
# El prefijo 'rf' evita el SyntaxWarning del símbolo \lambda
plt.title(rf'PMF - Distribución de Poisson ($\lambda$ = {lam})')
plt.xlabel('Número de peticiones erróneas (x)')
plt.ylabel('P(X = x)')
plt.xticks(x)
plt.grid(True, alpha=0.3)

plt.show()

# 5. Cálculo de la probabilidad exacta solicitada: P(X = 6)
prob_exacta_6 = poisson.pmf(6, lam)

print("-" * 50)
print(f"RESULTADO DE LA TAREA:")
print(f"La probabilidad exacta de recibir exactamente 6 peticiones es: {prob_exacta_6:.4f}")
print("-" * 50)
```


# **Tarea 3: Hito del Proyecto - Identificación de Variables de Conteo (ABP)**


```python
import pandas as pd
import numpy as np

# 1. Conexión a la nueva hoja del Spreadsheet
url = 'https://docs.google.com/spreadsheets/d/1E-rrTgcTIWS-UnuVf1ldKvhDkwT6uUgh/export?format=xlsx'

try:
    # Nombre exacto de la pestaña
    df = pd.read_excel(url, sheet_name='Subnational 1 tree cover loss')

    # Limpieza básica
    df = df.dropna(how='all').reset_index(drop=True)
    print("¡Conexión exitosa a la hoja de Provincias!")
    display(df.head(50))

except Exception as e:
    print(f"Error: {e}")
```


**1. Identificar una variable discreta de "conteo"** Para este ejemplo, seleccionaremos la columna `threshold` del DataFrame, que representa el umbral de pérdida de cobertura arbórea.


```python
# Identificar la variable de conteo
variable_conteo = 'threshold'
x = df[variable_conteo]

print(f"Variable de conteo seleccionada: '{variable_conteo}'")
print("Estadísticas descriptivas de la variable de conteo:")
display(x.describe())
```


**2. Calcular la media muestral y asumir como parámetro (lambda) para el modelo de Poisson**


```python
# Calcular la media muestral
media_muestral = x.mean()

# Asumir la media muestral como el parámetro lambda para la distribución de Poisson teórica
lambda_poisson = media_muestral

print(f"Media muestral de '{variable_conteo}': {media_muestral:.2f}")
print(f"Parámetro lambda para el modelo de Poisson teórico: {lambda_poisson:.2f}")
```


**3. Generar un gráfico superponiendo el histograma de densidad de la variable empírica contra la PMF teórica de Poisson**


```python
# Rango de valores para la PMF de Poisson
k = np.arange(0, x.max() + 3) # Valores de 0 hasta el máximo de la variable de conteo + un margen

# Calcular la PMF teórica de Poisson
pmf_poisson = poisson.pmf(k, mu=lambda_poisson)

# Crear el gráfico
plt.figure(figsize=(12, 6)) # Aumentar un poco el tamaño de la figura para más espacio

# Histograma de la variable empírica (normalizado para densidad)
plt.hist(x, bins=np.arange(x.min(), x.max() + 2) - 0.5, density=True, alpha=0.7, label='Histograma de Densidad Empírica', color='skyblue', edgecolor='black')

# PMF teórica de Poisson
plt.plot(k, pmf_poisson, 'ro-', label=f'PMF Teórica de Poisson (λ={lambda_poisson:.2f})', markersize=6)

plt.title(f'Distribución Empírica vs. Distribución de Poisson Teórica para {variable_conteo}')
plt.xlabel('Número de Eventos')
plt.ylabel('Densidad / Probabilidad')
plt.xticks(rotation=45, ha='right') # Rotar las etiquetas del eje X 45 grados y alinearlas a la derecha
# plt.xticks(np.arange(0, x.max() + 3, step=5)) # Opcional: Mostrar ticks cada 5 unidades si aún se solapan mucho
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout() # Ajusta automáticamente los parámetros de la subtrama para un diseño ajustado
plt.show()
```


**4. Discusión visual**

Al observar el gráfico, se puede comparar visualmente la forma del histograma de la variable `threshold` con la línea de la función de masa de probabilidad (PMF) de Poisson teórica.

*   **Si el histograma y la línea de la PMF se superponen estrechamente**, esto sugiere que los datos empíricos siguen una distribución de Poisson.
*   **Si hay desviaciones significativas**, por ejemplo, el histograma es más ancho (mayor varianza) o más sesgado de lo que predice la PMF de Poisson, esto indicaría que la distribución de Poisson podría no ser el modelo más adecuado para los datos.

Para tus datos, esta visualización te ayudará a determinar si la suposición de una distribución de Poisson es razonable.


# **Tarea 4: ABI - Aproximación Binomial a Poisson**


La distribución Binomial se aproxima a una distribución de Poisson cuando el número de ensayos \(n\) es muy grande y la probabilidad de éxito \(p\) es muy pequeña, manteniéndose constante el producto.

La razón de esta aproximación es que, cuando existen muchísimos ensayos y cada éxito es raro, el comportamiento probabilístico depende principalmente del número esperado de éxitos, y no del número exacto de intentos.


##**Aproximación de la distribución Binomial a la de Poisson**

Una distribución Binomial se aproxima a una distribución de Poisson (Binomial(n,p)≈Poisson(λ)) cuando se cumplen simultáneamente estas condiciones:

El número de ensayos n es muy grande **n → ∞**

La probabilidad de éxito p es muy pequeña **p → 0**

El producto n⋅p se mantiene constante y finito  **np = λ**

Intuición probabilística:

Sabemos que la Binomial modela: un número fijo de intentos, con éxito o fracaso,
y probabilidad constante p.

La Poisson aparece cuando: hay muchísimos intentos, pero cada éxito individual es extremadamente raro.

En ese caso, deja de importar el número exacto de intentos y solo importa el promedio esperado de éxitos λ = np.



##**Script en Python**


```python
from math import comb, exp, factorial

# Datos
n = 1000
p = 0.003
k = 2

# Parámetro lambda
lam = n * p

# Probabilidad Binomial
P_binomial = comb(n, k) * (p**k) * ((1-p)**(n-k))

# Probabilidad Poisson
P_poisson = (exp(-lam) * (lam**k)) / factorial(k)

# Resultados
print("=== Comparación Binomial vs Poisson ===")
print(f"n = {n}")
print(f"p = {p}")
print(f"λ = {lam}")
print(f"k = {k}")

print("\nProbabilidad Binomial:")
print(f"P(X = {k}) = {P_binomial}")

print("\nProbabilidad Poisson:")
print(f"P(X = {k}) = {P_poisson}")

print("\nDiferencia absoluta:")
print(abs(P_binomial - P_poisson))
```
