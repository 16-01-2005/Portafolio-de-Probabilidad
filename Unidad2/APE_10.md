# **Guía de Actividades Práctico-Experimentales Nro. 010**

---

## **1. Datos Generales**

| **Atributo** | **Descripción** |
| :--- | :--- |
| **Estudiantes** | Kiara Condoy, Héctor Guerrero, Javier Guarnizo, Ricardo Ochoa, Emily Salas |
| **Asignatura** | Teoría de la Distribución y Probabilidad |
| **Ciclo** | 2do “A” |
| **Unidad** | 1 |
| **Práctica Nro.** | 010 |
| **Título de la Práctica** | **Distribuciones Discretas Notables: Modelado y Simulación de Procesos de Bernoulli y Eventos Raros** |
| **Nombre del Docente** | Cristian Ramiro Narváez Guillén |
| **Fecha** | Lunes 22 de junio 2026 |

---


# **Tarea 1: Prueba de Hipótesis para Dos Muestras Independientes (A/B Testing)**
---


```python
# @title
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Datos empíricos (Consumo de RAM en MB)
np.random.seed(42)
memoria_algo_A = np.random.normal(loc=120.5, scale=12.0, size=35)
memoria_algo_B = np.random.normal(loc=128.2, scale=15.0, size=40)

alpha = 0.05

# Ejecución de la prueba T para muestras independientes (2 colas)
stat_ind, p_val_ind = ttest_ind(memoria_algo_A, memoria_algo_B, equal_var=True)

print("--- A/B Testing: Algoritmo A vs Algoritmo B ---")
print(f"Media Algo A: {np.mean(memoria_algo_A):.2f} MB | Media Algo B: {np.mean(memoria_algo_B):.2f} MB")
print(f"Estadístico T: {stat_ind:.4f}")
print(f"Valor-p: {p_val_ind:.4e}")

if p_val_ind < alpha:
    print("Conclusión: Se RECHAZA H0. Existe una diferencia significativa en el consumo de memoria.")
else:
    print("Conclusión: NO se rechaza H0. No hay evidencia de que los algoritmos difieran en consumo.")

# Visualización
plt.figure(figsize=(8, 4))
sns.kdeplot(memoria_algo_A, fill=True, label="Algoritmo A")
sns.kdeplot(memoria_algo_B, fill=True, label="Algoritmo B")
plt.title("Comparación de Distribuciones de Consumo de RAM")
plt.xlabel("Consumo (MB)")
plt.legend()
plt.show()
```


# **Tarea 2: Prueba de Hipótesis para Muestras Pareadas (Dependientes)**


```python
from scipy.stats import ttest_rel

# Latencia en ms (las posiciones en los arrays corresponden al mismo servidor)
latencia_antes = np.array([45, 52, 48, 55, 60, 42, 49, 58, 51, 46, 50, 47, 53, 59, 44])
latencia_despues = np.array([41, 50, 45, 50, 56, 40, 46, 53, 48, 42, 47, 45, 51, 55, 40])

# Prueba pareada (H1: la latencia "después" es menor, por tanto (antes - despues) > 0)
# Para evaluar si "después" bajó, usamos una prueba de cola superior sobre las diferencias (antes - despues)
# SciPy modernos permiten alternative='greater' en ttest_rel
stat_rel, p_val_rel = ttest_rel(latencia_antes, latencia_despues, alternative='greater')

print("\n--- Análisis Pareado: Impacto del Nuevo Firewall ---")
print(f"Media Diferencias (Antes - Después): {np.mean(latencia_antes - latencia_despues):.2f} ms")
print(f"Estadístico T Pareado: {stat_rel:.4f}")
print(f"Valor-p (1 cola): {p_val_rel:.4e}")

if p_val_rel < 0.05:
    print("Conclusión: Se RECHAZA H0. El firewall redujo significativamente la latencia.")
else:
    print("Conclusión: NO se rechaza H0. El firewall no mejoró la latencia.")
```
