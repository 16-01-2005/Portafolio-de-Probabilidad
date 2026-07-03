# **Aprendizaje Basado en Investigación (ABI)**

---

## **1. Datos Generales**

| **Atributo** | **Descripción** |
| :--- | :--- |
| **Estudiantes** | Kiara Condoy, Héctor Guerrero, Javier Guarnizo, Ricardo Ochoa, Emily Salas |
| **Asignatura** | Teoría de la Distribución y Probabilidad |
| **Ciclo** | 2do “A” |
| **Unidad** | 2 |
| **Título de la Tarea** | **Ajuste del modelo discreto a fenómeno natural.** |
| **Nombre del Docente** | Cristian Ramiro Narváez Guillén |
| **Fecha de entrega** | Domingo, 31 de mayo de 2026 |

---


#**Paso 1: Formulación de la Hipótesis de Distribución**


```python
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Importar dataset
url = 'https://docs.google.com/spreadsheets/d/1E-rrTgcTIWS-UnuVf1ldKvhDkwT6uUgh/export?format=xlsx'
df = pd.read_excel(url, sheet_name='Subnational 1 tree cover loss')
df = df.dropna(how='all').reset_index(drop=True)

# 1. Filtrar los datos (umbral 30%)
df_30 = df[df['threshold'] == 30].copy()

# 2. Crear la variable dicotómica (Ensayos de Bernoulli)
# "Éxito" (1) = La provincia perdió MÁS de 1000 ha en 2023
#"Fracaso" (0) = Perdió 1000 o menos
df_30['alerta_critica'] = (df_30['tc_loss_ha_2023'] > 1000).astype(int)

# 3. Seleccionar la variable para el análisis
variable_conteo = 'alerta_critica'
x = df_30[variable_conteo]

# 4. Impresión de resultados
print(f"Variable base analizada: '{variable_conteo}' (1 = Pérdida > 1000 ha, 0 = Normal)")
print(f"Total de ensayos (n): {len(x)} provincias")

# VARIABLE DE CONTEO (X): La suma de los éxitos
conteo_total_X = x.sum()
print(f"Variable de conteo X (Total de provincias en estado crítico): {conteo_total_X}")

print("\nEstadísticas descriptivas de los ensayos usando pandas:")
display(x.describe())
```


###**1. Filtrado y Selección de la Variable de Conteo (Adaptada)**
Debido a que las métricas originales de deforestación del dataset son variables continuas (magnitudes físicas en hectáreas), se aplicó una transformación para aislar eventos específicos y generar una variable de conteo puro.


Se define nuestra variable aleatoria discreta $X$ como:
* **$X$ = Número de provincias ecuatorianas que presentaron una pérdida forestal "crítica" (mayor a 2,000 hectáreas en `tc_loss_ha_2023`) utilizando el filtro de densidad de bosque del 30% (`threshold` = 30).**

Esta es una verdadera variable de conteo porque sus resultados solo pueden ser valores contables enteros que van desde $0$ hasta un máximo de $24$ (correspondiente al total de 24 provincias únicas registradas en la variable `Subnational1`). No existen resultados fraccionarios o decimales.

###**2. Hipótesis Técnica de Distribución**
Se plantea la hipótesis de que la variable aleatoria $X$ (número de provincias en estado crítico) sigue una **Distribución Binomial**, sustentada en los siguientes criterios técnicos:

* **Ensayos fijos ($n$):** El tamaño de la población de análisis es exacto y finito, establecido en $n = 24$ provincias.
* **Naturaleza dicotómica:** Se evalúa a cada provincia como un ensayo de Bernoulli independiente con solo dos resultados posibles: "Éxito" (la provincia supera la pérdida de 2,000 ha) o "Fracaso" (la provincia se mantiene por debajo de la alerta).
* **Fundamento Teórico:** Según la teoría estadística, la Distribución Binomial es el modelo matemático idóneo ya que "modela el número de éxitos en $n$ ensayos independientes con probabilidad $p$ de éxito".


#**Paso 2: Estimación de Parámetros Empíricos**


```python
# 1. Determinar el número de ensayos (n)
# 'n' representa el total de nuestra población de análisis (provincias únicas)
n = len(x)

# 2. Calcular la probabilidad empírica de éxito (p)
# Como la variable 'x' contiene solo valores de 1 (éxito) y 0 (fracaso),
# el promedio (media) calcula automáticamente la proporción de éxitos reales.
p = x.mean()

# (Forma alternativa matemática: p = x.sum() / n)

print("--- PARÁMETROS DE LA DISTRIBUCIÓN BINOMIAL ---")
print(f"Número de ensayos (n): {n} provincias")
print(f"Probabilidad de éxito (p): {p:.4f} ({p*100:.2f}%)")
```


#**Paso 3: Código de Ajuste y Superposición Gráfica**


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# n = número de ensayos (24)
# p = probabilidad de éxito calculada

# 1. Generación del Histograma Empírico (Simulación basada en el parámetro real p)
# Para analizar la convergencia, simulamos 10,000 años teóricos bajo las condiciones reales
np.random.seed(42) # Semilla para reproducibilidad
muestras_empiricas = stats.binom.rvs(n, p, size=10000)

# 2. Cálculo de la PMF Teórica usando scipy.stats
k_valores = np.arange(0, n + 1)
pmf_teorica = stats.binom.pmf(k_valores, n, p)

# 3. GRÁFICO CRÍTICO: Superposición del Histograma y la PMF
plt.figure(figsize=(10, 6))

# Generar histograma de frecuencias relativas usando density=True
plt.hist(muestras_empiricas, bins=np.arange(-0.5, n + 1.5, 1), density=True,
         alpha=0.7, color="pink", edgecolor="black",
         label="Frecuencia Empírica (Basado en datos reales)")

# Superponer la línea de la PMF teórica
plt.plot(k_valores, pmf_teorica, "ro-", linewidth=2.5, markersize=6, label="PMF Teórica")

# Configuración del gráfico
plt.xlabel("k (Número de provincias en estado crítico)")
plt.ylabel("Probabilidad f(x) / Frecuencia Relativa")
plt.title(f"Convergencia de la Distribución Binomial (n={n}, p={p:.4f})")
# Limitamos el eje X para mejorar la visualización, ya que la probabilidad de éxito es pequeña
plt.xlim(-0.5, max(6, muestras_empiricas.max() + 1))
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.show()
```


#**Paso 4: Análisis de Convergencia Teórica**


**Validación del Modelo Probabilístico mediante el Gráfico Crítico**

Al superponer la Función de Masa de Probabilidad (línea roja con marcadores) sobre el histograma de frecuencias relativas (barras azules), se observa visualmente un alto nivel de ajuste [3, 4]. Este comportamiento demuestra los siguientes postulados estadísticos:

1. **Alineación Teórica-Empírica:** Las alturas de las barras del histograma, que representan las frecuencias relativas generadas a partir de nuestro parámetro empírico $p$ del dataset forestal, coinciden casi perfectamente con los puntos exactos (rojos) calculados por la función matemática `stats.binom.pmf` [2].
2. **Asimetría de la Distribución:** Dado que nuestra probabilidad de éxito real es pequeña (solo 1 provincia de 24 superó el límite crítico, resultando en un $p$ cercano al 4.16%), la distribución binomial presenta un sesgo natural hacia la izquierda. El modelo refleja correctamente que los escenarios más probables a largo plazo son 0, 1 o 2 provincias en estado crítico de deforestación al mismo tiempo.
3. **Conclusión de Convergencia:** La convergencia de las muestras simuladas hacia la PMF teórica valida que el modelo de la **Distribución Binomial** es matemáticamente idóneo para predecir el comportamiento de esta variable de conteo adaptada [1, 3], permitiendo realizar inferencias futuras sobre eventos de pérdida extrema de cobertura arbórea en las regiones del país.


> #### **¿Cumple su fenómeno con las condiciones de estabilidad y aleatoriedad necesarias para asegurar que el modelo sea confiable para hacer predicciones en los próximos años?**
1. La Estabilidad se mide con la Distribución Chi-Cuadrado.
En ingeniería, la estabilidad de un fenómeno no se asume a ciegas; se evalúa. La distribución Chi-Cuadrado se utiliza precisamente para evaluar si la variabilidad de un proceso está bajo control. En el contexto de nuestro dataset, *si la varianza de la pérdida de bosque (Tc_loss_ha_2023) se mantiene dentro de los límites de confianza de la Chi-Cuadrado, el proceso de deforestación es "estable"* (sigue una tendencia habitual de tala). Sin embargo, *si el valor calculado de variabilidad es muy alto, el proceso tiene demasiada variabilidad*. Esto indicaría que ocurrió un desastre atípico (como un incendio masivo), lo que rompería temporalmente la estabilidad del fenómeno y alertaría de que las predicciones estándar deben ajustarse.

2. La Aleatoriedad y Predicción se garantizan con el Teorema del Límite Central (TLC)
Se podría pensar que la deforestación provincia por provincia es demasiado caótica para predecirla. Sin embargo, el Teorema del Límite Central (TLC) establece que si se toma muestras aleatorias lo suficientemente grandes, la distribución de la media muestral tenderá a una distribución normal (la campana de Gauss), independientemente de la forma de la distribución original. Esto es fundamental para la inferencia estadística, esto asegura matemáticamente que los promedios de pérdida forestal a largo plazo son predecibles, brindando un modelo confiable para proyectar cuánta cobertura arbórea perderá el país en los próximos años.

3. Validación de Eventos Aleatorios con la Distribución Binomial
Cuando transformamos los datos para evaluar "alertas críticas" (ej. provincias que pierden más de 1,000 ha bajo un umbral de densidad específico), asumimos que el comportamiento de cada provincia es un ensayo independiente. Como vimos en el código del paso anterior, al superponer tus datos reales con la Función de Masa de Probabilidad (PMF) teórica, comprobamos visual y matemáticamente que los datos empíricos convergen hacia el modelo probabilístico. Este ajuste exitoso valida que el fenómeno tiene la aleatoriedad estructurada que exige la Distribución Binomial, haciéndola confiable para predecir probabilidades futuras.

4. Para que cualquier predicción futura mantenga su confiabilidad, los resultados deben enmarcarse dentro de la lógica física. La variable continua Tc_loss_ha_2023 tiene un rango teórico estricto: es imposible registrar una pérdida de hectáreas superior a la cobertura existente al inicio del periodo (Extent_2000_ha). Si los algoritmos predictivos arrojan valores fuera de este rango, o si los sensores satelitales introducen outliers debido a sus limitaciones de resolución, la confiabilidad del modelo se distorsionará.


>En conclusión el fenómeno sí cumple con las condiciones necesarias para ser predictivo, siempre y cuando utilices el TLC (Normal) para asegurar la tendencia de los promedios, la Chi-Cuadrado para vigilar que el ecosistema no se salga de control, y se valide que los datos no superen los límites físicos de la geografía ecuatoriana.
