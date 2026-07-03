# **Guía de Actividades Práctico-Experimentales Nro. 004**

---

## **1. Datos Generales**

| **Atributo** | **Descripción** |
| :--- | :--- |
| **Estudiantes** | Luis Armijos, Kiara Condoy, Héctor Guerrero, Javier Guarnizo, Ricardo Ochoa, Emily Salas |
| **Asignatura** | Teoría de la Distribución y Probabilidad |
| **Ciclo** | 2do “A” |
| **Unidad** | 1 |
| **Resultado de aprendizaje** | Calcula probabilidades y momentos de variables aleatorias discretas, bajo los principios de solidaridad, transparencia, responsabilidad y honestidad. |
| **Práctica Nro.** | 004 |
| **Título de la Práctica** | **Momentos Estadísticos:** Esperanza Matemática, Varianza y Análisis de Tendencia Central con Python |
| **Nombre del Docente** | Cristian Ramiro Narváez Guillén |
| **Fecha** | Martes 05 de abril 2026 |
| **Horario** | 11h30 – 13h30 |
| **Lugar** | EVA / Aula |
| **Tiempo planificado** | 2 horas |

---


# **Tarea 1: Validación de Clase Invertida**


## **Distribución adicional: Binomial**
### **Lanzar un dado 10 veces y calcular la probabilidad de obtener exactamente 3 veces el número 6.**
### _Representación gráfica_

### **Justificación de esta elección**
La elección de este trabajo se realizó entre un conjunto de propuestas, por su estilo claro y ordenado. Se caracteriza por ofrecer una representación gráfica y analítica que facilita la comprensión de los conceptos, además de incluir un ejemplo práctico que permite entender de manera accesible la distribución binomial. Estas cualidades lo convierten en una opción adecuada para fortalecer tanto el aprendizaje teórico como la aplicación práctica de la estadística.


```python
# Parámetros de la distribucion binomial
n = 10          # numero de lanzamientos
p = 1/6         # probabilidad de sacar 6

# Valores posibles (0 a 10 veces que puede salir 6)
x = np.arange(0, n+1)

# PMF (probabilidad exacta)
pmf_binomial = binom.pmf(x, n, p)

# CDF (probabilidad acumulada)
cdf_binomial = binom.cdf(x, n, p)

# Visualización
plt.figure(figsize=(10, 5))

# PMF
plt.subplot(1, 2, 1)
plt.stem(x, pmf_binomial, basefmt=" ")
plt.xlabel('k (numero de veces que sale 6)')
plt.ylabel('P(X = k)')
plt.title('PMF: Distribucion Binomial (n=10, p=1/6)')
plt.grid(alpha=0.3)

# CDF
plt.subplot(1, 2, 2)
plt.step(x, cdf_binomial, where='post')
plt.xlabel('k')
plt.ylabel('P(X <= k)')
plt.title('CDF: Distribucion Binomial')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```


### _Representación analítica_


```python
# Cálculo de Probabilidades
# a) P(X = 3)
prob_3 = binom.pmf(3, n, p)
print(f"P(X = 3) = {prob_3:.4f} ({prob_3*100:.2f}%)")

# b) P(X <= 3)
prob_menor_igual_3 = binom.cdf(3, n, p)
print(f"P(X <= 3) = {prob_menor_igual_3:.4f} ({prob_menor_igual_3*100:.2f}%)")

# c) P(X > 3)
prob_mayor_3 = 1 - binom.cdf(3, n, p)
print(f"P(X > 3) = {prob_mayor_3:.4f} ({prob_mayor_3*100:.2f}%)")

# d) P(2 <= X <= 5)
prob_rango = binom.cdf(5, n, p) - binom.cdf(2-1, n, p)
print(f"P(2 <= X <= 5) = {prob_rango:.4f} ({prob_rango*100:.2f}%)")
```


# **Tarea 2: Cálculo Teórico y Simulación de Esperanza Matemática**


```python
import numpy as np
from scipy.stats import binom, norm
# 1. Variable Aleatoria Discreta (Distribución Binomial)
n_ensayos, p_exito = 10, 0.4
var_discreta = binom(n_ensayos, p_exito)
# Cálculo de Momentos Teóricos (Mean, Variance)
esperanza_d, varianza_d = var_discreta.stats(moments='mv')
print(f"--- Variable Discreta (Binomial n={n_ensayos}, p={p_exito}) ---")
print(f"Esperanza E[X]: {esperanza_d}")
print(f"Varianza V[X]: {varianza_d}\n")
# 2. Variable Aleatoria Continua (Distribución Normal)
mu, sigma = 50, 5  # Media y Desviación Estándar
var_continua = norm(loc=mu, scale=sigma)
esperanza_c, varianza_c = var_continua.stats(moments='mv')
print(f"--- Variable Continua (Normal mu={mu}, sigma={sigma}) ---")
print(f"Esperanza E[X]: {esperanza_c}")
print(f"Varianza V[X]: {varianza_c}")
```


# **Tarea 3: Hito del Proyecto - Análisis de Tendencia Central y Dispersión**

## **Cargue su dataset regional utilizando pandas**


```python
import pandas as pd

url = 'https://docs.google.com/spreadsheets/d/1E-rrTgcTIWS-UnuVf1ldKvhDkwT6uUgh/export?format=xlsx'

try:
    df = pd.read_excel(url, sheet_name='Country tree cover loss')

    print("¡Conexión exitosa con la pestaña correcta!")


    df = df.dropna(how='all').reset_index(drop=True)

    display(df.head())
except Exception as e:
    print(f"Error: {e}")
```


## **Desarrolle un script que calcule la media muestral y la varianza muestral.**
## **Utilice la formulación insesgada donde la suma de cuadrados se divide por**


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Carga de datos reales
df_regional = pd.read_excel(url, sheet_name='Country tree cover loss')

# 2. Selección de variables (años de pérdida de cobertura)
columnas_loss = [
    'tc_loss_ha_2001',
    'tc_loss_ha_2002',
    'tc_loss_ha_2003',
    'tc_loss_ha_2004',
    'tc_loss_ha_2005'
]

# 3. Limpieza y transformación (unir todos los años en una sola variable)
data_clean = df_regional[columnas_loss].values.flatten()
data_clean = pd.Series(data_clean).dropna()

# 4. Cálculo de estadísticos
media_muestral = data_clean.mean()
varianza_muestral = data_clean.var(ddof=1)

print("Análisis de Pérdida de Cobertura Forestal (2001–2005)")
print(f"Media Muestral: {media_muestral:.2f}")
print(f"Varianza Muestral: {varianza_muestral:.2f}")

# 5. Visualización
plt.figure(figsize=(8, 5))

sns.histplot(data_clean, kde=True, color='forestgreen', stat='density')

plt.axvline(media_muestral, color='red', linestyle='--',
            label=f'Media: {media_muestral:.2f}')

plt.title('Distribución de Pérdida de Cobertura Forestal (ha)')
plt.xlabel('Hectáreas perdidas (ha)')
plt.ylabel('Densidad')
plt.legend()

plt.show()
```


# **Tarea 4: Justificación estadística del parámetro ddof=1**

En el análisis de pérdida de cobertura forestal, los datos disponibles
corresponden a una *muestra* del comportamiento real del ecosistema ecuatoriano,
no a la totalidad de los registros históricos posibles. Por ello, al calcular
la varianza con data_clean.var(ddof=1), se emplea la *corrección de Bessel*,
que divide la suma de desviaciones cuadradas entre $n - 1$ en lugar de $n$.

Esta corrección es estadísticamente necesaria porque el estimador que usa $n$
en el denominador es *sesgado*: en promedio, subestima la verdadera varianza
poblacional $\sigma^2$. Al utilizar $n - 1$ como denominador, se obtiene un
*estimador insesgado* $S^2$, lo que significa que $E[S^2] = \sigma^2$, es
decir, el valor esperado del estadístico coincide con el parámetro poblacional
que se desea estimar. El grado de libertad "perdido" se debe a que la media
muestral $\bar{x}$ ya fue calculada a partir de los mismos datos, restringiendo
la variabilidad libre en la muestra.


## **Preguntas de Control**

---

### **1. Diferencia entre Esperanza Teórica y Media Muestral**
* **¿Cuál es la diferencia matemática y conceptual entre la esperanza matemática teórica calculada a partir de un modelo de probabilidad (ej. `binom.stats()`) y la media muestral calculada de un DataFrame de pandas?**

    > **Respuesta:** La esperanza matemática teórica es el valor promedio ideal que predice un modelo de probabilidad (parámetro poblacional), mientras que la media muestral es el promedio real observado en un conjunto de datos finito (estadístico). La primera es un valor fijo basado en la distribución, y la segunda es una aproximación que, según la Ley de los Grandes Números, tiende a la esperanza teórica conforme aumenta el tamaño de la muestra.

---

### **2. Demostración Matemática de la Varianza**
* **Demuestre teóricamente, utilizando las propiedades de la esperanza, por qué la varianza se puede reescribir como:**
$$V[X] = E[X^2] - (E[X])^2$$

    > **Demostración:**
    > La varianza se define como: $\text{Var}(X) = E[(X - E[X])^2]$
    > 1. **Expandimos el binomio:** $E[X^2 - 2X E[X] + (E[X])^2]$
    > 2. **Distribuimos el operador esperanza ($E$):** $\text{Var}(X) = E[X^2] - E[2X E[X]] + E[(E[X])^2]$
    > 3. **Extraemos las constantes ($E[X]$ es constante):** $\text{Var}(X) = E[X^2] - 2E[X]E[X] + (E[X])^2$
    > 4. **Simplificamos términos:** $\text{Var}(X) = E[X^2] - 2(E[X])^2 + (E[X])^2 = E[X^2] - (E[X])^2$

---

### **3. Implicaciones de una Varianza Alta**
* **Si la varianza calculada en su variable regional es inusualmente alta, ¿qué implicaciones prácticas tiene esto sobre la confiabilidad de la media como predictor del comportamiento de esos datos en la región de Loja?**

    > **Análisis:** Una varianza alta indica que los datos están muy dispersos respecto al centro. Prácticamente, esto implica una **pérdida de capacidad predictiva**, ya que la media deja de ser un valor representativo al estar influenciada por valores extremos. En Loja, esto sugeriría que la pérdida de cobertura no es constante, sino que existen años o zonas con comportamientos drásticamente distintos que el promedio no logra capturar.

---

### **4. Ajuste de Grados de Libertad en Pandas (`ddof`)**
* **Revise el parámetro `ddof` de la función `var()` en Pandas. ¿Qué ocurre con el estimador de la varianza si establecemos `ddof=0` y en qué escenario específico sería matemáticamente correcto?**

    > **Respuesta:** > * **Varianza Poblacional (`ddof=0`):** Divide la suma de cuadrados por $N$. Es correcto solo cuando se analizan los datos de la **población total**.
    > * **Varianza Muestral (`ddof=1`):** Es el estándar en Pandas; divide por $N-1$ para corregir el sesgo y obtener un estimador insesgado de la población a partir de una muestra.
    > * **Efecto:** Usar `ddof=0` en una muestra tiende a subestimar la variabilidad real (produce un sesgo a la baja).

---

### **5. Identificación de Outliers mediante Media y Varianza**
* **Observe el histograma generado en la Tarea 3. ¿De qué manera el cálculo combinado de la esperanza matemática (media) y la varianza apoyan en la identificación estadística de valores atípicos (*outliers*)?**

    > **Análisis:** La media y la varianza permiten establecer "umbrales de normalidad". Según la regla empírica, datos que se alejan más de 2 o 3 desviaciones estándar ($\sigma = \sqrt{V}$) de la media suelen considerarse *outliers*. En el gráfico, la media ($32,873.32$) se ve desplazada hacia la izquierda por los valores bajos (rango de $15,000$ ha), lo que demuestra cómo estos valores atípicos incrementan la varianza y sesgan el promedio lejos de la zona de mayor frecuencia (los $40,000$ ha).
