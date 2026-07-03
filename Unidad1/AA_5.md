# **Guía de Actividades Práctico-Experimentales Nro. 005**

---

## **1. Datos Generales**

| **Atributo** | **Descripción** |
| :--- | :--- |
| **Estudiantes** | Luis Armijos, Kiara Condoy, Héctor Guerrero, Javier Guarnizo, Ricardo Ochoa, Emily Salas |
| **Asignatura** | Teoría de la Distribución y Probabilidad |
| **Ciclo** | 2do “A” |
| **Unidad** | 1 |
| **Práctica Nro.** | 005 |
| **Título de la Práctica** | **Clasificar las variables del proyecto en discretas o continuas y definir el rango de valores posibles** |
| **Nombre del Docente** | Cristian Ramiro Narváez Guillén |
| **Fecha** | Domingo 10 de abril 2026 |

---


# **Fase 1: Investigación y Clasificación Teórica (ABI)**
---


## **Variables Relevantes para el proyecto**
### **1. Subnational1 (Provincia o subdivisión regional)**

La variable Subnational1 es de naturaleza discreta y se clasifica específicamente como una variable categórica nominal. La variable representa divisiones administrativas o provincias dentro del territorio analizado, como por ejemplo Loja, Azuay o Zamora Chinchipe.

### **2.	Area_ha (Área total)**

La variable Area_ha es de naturaleza continua, ya que representa una medición cuantitativa de la superficie física total de una unidad geográfica expresada en hectáreas.
Este tipo de variable puede tomar cualquier valor dentro de un intervalo, incluyendo valores decimales, dependiendo de la precisión de la medición.

Se clasifica como continua porque proviene de un proceso de medición física del espacio geográfico y no de un conteo de elementos discretos.


### **3. Extent_2000_ha (Extensión en el año 2000)**

La variable es continua, ya que representa la superficie de cobertura forestal en el año 2000 medida en hectáreas. Esta variable surge de estimaciones espaciales derivadas de imágenes satelitales y modelos de cobertura terrestre.

Al ser una magnitud física, puede tomar valores decimales y cualquier valor dentro de un rango continuo, dependiendo de la resolución del análisis geoespacial.


### **4. Tc_loss_ha_2024 (Pérdida de cobertura arbórea)**

Esta variable es continua, ya que representa la cantidad de superficie perdida de cobertura arbórea durante el año 2024, expresada en hectáreas.

Esta variable proviene de mediciones espaciales derivadas de observaciones satelitales, por lo que puede tomar valores decimales y cualquier valor dentro de un intervalo continuo.


### **5. Threshold (Umbral)**

Esta variable es de naturaleza discreta y, específicamente, se clasifica como ordinal. En el contexto de los datos forestales, representa niveles porcentuales predefinidos de cobertura del dosel arbóreo (por ejemplo: 10%, 15%, 20%, 25%, 30%, 50% y 75%).

No es una escala infinita, sino un conjunto de valores finitos seleccionados para actuar como filtros. Su propósito es definir el criterio técnico de lo que se considera "bosque" para una región específica antes de realizar los cálculos de área.

## **Fundamentación teórica de la definición de variables**

- **Variables discretas**

Según Walpole et al. (2012), una variable discreta es aquella cuyos valores posibles forman un conjunto contable y separado. Además, las variables ordinales corresponden a categorías que poseen un orden natural entre sus valores, aunque las diferencias entre ellos no necesariamente sean uniformes.

- **Variables continuas**

Según Walpole et al. (2012), las variables continuas son aquellas derivadas de mediciones físicas que no están restringidas a valores enteros, sino que pueden variar de forma infinitesimal dentro de un rango.


## **Bibliografía**

Walpole, R. E., Myers, R. H., Myers, S. L., & Ye, K. (2012). Probabilidad y estadística para ingeniería y ciencias (9.ª ed.). Pearson Education.


---
 # **Fase 2: Análisis de Rangos y Dominios con Python**
 ---
 ## Análisis de Rangos y Estadísticas Descriptivas

* Utilicen el método `.describe()` y funciones de `numpy` para identificar los valores mínimos y máximos observados en su conjunto de datos.
* **Definición de rangos:** Para cada variable, establezcan dos tipos de rangos:

1. **Rango observado:** El valor mínimo y el máximo reales presentes en sus datos actuales.
2. **Rango Teórico/Lógico:** El rango de valores que la variable *podría* tomar en la realidad (p. ej., la temperatura en Loja no puede ser de 500 °C aunque el sensor falle).

# Análisis Estadístico y Definición de Rangos

En esta sección aplicamos el análisis descriptivo para validar la integridad de los datos frente a su naturaleza teórica.

## 1. Identificación de Estadísticos con Pandas y Numpy

Utilizamos `.describe()` para obtener una visión general y `numpy` para precisiones puntuales sobre los valores extremos.


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


```python
import numpy as np

# Definimos las variables numéricas y la categórica
vars_numericas = ['area_ha', 'extent_2000_ha', 'tc_loss_ha_2023', 'threshold']
var_categorica = 'subnational1'

# 1. Análisis de la variable categórica (Subnational1)
print("--- Análisis de Variable Categórica ---")
print(f"Número de provincias únicas: {df[var_categorica].nunique()}")
print("Listado de provincias encontradas:")
print(df[var_categorica].unique())

# 2. Resumen estadístico de las numéricas
print("\n--- Resumen Estadístico de Variables Continuas ---")
display(df[vars_numericas].describe())

# 3. Rangos observados con Numpy
print("\n--- Rangos Observados Identificados ---")
for col in vars_numericas:
    min_val = np.min(df[col])
    max_val = np.max(df[col])
    print(f"{col:15} -> Mín: {min_val:>12} | Máx: {max_val:>12}")
```


## **Fase 2: Análisis de Rangos y Dominios con Python**

En esta sección, se presentan los límites detectados en el conjunto de datos y se contrastan con los límites lógicos fundamentados en la naturaleza de cada variable.

| Variable | Naturaleza | Rango Observado (Mín - Máx) | Rango Teórico / Lógico | Justificación del Rango Teórico |
| :--- | :--- | :--- | :--- | :--- |
| **subnational1** | Discreta (Nominal) | Azuay - Zamora Chinchipe | 24 provincias oficiales del Ecuador | Corresponde a las divisiones políticas administrativas vigentes del país. |
| **threshold** | Discreta (Ordinal) | 0 - 75 | {0, 10, 15, 20, 25, 30, 50, 75} | Valores fijos establecidos por el protocolo técnico para clasificar la densidad del dosel arbóreo. |
| **area_ha** | Continua | *Según resultado de código* | > 0 hasta ~4,000,000 ha | La superficie de una provincia individual no puede ser negativa ni superar el tamaño de la región más extensa. |
| **extent_2000_ha** | Continua | *Según resultado de código* | 0 a [area_ha] | La cobertura forestal inicial está físicamente limitada por el área total de la provincia. |
| **tc_loss_ha_2023**| Continua | *Según resultado de código* | 0 a [extent_2000_ha] | No es posible registrar una pérdida de hectáreas superior a la cobertura existente al inicio del periodo. |

---


> **Nota de validación:** El contraste entre el **Rango Observado** (extraído con `.describe()` y `numpy`) y el **Rango Teórico** permite asegurar la calidad de la información. Si los valores observados exceden los lógicos, se identifica una anomalía o error en la fuente de datos.

### Resumen de la tabla:
* **Variable Categórica:** Se incluyó `subnational1` (discreta nominal) con un rango lógico basado en las provincias del Ecuador.
* **Variables Continuas:** Se ajustaron los rangos teóricos de `area_ha` y `extent_ha` para reflejar que ahora estás analizando provincias individuales y no el total nacional.
* **Consistencia:** Los nombres de las variables coinciden con el código que ya corregimos para evitar errores de sintaxis en Colab.


---
# **Fase 3: Identificación de Espacios Muestrales**
---


Se identificaron como **variables discretas** en el dataset:
- `subnational1`: Variable categórica nominal (provincias/regiones)
- `threshold`: Variable ordinal (niveles de cobertura de dosel arbóreo en %)

Para estas variables se utiliza la librería `itertools` y el método `.unique()` de Pandas
para listar y simular el espacio muestral posible.


```python
import pandas as pd
import itertools

# ──────────────────────────────────────────────
# Carga del dataset
# ──────────────────────────────────────────────
url = 'https://docs.google.com/spreadsheets/d/1E-rrTgcTIWS-UnuVf1ldKvhDkwT6uUgh/export?format=xlsx'
df = pd.read_excel(url, sheet_name='Subnational 1 tree cover loss') # MODIFICADO: Cambiado a la hoja 'Subnational 1 tree cover loss'
df = df.dropna(how='all').reset_index(drop=True)

print("Dataset cargado correctamente de la hoja 'Subnational 1 tree cover loss'.")
print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
```


---
## 3.1 Variable Discreta: `subnational1` (Provincia/Subdivisión Regional)

**Clasificación:** Discreta – Categórica Nominal  
**Método:** `.unique()` de Pandas para listar el espacio muestral real del dataset.


```python
# ──────────────────────────────────────────────
# Espacio muestral de subnational1 con .unique()
# ──────────────────────────────────────────────
espacio_subnational = df['subnational1'].unique()
espacio_subnational_sorted = sorted(espacio_subnational)

print("═" * 55)
print("  ESPACIO MUESTRAL — subnational1 (Provincias/Regiones)")
print("═" * 55)
print(f"  Total de valores únicos (|Ω|): {len(espacio_subnational_sorted)}")
print("-" * 55)
for i, provincia in enumerate(espacio_subnational_sorted, start=1):
    print(f"  {i:>3}. {provincia}")
print("═" * 55)
```


---
## 3.2 Variable Discreta: `threshold` (Umbral de Cobertura de Dosel)

**Clasificación:** Discreta – Ordinal  
**Método 1:** `.unique()` de Pandas para listar los valores reales presentes en el dataset.  
**Método 2:** `itertools` para simular combinaciones posibles del espacio muestral.


```python
# ──────────────────────────────────────────────────────────
# Método 1: Espacio muestral de threshold con .unique()
# ──────────────────────────────────────────────────────────
espacio_threshold = sorted(df['threshold'].unique())

print("═" * 55)
print("  ESPACIO MUESTRAL — threshold (Umbral %)")
print("═" * 55)
print(f"  Ω = {set(espacio_threshold)}")
print(f"  Total de valores únicos (|Ω|): {len(espacio_threshold)}")
print("-" * 55)
for val in espacio_threshold:
    print(f"  Threshold = {val}%")
print("═" * 55)
```


```python
# ──────────────────────────────────────────────────────────────────────
# Método 2: Combinaciones del espacio muestral con itertools
#
# Simulación de pares de umbrales posibles (combinaciones de 2)
# Útil para analizar escenarios donde se comparan dos niveles de umbral
# ──────────────────────────────────────────────────────────────────────
combinaciones_pares = list(itertools.combinations(espacio_threshold, 2))

print("═" * 55)
print("  COMBINACIONES DE PARES DE UMBRALES (itertools)")
print("═" * 55)
print(f"  Fórmula: C({len(espacio_threshold)}, 2) = {len(combinaciones_pares)} combinaciones")
print("-" * 55)
for i, par in enumerate(combinaciones_pares, start=1):
    print(f"  {i:>2}. Umbral A = {par[0]}%  →  Umbral B = {par[1]}%")
print("═" * 55)
```


```python
import pandas as pd
import itertools

# Ensure df is loaded (though it should be from previous cells, good practice for self-contained execution)
# If df is not in the kernel, uncomment the following lines:
# url = 'https://docs.google.com/sheets/d/1E-rrTgcTIWS-UnuVf1ldKvhDkwT6uUgh/export?format=xlsx'
# df = pd.read_excel(url, sheet_name='Subnational 1 tree cover loss')
# df = df.dropna(how='all').reset_index(drop=True)

# Redefine espacio_subnational_sorted and espacio_threshold to ensure they are available
espacio_subnational = df['subnational1'].unique()
espacio_subnational_sorted = sorted(espacio_subnational)
espacio_threshold = sorted(df['threshold'].unique())

# ──────────────────────────────────────────────────────────────────────
# Producto cartesiano: subnational1 × threshold
# Espacio muestral conjunto de ambas variables discretas
# ──────────────────────────────────────────────────────────────────────
espacio_conjunto = list(itertools.product(espacio_subnational_sorted, espacio_threshold))

print("═" * 70)
print("  ESPACIO MUESTRAL CONJUNTO — subnational1 × threshold")
print("═" * 70)
print(f"  |Ω_conjunto| = |subnational1| × |threshold|")
print(f"               = {len(espacio_subnational_sorted)} × {len(espacio_threshold)} = {len(espacio_conjunto)} combinaciones")
print("-" * 70)
print("  Primeros 15 elementos del espacio muestral conjunto:")
for i, par in enumerate(espacio_conjunto[:15], start=1):
    print(f"  {i:>3}. Provincia: {par[0]:<25} | Threshold: {par[1]}%")
print("  ...")
print("═" * 70)
```


---
##Resumen de Espacios Muestrales
---

| Variable | Tipo | Método | Cardinalidad \|Ω\| |
|---|---|---|---|
| `subnational1` | Discreta – Nominal | `.unique()` | 24 provincias/regiones |
| `threshold` | Discreta – Ordinal | `.unique()` + `itertools` | 8 valores: {0, 10, 15, 20, 25, 30, 50, 75} |
| `subnational1` × `threshold` | Conjunto | `itertools.product()` | 24 × 8 = 192 |

> **Nota:** Las variables `area_ha`, `extent_2000_ha` y `tc_loss_ha_2024` son **continuas**,  
> por lo que no tienen un espacio muestral finito enumerable y no aplica esta fase para ellas.


---
# **Fase 4: Preguntas de Reflexión Científica**
---


## **Preguntas de Control**

---

### **1. Respondan**
* **Si una variable continua se redondea a dos decimales debido a limitaciones del sensor, ¿debería tratarse matemáticamente como una variable  discreta?**

    > **Respuesta:** No. Aunque el redondeo limite el número de valores posibles a un conjunto finito, la variable debe seguir tratándose matemáticamente como continua.

    - Naturaleza del fenómeno: El hecho de que un sensor o instrumento de medición tenga una resolución limitada (como dos decimales) no cambia la naturaleza física del objeto medido. El área de una provincia o la pérdida de bosque son magnitudes que existen en una escala infinita de valores reales.
    - Distribución de probabilidad: Para estas variables se siguen utilizando funciones de densidad de probabilidad (como la Normal o Gamma) y no funciones de masa de probabilidad (usadas para conteos discretos).
    - Error de medición: El redondeo se considera simplemente un error de cuantificación o una limitación de precisión, pero las operaciones matemáticas (promedios, derivadas, integrales) asumen la continuidad subyacente para mantener la validez de los modelos geoespaciales.


---

### **2. Analicen**
* **¿Cómo afecta un rango teórico mal definido a la detección de valores atípicos (outliers) en su proyecto regional?**

    > **Respuesta:** En el proyecto, manejamos variables como Area_ha, Extent_2000_ha y Tc_loss_ha_2024, aquí, un rango teórico mal definido es un error crítico. El rango teórico son los límites lógicos que una variable puede alcanzar (ej. la pérdida de bosque no puede ser mayor que el área total).

  >Impactos en la detección de valores atípicos (outliers):

  - *Falsos Negativos* (Invisibilidad del error): Si el rango teórico se define de forma demasiado amplia o laxa, valores que son físicamente imposibles podrían ser aceptados como "normales".
  - *Falsos Positivos* (Validación incorrecta): Si el rango es demasiado estrecho y no considera la diversidad geográfica de las provincias (Subnational1), valores reales pero extremos (como una provincia excepcionalmente grande o una deforestación masiva por un incendio real) podrían ser eliminados erróneamente como ruido.
  - *Distorsión de Umbrales Estadísticos*: La detección de outliers suele basarse en la desviación estándar o el rango intercuartílico. Si el rango teórico incluye datos basura o "puntos de relleno" (como usar un 9999 para datos faltantes sin declararlo), las medidas de dispersión se inflan, haciendo que los verdaderos outliers queden ocultos dentro del "ruido" estadístico.
