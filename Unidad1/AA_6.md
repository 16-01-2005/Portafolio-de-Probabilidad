# **AA - Cálculo de Momentos: Algoritmos para el Cálculo de Momentos y Validación Computacional**

---

## **1. Datos Generales**

| **Atributo** | **Descripción** |
| :--- | :--- |
| **Estudiantes** | Luis Armijos, Kiara Condoy, Héctor Guerrero, Javier Guarnizo, Ricardo Ochoa, Emily Salas |
| **Asignatura** | Teoría de la Distribución y Probabilidad |
| **Ciclo** | 2do “A” |
| **Unidad** | 1 |
| **Nombre del Docente** | Cristian Ramiro Narváez Guillén |
| **Fecha de entrega** | Domingo 16 de abril 2026 |

---


##### **Paso 1: Carga de Datos Filtrados, solo de Loja**


```python
import pandas as pd
import numpy as np

url = 'https://docs.google.com/spreadsheets/d/1E-rrTgcTIWS-UnuVf1ldKvhDkwT6uUgh/export?format=xlsx'

try:
    df = pd.read_excel(url, sheet_name='Subnational 1 tree cover loss')
    df = df.dropna(how='all').reset_index(drop=True)
    print("¡Conexión exitosa a la hoja de Provincias!")

    columna_provincia = 'subnational1'

    # NOTA: Vamos a elegir una columna de pérdida real.
    columna_variable = 'tc_loss_ha_2024'

    # Filtramos para la región de Loja
    df_loja = df[df[columna_provincia].astype(str).str.contains('Loja', case=False, na=False)]

    # Extraemos los datos numéricos de la pérdida forestal
    datos_analizar = df_loja[columna_variable].dropna().tolist()

    print(f"\n¡Todo listo! Datos de pérdida forestal para Loja ({columna_variable}) extraídos.")
    print(f"Total de registros numéricos encontrados: {len(datos_analizar)}")
    print(f"Valores a analizar: {datos_analizar}")

except Exception as e:
    print(f"Error: {e}. Si la columna '{columna_variable}' no existe, verifica el nombre exacto en el paso anterior.")
except Exception as e:
    print(f"Error general: {e}")
```


##### **Paso 2: Desarrollo Algorítmico (Funciones Manuales)**


```python
def calcular_esperanza(datos):
    """Calcula la Esperanza Matemática de manera manual."""
    suma_total = 0
    n = 0
    for x in datos:
        suma_total += x
        n += 1
    return suma_total / n if n > 0 else 0

def calcular_varianza(datos, insesgada=True):
    """Calcula la Varianza manual (muestral si insesgada=True, poblacional si False)."""
    media = calcular_esperanza(datos)
    suma_cuadrados_dif = 0
    n = 0
    for x in datos:
        suma_cuadrados_dif += (x - media) ** 2
        n += 1

    if n <= 1:
        return 0
    # n - 1 para la muestral (insesgada) y n para la poblacional
    return suma_cuadrados_dif / (n - 1) if insesgada else suma_cuadrados_dif / n

# Ejecutamos las funciones con tus datos reales de 2024
esp_manual = calcular_esperanza(datos_analizar)
var_muestral_manual = calcular_varianza(datos_analizar, insesgada=True)
var_pob_manual = calcular_varianza(datos_analizar, insesgada=False)

print("--- CÁLCULOS MANUALES COMPLETADOS ---")
print(f"Esperanza Manual: {esp_manual}")
print(f"Varianza Muestral Manual: {var_muestral_manual}")
print(f"Varianza Poblacional Manual: {var_pob_manual}")
```


##### **Paso 3: Validación Cruzada (Cuadro Comparativo con NumPy)**


```python
# 1. Cálculos usando la librería NumPy
esp_numpy = np.mean(datos_analizar)
var_muestral_numpy = np.var(datos_analizar, ddof=1) # ddof=1 fuerza la varianza muestral (n-1)
var_pob_numpy = np.var(datos_analizar)             # ddof=0 por defecto (poblacional)

# 2. Construcción de la tabla comparativa con Pandas
datos_tabla = {
    "Métrica": ["Esperanza Matemática (Media)", "Varianza Muestral (n-1)", "Varianza Poblacional (n)"],
    "Manual (Hecho a mano)": [esp_manual, var_muestral_manual, var_pob_manual],
    "NumPy (Estándar)": [esp_numpy, var_muestral_numpy, var_pob_numpy],
    "Diferencia Absoluta": [
        abs(esp_manual - esp_numpy),
        abs(var_muestral_manual - var_muestral_numpy),
        abs(var_pob_manual - var_pob_numpy)
    ]
}

df_comparativo = pd.DataFrame(datos_tabla)

# Configuramos Pandas para que muestre hasta 15 decimales
# Esto es vital para evaluar la precisión de punto flotante
pd.options.display.float_format = '{:.15f}'.format

print("CUADRO COMPARATIVO DE RESULTADOS:")
display(df_comparativo)
```


##### **Paso 4: Análisis de Resultados**


**Comprensión de Parámetros Ocultos:** Si usamos np.var() a ciegas sin entender la diferencia matemática entre muestra ($n-1$) y población ($n$), cometeríamos un error metodológico grave. NumPy calcula por defecto la poblacional, obligándonos a conocer su trasfondo para aplicar correctamente ddof=1 en estudios estadísticos muestrales.

**Precisión de Punto Flotante (IEEE 754):** Al procesar grandes volúmenes de datos mediante sumatorias manuales (for), las computadoras acumulan micro-errores de redondeo binario. Las librerías de la industria aplican optimizaciones de bajo nivel (como algoritmos de sumatoria compensada de Kahan) para mitigar esta pérdida de precisión.

**Optimización de Software:** Aunque recrear el algoritmo nos enseña la lógica de la sumatoria ($\sum$), en entornos de producción reales usar bucles for en Python puro es ineficiente. Entender esto justifica la necesidad de utilizar librerías vectorizadas como NumPy, las cuales ejecutan estas operaciones de forma paralela directamente en memoria.
