```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint, norm, binom, uniform
import seaborn as sns

# Configuracion de estilo
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 11

print("Entorno configurado correctamente")


# Parametros de la distribucion normal
mu = 200  # media
sigma = 30  # desviacion estandar

# Crear la distribucion
dist_normal = norm(loc=mu, scale=sigma)

# Rango de valores para graficar
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)

# PDF
pdf_normal = dist_normal.pdf(x)

# Visualizacion
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, pdf_normal, 'b-', linewidth=2, label=f'N({mu}, {sigma}^2)')
plt.fill_between(x, pdf_normal, alpha=0.3)
plt.axvline(mu, color='red', linestyle='--', label=f'Media = {mu}')
plt.xlabel('x (Tiempo de respuesta en ms)')
plt.ylabel('f(x)')
plt.title('PDF: Variable Aleatoria Continua (Normal)')
plt.legend()
plt.grid(alpha=0.3)
# CDF
cdf_normal = dist_normal.cdf(x)
plt.subplot(1, 2, 2)
plt.plot(x, cdf_normal, 'r-', linewidth=2)
plt.xlabel('x')
plt.ylabel('F(x) = P(X <= x)')
plt.title('CDF: Distribucion Normal')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import seaborn as sns

# 1. Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

# 2. Definición de Parámetros
# p = probabilidad de éxito (ej. lanzar una moneda trucada donde cara = 1)
p = 0.6
dist_bernoulli = bernoulli(p)

# Valores posibles (0 = fracaso, 1 = éxito)
x = [0, 1]

# 3. Cálculo de PMF y CDF
pmf_values = dist_bernoulli.pmf(x)
cdf_values = dist_bernoulli.cdf(x)

# 4. Visualización
plt.figure(figsize=(12, 5))

# Gráfica PMF (Función de Masa de Probabilidad)
plt.subplot(1, 2, 1)
plt.bar(x, pmf_values, color=['red', 'green'], alpha=0.6, width=0.1)
plt.xticks(x, ['Fracaso (0)', 'Éxito (1)'])
plt.ylim(0, 1)
plt.title(f'PMF: Distribución Bernoulli (p={p})')
plt.ylabel('Probabilidad P(X=x)')

# Gráfica CDF (Función de Distribución Acumulada)
plt.subplot(1, 2, 2)
# Para la CDF de Bernoulli, usamos una gráfica escalonada
x_step = np.linspace(-0.5, 1.5, 1000)
plt.plot(x_step, dist_bernoulli.cdf(x_step), 'b-', drawstyle='steps-post', linewidth=2)
plt.fill_between(x_step, dist_bernoulli.cdf(x_step), step="post", alpha=0.1)
plt.xticks([0, 1])
plt.title('CDF: Probabilidad Acumulada')
plt.ylabel('F(x) = P(X <= x)')

plt.tight_layout()
plt.show()

# 5. Cálculo e Interpretación de Probabilidades
print("--- INTERPRETACIÓN DE RESULTADOS ---")
# Probabilidad de éxito
p_exito = dist_bernoulli.pmf(1)
# Probabilidad de fracaso
p_fracaso = dist_bernoulli.pmf(0)
# Esperanza matemática (Media)
esperanza = dist_bernoulli.mean()

print(f"1. P(X=1) [Éxito]: {p_exito:.2f}")
print(f"   Interpretación: Si lanzamos esta moneda, hay un {p_exito*100}% de probabilidad de obtener cara.")

print(f"\n2. P(X=0) [Fracaso]: {p_fracaso:.2f}")
print(f"   Interpretación: La probabilidad complementaria de fallar (obtener cruz) es del {p_fracaso*100}%.")

print(f"\n3. E[X] (Media): {esperanza:.2f}")
print(f"   Interpretación: En una distribución Bernoulli, la media es igual a 'p'. ")
print(f"   Indica que en promedio, esperaríamos un éxito el {esperanza*100}% de las veces.")
```


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint
import seaborn as sns

# 1. Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 11

# 2. Definición de Parámetros: Lanzamiento de un dado de 6 caras
# low (incluido), high (excluido por scipy.stats.randint)
bajo, alto = 1, 7
dist_uniforme = randint(low=bajo, high=alto)

# Rango de valores posibles
x = np.arange(bajo, alto)

# 3. Cálculo de PMF y CDF
pmf_values = dist_uniforme.pmf(x)
cdf_values = dist_uniforme.cdf(x)

# 4. Visualización
plt.figure(figsize=(13, 5))

# Gráfica PMF (Función de Masa de Probabilidad)
plt.subplot(1, 2, 1)
plt.stem(x, pmf_values, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.title(f'PMF: Distribución Uniforme Discreta\n(Dado de {alto-1} caras)')
plt.xlabel('Resultado (x)')
plt.ylabel('P(X = x)')
plt.xticks(x)
plt.ylim(0, 0.3)

# Gráfica CDF (Función de Distribución Acumulada)
plt.subplot(1, 2, 2)
# Creamos un eje x más denso para mostrar el efecto de "escalones"
x_step = np.linspace(bajo - 1, alto, 1000)
plt.plot(x_step, dist_uniforme.cdf(x_step), 'r-', drawstyle='steps-post', linewidth=2)
plt.title('CDF: Probabilidad Acumulada')
plt.xlabel('x')
plt.ylabel('F(x) = P(X <= x)')
plt.xticks(np.arange(bajo - 1, alto + 1))
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 5. Cálculo e Interpretación de Probabilidades Específicas
print("-" * 40)
print("INTERPRETACIÓN DE RESULTADOS")
print("-" * 40)

# Probabilidad 1: Obtener un valor exacto (ej. un 3)
p_exacta = dist_uniforme.pmf(3)
print(f"1. P(X = 3): {p_exacta:.4f}")
print(f"   Interpretación: La probabilidad de obtener exactamente 3 es del {p_exacta*100:.2f}%.")
print(f"   Al ser uniforme, cualquier número del 1 al 6 tiene esta misma probabilidad (1/6).")

# Probabilidad 2: Obtener un valor menor o igual a 4
p_acumulada = dist_uniforme.cdf(4)
print(f"\n2. P(X <= 4): {p_acumulada:.4f}")
print(f"   Interpretación: Existe un {p_acumulada*100:.2f}% de probabilidad de que el resultado")
print(f"   esté en el conjunto {{1, 2, 3, 4}}.")

# Probabilidad 3: Obtener un valor mayor a 2
p_mayor = 1 - dist_uniforme.cdf(2)
print(f"\n3. P(X > 2): {p_mayor:.4f}")
print(f"   Interpretación: La probabilidad de obtener un resultado mayor a 2 (es decir, 3, 4, 5 o 6)")
print(f"   es del {p_mayor*100:.2f}%.")
```


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import seaborn as sns

# 1. Configuración de estilo (siguiendo tu formato)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 11

print("Entorno configurado para Distribución Binomial")

# 2. Parámetros de la distribución Binomial
# Supongamos un examen de 20 preguntas (n=20) donde la probabilidad
# de acertar cada una al azar es del 25% (p=0.25)
n = 20  # Número de ensayos
p = 0.25  # Probabilidad de éxito

# Crear el objeto de la distribución
dist_binomial = binom(n, p)

# Rango de valores (éxitos posibles de 0 a n)
x = np.arange(0, n + 1)

# PMF (Probability Mass Function) - Función de Masa de Probabilidad
pmf_bi = dist_binomial.pmf(x)

# CDF (Cumulative Distribution Function) - Función de Distribución Acumulada
cdf_bi = dist_binomial.cdf(x)

# 3. Visualización
plt.figure(figsize=(13, 5))

# Gráfica PMF
plt.subplot(1, 2, 1)
plt.bar(x, pmf_bi, color='skyblue', edgecolor='navy', alpha=0.7, label=f'B(n={n}, p={p})')
plt.axvline(n*p, color='red', linestyle='--', label=f'Media (E[X]) = {n*p}')
plt.xlabel('Número de éxitos (k)')
plt.ylabel('P(X = k)')
plt.title('PMF: Probabilidad de Éxitos Exactos')
plt.legend()
plt.xticks(x[::2]) # Mostrar etiquetas de 2 en 2 para claridad

# Gráfica CDF
plt.subplot(1, 2, 2)
plt.step(x, cdf_bi, where='mid', color='salmon', linewidth=2, label='Prob. Acumulada')
plt.fill_between(x, cdf_bi, step="mid", alpha=0.2, color='salmon')
plt.xlabel('Número de éxitos (k)')
plt.ylabel('F(k) = P(X <= k)')
plt.title('CDF: Probabilidad Acumulada')
plt.legend()
plt.xticks(x[::2])

plt.tight_layout()
plt.show()

# 4. Cálculo de Probabilidades Específicas e Interpretación
print("-" * 50)
print("CÁLCULO E INTERPRETACIÓN DE RESULTADOS")
print("-" * 50)

# Probabilidad 1: Exactamente 5 éxitos
p1 = dist_binomial.pmf(5)
print(f"1. P(X = 5): {p1:.4f}")
print(f"   Interpretación: Hay un {p1*100:.2f}% de probabilidad de obtener exactamente 5 aciertos de 20.")

# Probabilidad 2: 4 éxitos o menos (Acumulada)
p2 = dist_binomial.cdf(4)
print(f"\n2. P(X <= 4): {p2:.4f}")
print(f"   Interpretación: Existe un {p2*100:.2f}% de probabilidad de que el número de aciertos")
print(f"   sea 4 o menos (zona de reprobación).")

# Probabilidad 3: Más de 8 éxitos
p3 = 1 - dist_binomial.cdf(8)
print(f"\n3. P(X > 8): {p3:.4f}")
print(f"   Interpretación: La probabilidad de tener un desempeño sobresaliente (más de 8 aciertos)")
print(f"   es muy baja, apenas un {p3*100:.2f}%.")
```


1. ¿Cual es la diferencia fundamental entre una variable aleatoria discreta y una continua?

    Variable Aleatoria Discreta: Tiene un número finito o infinito contable de valores. Generalmente, son resultados de un proceso de conteo. (Ej: Número de hijos, caras en un dado).

    Variable Aleatoria Continua: Puede tomar cualquier valor dentro de un intervalo (o unión de intervalos) de números reales. Son resultados de un proceso de medición. (Ej: Estatura, temperatura, tiempo).

2. Por qué en una variable continua P(X = x) = 0 para cualquier valor especifico x?

    Matemáticamente, la probabilidad en una distribución continua se define como el área bajo la curva (PDF) en un intervalo. Dado que un punto específico $x$ no tiene "ancho", el área sobre ese punto es cero:
    
  $$\int_{a}^{a} f(x) \, dx = 0$$
    Es como intentar medir el peso de una línea sin grosor; por más que la línea exista, su masa es nula porque no tiene volumen.

3. Explique la relacion entre la PMF/PDF y la CDF. ¿Como se obtiene una de la otra?

    La CDF ($F(x)$) siempre representa la probabilidad acumulada hasta el valor $x$, es decir, $P(X \leq x)$.
    De PMF/PDF a CDF: * Discreta: Se obtiene sumando las probabilidades individuales: $F(x) = \sum_{x_i \leq x} P(X = x_i)$.
    Continua: Se obtiene integrando la PDF: $F(x) = \int_{-\infty}^{x} f(t) \, dt$.

    De CDF a PMF/PDF:Discreta: Restando valores sucesivos de la CDF.Continua: Derivando la CDF: $f(x) = \frac{d}{dx} F(x)$.

4. Una variable aleatoria X representa el número de estudiantes que llegan tarde a clase. ¿Es X discreta o continua? Justifique.
    La variable X es discreta.
    Justificación: El número de estudiantes es un proceso de conteo. No pueden llegar "2.5 estudiantes" tarde; los valores posibles son números enteros
    
  $\{0, 1, 2, \dots, n\}$.

5. Calcule P(X = 5) para una variable continua con PDF f(x) = 2x en [0,1]. Explique su respuesta.

    Resultado: $P(X = 5) = 0$.
    Explicación: Primero, el valor $x=5$ está fuera del dominio definido $[0, 1]$, por lo que su probabilidad sería cero de todos modos. Segundo, e incluso si el valor estuviera dentro del rango (por ejemplo, $P(X = 0.5)$), el resultado seguiría siendo cero porque es una variable continua y, como explicamos antes, la probabilidad puntual en distribuciones continuas siempre es nula.

6. Dibuje aproximadamente la CDF de una variable discreta que toma valores {1, 2, 3} con probabilidades {0.3, 0.5, 0.2}.

    La CDF de una variable discreta se ve como una escalera:

    Para $x < 1$: $F(x) = 0$Para $1 \leq x < 2$: $F(x) = 0.3$ (Primer salto)

    Para $2 \leq x < 3$: $F(x) = 0.3 + 0.5 = 0.8$ (Segundo salto)

    Para $x \geq 3$: $F(x) = 0.8 + 0.2 = 1.0$ (Tercer salto, llega al tope)

  7. Identifique una variable aleatoria continua en el contexto de la carrera de Computación y justifique porque es continua
  Variable: El tiempo de latencia de un paquete de datos en una red local.
  Justificación: Es una variable continua porque el tiempo es una magnitud física que puede medirse con precisión infinita. Entre 10 ms y 11 ms pueden existir infinitos valores (10.05 ms, 10.05001 ms, etc.), dependiendo de la sensibilidad del reloj del sistema. No se cuenta, se mide.
