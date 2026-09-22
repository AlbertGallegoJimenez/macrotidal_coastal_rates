# Metodología de Análisis de Evolución de Línea de Costa en Entornos Complejos (Macromareales)

La extracción de tasas representativas de evolución costera (m/año) a partir de imágenes satelitales en entornos macromareales y de alta energía se ve severamente afectada por el ruido de alta frecuencia. La variabilidad intradía (carrera de marea, *run-up*) y la estacionalidad (perfil de verano vs. invierno) pueden enmascarar por completo la tendencia interanual subyacente. 

Para resolver este problema de relación señal-ruido, se ha desarrollado un *pipeline* analítico estructurado en tres fases: un filtrado dinámico por idoneidad de marea, una estabilización no lineal de la serie y un ajuste de regresión robusta.

---

## Fase 1: Filtrado Inteligente de Mareas

En lugar de aplicar una corrección geométrica basada en una pendiente constante (inviable por la variabilidad del perfil de playa) o un umbral de marea estático, se implementa un algoritmo de búsqueda de ventana deslizante (*Sliding Window Search*). Este algoritmo evalúa múltiples rangos de marea para cada playa y selecciona el subconjunto de datos óptimo resolviendo el dilema sesgo-varianza.

El algoritmo descarta automáticamente combinaciones que no cumplen con criterios de supervivencia (cobertura temporal mínima y ausencia de huecos mayores a 4 años) y evalúa las ventanas válidas maximizando la siguiente función de idoneidad matemática ($S$):

$$S = \left[ \ln(N) \cdot W(z_{mean}) \cdot C_{estacional} \right] - \lambda_1(\Delta z) - \lambda_2(T_{gap})$$

Donde:
*   **$N$ (Volumen de datos):** El logaritmo natural del número de observaciones. Premia la densidad estadística pero asume rendimientos decrecientes (pasar de 5 a 15 datos es crítico; pasar de 80 a 90 es marginal).
*   **$W(z_{mean})$ (Calidad espectral):** Ponderación lineal que favorece las ventanas más cercanas a la pleamar máxima local. En niveles altos, el índice espectral no se contamina por la humedad de la zona intermareal.
*   **$C_{estacional}$ (Concentración estacional):** Calculado mediante la varianza circular de los meses de captura. Premia las ventanas temporales que agrupan imágenes en la misma estación (ej. aislando la envolvente de verano), eliminando la variabilidad morfológica estacional del eje Y.
*   **$\lambda_1(\Delta z)$ (Penalización geométrica):** Castiga ventanas de marea excesivamente anchas que introducirían ruido topográfico al mezclar cotas muy dispares.
*   **$\lambda_2(T_{gap})$ (Penalización por obsolescencia):** Resta puntuación si la ventana de marea obliga a descartar los datos más recientes del periodo de estudio, garantizando que la tendencia calculada conecte con la dinámica actual de la playa.

El algoritmo se ejecuta a escala de unidad geomorfológica (playa completa), garantizando coherencia espacial a lo largo de todos los perfiles transversales.

---

## Fase 2: Estabilización de la Serie Temporal

Una vez aislada la ventana de marea óptima, la serie temporal transversal (*cross-shore*) aún retiene ruido de alta frecuencia derivado de errores sub-píxel, nubosidad no detectada o eventos de tormenta puntuales. Para estabilizar la señal antes de modelar la tendencia, se aplica un filtro temporal.

1.  **Mediana frente a Media:** Se utiliza una **mediana móvil** en lugar de una media móvil. La media es altamente sensible a valores atípicos (*outliers*), arrastrando la posición calculada hacia el error. La mediana ignora la magnitud del error y se ancla en la posición morfodinámica predominante.
2.  **Ventana Basada en Tiempo:** Debido al muestreo irregular de los satélites (especialmente por la cobertura de nubes), la ventana de suavizado no se define por un número fijo de observaciones, sino por un intervalo de tiempo estricto (por defecto, $360$ días). Esto garantiza que el filtro colapse de forma homogénea el ruido a lo largo de toda la serie, independientemente de la densidad temporal local.

---

## Fase 3: Modelado y Cálculo de Tasas

El paso final consiste en extraer la tasa interanual de avance o retroceso (m/año). La metodología calcula y compara dos modelos estadísticos distintos sobre la serie temporal estabilizada.

### 1. El Problema de Mínimos Cuadrados Ordinarios (OLS)
La regresión lineal clásica (OLS) traza la recta que minimiza la suma de los residuos al cuadrado: $\sum (y_i - \hat{y}_i)^2$. 
En dinámica litoral, esto representa un riesgo severo: si un temporal extremo desplaza puntualmente la línea de costa, OLS eleva ese error al cuadrado, otorgándole un peso masivo. Este "efecto palanca" deforma la pendiente de la recta, falseando la tasa interanual. Por ello, el modelo OLS se calcula únicamente como referencia secundaria para evaluar la linealidad ($R^2$) y el p-valor de la serie ya suavizada.

### 2. Regresión Robusta (Estimador Theil-Sen)
Para garantizar una métrica fidedigna, la tasa principal se extrae utilizando el estimador de Theil-Sen, un método no paramétrico basado en geometría combinatoria. El algoritmo funciona de la siguiente manera:

*   Calcula la pendiente matemática (tasa puntual) entre todos los pares posibles de observaciones en la serie temporal:
    $$m_{ij} = \frac{y_j - y_i}{x_j - x_i}$$
*   La tasa global representativa se obtiene calculando la mediana de todas las combinaciones posibles:
    $$m = \text{Mediana}(m_{ij})$$

**Ventajas operativas:**
Al depender de la mediana, el estimador Theil-Sen es insensible a la magnitud de los valores atípicos. Las deformaciones generadas por ruido extremo quedan relegadas a las colas de la distribución estadística, sin afectar a la tendencia central. Posee un punto de ruptura estadístico de aproximadamente el $29\%$; es decir, casi un tercio de las observaciones de la playa podrían ser datos erróneos o tormentas extremas y el algoritmo seguiría extrayendo la tasa morfológica de fondo exacta.

Junto a la tasa (m/año), el método calcula los intervalos de confianza (IC) al $95\%$. La tendencia solo se clasifica como **estadísticamente significativa** si el p-valor es inferior a $0.05$ y los límites inferior y superior del intervalo de confianza poseen el mismo signo (sin cruzar el valor cero), certificando que la señal de evolución interanual supera a la variabilidad natural del sistema costero.
