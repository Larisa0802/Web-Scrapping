# Web Scraping y Predicción de Salarios en Ofertas de Empleo

## Trabajo realizado por
- Borja Arranz  
- Alba Rojas  
- Diego Verdejo  
- Larisa Mancebo  

## Curso
C.F.G.S. 2º Desarrollo de Aplicaciones Web (DAW)

## Módulo
Ampliación de Programación  
Profesor: Javier Ortiz Laguna

---

## Descripción general del proyecto

Este proyecto aborda la predicción de salarios en ofertas de empleo mediante un flujo completo dividido en varias fases. En primer lugar, se realiza un proceso de web scraping para recopilar datos reales de ofertas de empleo relacionadas con el desarrollo de software. Posteriormente, los datos obtenidos se procesan y limpian utilizando técnicas de análisis de datos. Finalmente, se entrena un modelo de Machine Learning con el objetivo de predecir el salario en función del nivel de experiencia requerido.

El proyecto refleja un escenario realista de adquisición y análisis de datos, teniendo en cuenta las limitaciones técnicas actuales de muchos portales web y la necesidad de adaptar las herramientas y estrategias empleadas.

---

## FASE 1: Scraping

En esta fase del proyecto se llevó a cabo la obtención de datos reales de ofertas de empleo mediante técnicas de web scraping. El objetivo principal fue recopilar información relevante sobre puestos relacionados con el desarrollo de aplicaciones web, que posteriormente sería utilizada en las fases de procesamiento de datos y machine learning.

### 1.1 Configuración del entorno y enfoque inicial

Para el desarrollo del scraping se utilizó Python 3.11 como lenguaje principal, trabajando desde el entorno Visual Studio Code. En una primera aproximación se emplearon las librerías `requests` y `BeautifulSoup`, al tratarse de herramientas adecuadas para la extracción de información en páginas con contenido HTML estático y ampliamente utilizadas en contextos educativos.

El primer portal analizado fue Indeed, ya que aparecía propuesto en el enunciado del ejercicio. Sin embargo, durante las pruebas se detectaron respuestas de tipo 403 (Forbidden), lo que indicaba la presencia de sistemas de protección contra accesos automatizados. A pesar de simular peticiones desde un navegador real mediante cabeceras HTTP, el contenido devuelto no incluía las ofertas de empleo, sino páginas de verificación, por lo que se descartó esta opción.

### 1.2 Análisis de alternativas y cambio de herramienta

Como segunda alternativa se probó con InfoJobs. En este caso, aunque las peticiones iniciales devolvían un código de estado correcto, las ofertas de empleo no estaban presentes en el HTML inicial, ya que se cargaban dinámicamente mediante JavaScript. Para evaluar esta opción se decidió utilizar Playwright, una herramienta que permite automatizar un navegador real y ejecutar contenido dinámico.

Mediante Playwright se consiguió visualizar las ofertas tras resolver manualmente un CAPTCHA, confirmando que los datos existían y eran accesibles para un usuario humano. No obstante, al inspeccionar el DOM se comprobó que las ofertas no estaban estructuradas de forma clara en enlaces o contenedores HTML, sino gestionadas mediante componentes dinámicos internos, lo que dificultaba su extracción estructurada sin recurrir a técnicas avanzadas fuera del alcance del proyecto. Por este motivo, también se decidió descartar InfoJobs como fuente de datos.

### 1.3 Selección de la fuente final y estrategia aplicada

Tras documentar las limitaciones de los portales anteriores, se optó por cambiar de fuente y utilizar Tecnoempleo, un portal especializado en ofertas de empleo del ámbito tecnológico y del desarrollo de software, alineado con el perfil de Desarrollo de Aplicaciones Web. En este caso, el sitio permitía el acceso a los datos mediante peticiones HTTP simples y el contenido relevante estaba disponible en el HTML sin necesidad de ejecutar JavaScript, lo que permitió volver a utilizar `requests` y `BeautifulSoup`.

Durante el análisis de la estructura de Tecnoempleo se observó que las ofertas no estaban separadas en bloques HTML claramente delimitados, sino integradas dentro de un contenedor general. No obstante, al extraer el texto completo de la página se identificó un patrón repetido en cada oferta, compuesto por el título del puesto, la empresa, la ubicación, la modalidad o fecha de publicación y el salario. A partir de este patrón se implementó una solución basada en la extracción del texto completo y la reconstrucción de cada oferta utilizando la posición relativa de las líneas asociadas al salario.

### 1.4 Resultado del scraping

Como resultado de esta fase se generó un archivo CSV estructurado que contiene las ofertas de empleo obtenidas, con los campos necesarios para su posterior procesamiento. Este conjunto de datos sirvió como entrada para la fase de limpieza y transformación, permitiendo continuar con el flujo completo del proyecto. La fase de scraping pone de manifiesto la necesidad de adaptar las herramientas y estrategias empleadas en función de las características técnicas de cada plataforma web.

---

## FASE 2: Procesamiento de datos

Tras completar la fase de scraping, se llevó a cabo el procesamiento de los datos obtenidos. En esta etapa se trabajó con el CSV generado previamente, que contenía la información en bruto, con el objetivo de limpiarlo y transformarlo para facilitar su uso en la fase de machine learning.

Para el procesamiento se utilizaron las librerías Pandas y NumPy, que permiten manipular datos de forma eficiente y realizar operaciones matemáticas sobre ellos. Uno de los principales ajustes realizados fue el tratamiento del salario, que inicialmente se presentaba como un rango de valores. Este rango se transformó en un valor medio, facilitando su interpretación y uso posterior.

Los atributos finales utilizados en el dataset procesado fueron el título del puesto, el salario y el nivel de experiencia. La experiencia se clasificó en tres categorías (junior, mid y senior), generando un conjunto de datos limpio y estructurado.

---

## FASE 3: Machine Learning

En esta fase del proyecto se desarrolló un modelo de Machine Learning utilizando la librería scikit-learn, con el objetivo de predecir el salario de una oferta de empleo a partir del nivel de experiencia requerido.

### Preparación de los datos

La variable categórica experiencia fue codificada en formato numérico utilizando `OrdinalEncoder`, asignando un orden lógico a los niveles junior, mid y senior. Posteriormente, el dataset se dividió en conjuntos de entrenamiento y prueba.

### Entrenamiento y evaluación

Se entrenaron distintos modelos de regresión y se evaluaron utilizando métricas como el Error Cuadrático Medio (MSE) y el coeficiente de determinación (R²), seleccionando el modelo con mejor rendimiento.

---

## Conclusión

El proyecto demuestra la viabilidad de aplicar técnicas de web scraping, procesamiento de datos y machine learning para analizar ofertas de empleo reales. Asimismo, pone de manifiesto la importancia de adaptar las herramientas utilizadas a las limitaciones técnicas de cada plataforma web y de justificar adecuadamente las decisiones tomadas.
