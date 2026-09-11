# Computer Vision & Deep Learning Sudoku Solver

Solución integral orientada a la digitalización, extracción óptica y resolución automatizada de tableros de Sudoku a partir de imágenes capturadas en escenarios no controlados. El sistema desacopla el entrenamiento de una red neuronal convolucional (CNN) para el reconocimiento de caracteres de las fases de visión por computador (OpenCV) y del motor algorítmico de resolución lógica.

---

## Pipeline del Proyecto

### 1. Entrenamiento y Validación del Modelo (`notebooks/model_training.ipynb`)
* **Conjunto de datos:** Dataset estructurado de dígitos (clases 0 a 9) sometido a partición estricta de entrenamiento, validación y prueba independiente.
* **Aumento de Datos (Data Augmentation):** Generación de variaciones morfológicas (desplazamientos axiales, zoom, cizallamiento y rotaciones leves) para maximizar la generalización ante distintas tipografías y artefactos de captura.
* **Arquitectura CNN:** Red secuencial con capas convolucionales alternadas (`Conv2D`), submuestreo espacial (`MaxPooling2D`), capas densas y regularización por desconexión aleatoria (`Dropout`) para mitigar el sobreajuste.
* **Optimización y Rendimiento:** Compilado mediante RMSprop y función de pérdida de entropía cruzada categórica, alcanzando un *Test Accuracy* del **99.80%** en datos no vistos.
* **Exportación:** Serialización de los pesos y arquitectura en el artefacto `models/modelo_digitos.h5` para su consumo en entornos de inferencia.

### 2. Visión Artificial y Segmentación Geométrica (`src/vision.py`)
* **Preprocesamiento:** Conversión del espacio de color a escala de grises, filtrado de ruido mediante desenfoque gaussiano y umbralización adaptativa para aislar bordes estructurales.
* **Detección y Corrección de Perspectiva:** Identificación del contorno cuadrangular de mayor superficie y aplicación de una transformación proyectiva (`warpPerspective`) para generar un plano ortogonal de 450x450 píxeles.
* **Segmentación Matricial:** División del lienzo rectificado en 81 subregiones (9x9) con recorte paramétrico de márgenes para eliminar trazos adyacentes de las celdas.

### 3. Inferencia Óptica de Dígitos (`src/model.py`)
* **Acondicionamiento de Tensores:** Redimensionado de cada celda a 32x32 píxeles, adición del canal monocromático y normalización de intensidades en el rango [0, 1].
* **Predicción y Umbralización:** Evaluación mediante `modelo_digitos.h5`, asignando el dígito correspondiente si la certidumbre de la activación softmax supera el 75%; las activaciones inferiores se etiquetan como celdas vacías (0).

### 4. Motor Lógico y Resolución Asíncrona (`src/solver.py`)
* **Algoritmo de Búsqueda:** Implementación de *backtracking* recursivo para satisfacer de forma determinista las restricciones de fila, columna y cuadrícula de 3x3.
* **Tolerancia a Fallos:** Ejecución encapsulada dentro de un `ThreadPoolExecutor` con límite estricto de tiempo (timeout de 5 segundos), previniendo bloqueos ante tableros irresolubles o matrices inconsistentes.

### 5. Despliegue y Servicio Web (`src/app.py`)
* **Capa Backend:** Servidor HTTP montado sobre Flask que expone rutas para el procesamiento asíncrono y renderizado dinámico mediante Jinja2.
* **Gestión de Rutas:** Configuración de rutas canónicas absolutas para aislar la carga de plantillas (`templates/`), archivos estáticos (`static/`) y artefactos de modelos (`models/`).

## Demostración y Validación

El funcionamiento del pipeline completo (ingesta de imagen, procesamiento morfológico, inferencia mediante la red neuronal y resolución algorítmica) se encuentra documentado en la siguiente prueba en vídeo:
https://drive.google.com/drive/folders/1qX0469hp3R6IVZ6obqvA74aKHGZwPkxn?usp=sharing
