# 🌌 AstroNet: Clasificación de Imágenes Astronómicas con CNN
![Banner de Ciencia de Datos](https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif)

Proyecto final de la materia de Deep Learning.  
El objetivo es construir un modelo de **red neuronal convolucional (CNN)** capaz de **clasificar imágenes astronómicas** del dataset [SpaceNet](https://www.kaggle.com/datasets/razaimam45/spacenet-an-optimally-distributed-astronomy-data).

---

## Descripción del proyecto

El dataset *SpaceNet* contiene diversas imágenes astronómicas distribuidas de forma óptima entre diferentes categorías.  
A partir de este conjunto de datos, el proyecto busca desarrollar una CNN que pueda **reconocer y clasificar automáticamente** el tipo de objeto celeste presente en cada imagen, como galaxias, nebulosas, agujeros negros o estrellas.

---

## Objetivos
- Analizar y comprender la estructura del dataset SpaceNet.  
- Implementacion de multiples modelos para poder obtener el máximo desempeño en la calsificacion de las imagenes
    - CNN.
    - Modelo EfficientNet-B0.
    - Modelo ResNet50
    - Modelo Vision Transformer (ViT).
- Comparacion de los resultados de los modelos.

---

### Estructura de los modelos:

### En este modelo se realizo una estructura:

**La Entrada y el Calentamiento:**
- Entrada ($256 \times 256$ píxeles)

- Rescaling (Normalización)

- Aumento de Datos ('RandomFlip', 'RandomRotation', 'RandomZoom')

**La red convolucional:**
- Bloque 1 (Detalles finos)
    - Usa 32 filtros

    - Tiene dos capas Conv2D seguidas

    - Termina con MaxPooling2D

- Bloque 2 (Formas):
    - Sube a 64 filtros

    - Tiene dos capas Conv2D seguidas

    - Termina con MaxPooling2D

- Bloque 3 (Conceptos complejos)
    - Sube a 128 filtros

    - Tiene dos capas Conv2D seguidas

    - Termina con MaxPooling2D

**"El Cerebro": Clasificación**:
- Flatten (Aplanar)

- Dense (256 neuronas)

- Dropout (0.2)

- Dense (Salida - 6 neuronas)
