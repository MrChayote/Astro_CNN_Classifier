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

## Procedimiento

Dado que la cantidad de imagenes por clase es grande se decidio aumentar los datos y eliminar las clases que eran muy pocoas (Asteroides, Cometas). Para aumentar la cantidad de datos puedes usar este codigo 
