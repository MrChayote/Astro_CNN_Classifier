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

Dado que la cantidad de imagenes por clase es grande se decidio aumentar los datos y eliminar las clases que eran muy pocoas (Asteroides, Cometas). [code](https://github.com/MrChayote/Astro_CNN_Classifier/blob/codigos/augment_dataset.py) 

> Nota:
> En caso de que los archivos esten corruptos puedes usar el siguiente [code](https://github.com/MrChayote/Astro_CNN_Classifier/blob/codigos/verificacion.py)

## Modelos
Aplicacion de los modelos:
- [CNN](https://github.com/MrChayote/Astro_CNN_Classifier/blob/codigos/Stellar-Clasification-CNN.py)
- [Modelo EfficientNet-B0](https://github.com/MrChayote/Astro_CNN_Classifier/blob/codigos/Stellar-Clasification-Model-EfficientNet_B0.py)
- [Modelo ResNet50](https://github.com/MrChayote/Astro_CNN_Classifier/blob/codigos/Stellar-Clasification-Model-ResNet.py)
- [Modelo Vision Transformer (ViT)](https://github.com/MrChayote/Astro_CNN_Classifier/blob/codigos/Stellar-Clasification-ViT.py)

## Resultados:

Tras entrenar y evaluar cuatro arquitecturas diferentes para la clasificación estelar, se obtuvieron los siguientes resultados comparativos:

| Modelo | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **EfficientNet** | **0.975** | **0.975** | **0.975** | **0.975** |
| Vision Transformer (ViT) | 0.967 | 0.967 | 0.967 | 0.967 |
| ResNet | 0.956 | 0.956 | 0.956 | 0.956 |
| CNN Personalizada | 0.768 | 0.766 | 0.768 | 0.764 |

**Hallazgos clave:**
![Gráfica comparativa de modelos](https://github.com/MrChayote/Astro_CNN_Classifier/blob/codigos/comparativa_modelos_final.png)

* **Mejor Desempeño:** El modelo basado en **EfficientNet** demostró ser el más robusto, alcanzando la métrica más alta en todas las categorías (>97%), consolidándose como la mejor opción para este dataset.
* **Arquitecturas Modernas vs. Tradicionales:** Se observa una diferencia significativa (+20%) entre la CNN personalizada y los modelos pre-entrenados (ResNet, ViT, EfficientNet), lo que valida el uso de *Transfer Learning* para esta tarea.
* **Transformers en Visión:** El modelo **ViT (Vision Transformer)** obtuvo un segundo lugar muy cercano (96.7%), demostrando que las arquitecturas basadas en atención son altamente competitivas frente a las convolucionales clásicas para la clasificación de cuerpos estelares.
