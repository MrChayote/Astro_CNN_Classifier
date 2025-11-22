# Librerías estándar de Python
import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# Librerías de terceros
import numpy as np
import tensorflow as tf
from IPython.display import Image, display
from joblib import Parallel, delayed
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Componentes específicos de TensorFlow / Keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  array_to_img, img_to_array,
                                                  load_img)
from tensorflow.keras import layers, models

ruta = r'~/SpaceNet.FLARE.imam_alam'

NUM_CLASS = ['black hole', 'constellation', 'galaxy', 'nebula', 'planet', 'star']

train_dataset = tf.keras.utils.image_dataset_from_directory(
    ruta,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    class_names=NUM_CLASS
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    ruta,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    class_names=NUM_CLASS
)


class_names = train_dataset.class_names
print(f"Clases encontradas (en orden): {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
print("\n¡Datasets listos y pesos calculados!")

INPUT_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3) # (128, 128, 3)
model = models.Sequential(name="Stellar_CNN")
# Capa de entrada y re-escalado: Normaliza los pixeles de [0, 255] a [0, 1]
model.add(layers.Rescaling(1./255, input_shape=INPUT_SHAPE))
model.add(layers.RandomFlip("horizontal_and_vertical"))
model.add(layers.RandomRotation(0.2))
model.add(layers.RandomZoom(0.2))

# --- Bloque Convolucional 1 ---
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same')) # Aumentar filtros
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same')) # Aumentar filtros
model.add(layers.MaxPooling2D((2, 2)))


# --- Bloque Convolucional 2 ---
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same')) # Aumentar filtros
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same')) # Aumentar filtros
model.add(layers.MaxPooling2D((2, 2)))

# --- Bloque Convolucional 3 ---
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same')) # Aumentar filtros
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same')) # Aumentar filtros
model.add(layers.MaxPooling2D((2, 2)))

# --- Capas de Clasificación (Cabeza) ---
model.add(layers.Flatten()) # "Aplana" los mapas de características a un vector
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2)) # Regularización para evitar sobreajuste previo era .5
model.add(layers.Dense(6, activation='softmax')) # Capa de salida

# Muestra un resumen del modelo
model.summary()

# --- 2. Compilar el Modelo ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModelo compilado exitosamente.")

# --- 3. Entrenar el Modelo ---

# Definir el callback de EarlyStopping
# 'val_loss' = monitorea la pérdida en los datos de validacion
# 'patience' = cuantas epochs espera si no hay mejora
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

EPOCHS = 50 # EarlyStopping lo parara si es necesario.

print("Iniciando entrenamiento...")

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)

print("¡Entrenamiento completado!")

# (Opcional) Guardar el modelo entrenado
# model.save('mi_modelo_estelar_v1.h5')
