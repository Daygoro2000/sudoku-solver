import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

#Carga el modelo pre-entrenado (.h5)
def load_sudoku_model(path):
    return load_model(path)

#Reconoce los dígitos en las celdas usando el modelo
def read_cells(cell, model):
    result = []
    for image in cell:
        img = np.asarray(image)
        img = img[4:img.shape[0]-4, 4:img.shape[1]-4]
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        img = img.reshape(1, 32, 32, 1)
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=1)
        probabilityValue = np.max(predictions)
        result.append(classIndex[0] if probabilityValue > 0.75 else 0)
    return result