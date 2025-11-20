"""
Modelo de red neuronal para Federated Learning
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2 


def create_model(input_shape=(8,), num_classes=4, learning_rate=0.0001):
    """
    Crea un modelo de red neuronal para clasificación multiclase.
    OPTIMIZADO PARA FEDERATED LEARNING.
    
    CAMBIOS:
    - Learning rate: 0.001 → 0.0001 (más estable en FL)
    - Dropout ELIMINADO (causa inestabilidad en agregación)
    - BatchNormalization agregado (mejor convergencia)
    - Regularización L2 (previene overfitting)
    
    Args:
        input_shape: Forma de entrada (número de features)
        num_classes: Número de clases para clasificación
        learning_rate: Tasa de aprendizaje para el optimizador
    
    Returns:
        Modelo compilado de Keras
    """
    model = Sequential([
        Input(shape=input_shape, dtype='float32'),
        
        # Capa 1: Dense + BatchNorm (sin Dropout)
        Dense(64, activation='relu', 
              kernel_regularizer=l2(0.001),
              name='dense_1'),
        BatchNormalization(name='bn_1'),
        
        # Capa 2: Dense + BatchNorm (sin Dropout)
        Dense(32, activation='relu',
              kernel_regularizer=l2(0.001),
              name='dense_2'),
        BatchNormalization(name='bn_2'),
        
        # Salida
        Dense(num_classes, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_model_weights(model):
    """
    Extrae los pesos del modelo en formato serializable.
    
    Args:
        model: Modelo de Keras
    
    Returns:
        Lista de arrays numpy con los pesos
    """
    return model.get_weights()


def set_model_weights(model, weights):
    """
    Establece los pesos del modelo.
    
    Args:
        model: Modelo de Keras
        weights: Lista de arrays numpy con los pesos
    """
    model.set_weights(weights)


def serialize_weights(weights):
    """
    Serializa los pesos del modelo para transmisión.
    
    Args:
        weights: Lista de arrays numpy
    
    Returns:
        Lista de listas (JSON serializable)
    """
    return [w.tolist() for w in weights]


def deserialize_weights(weights_list):
    """
    Deserializa los pesos del modelo.
    
    Args:
        weights_list: Lista de listas
    
    Returns:
        Lista de arrays numpy
    """
    import numpy as np
    return [np.array(w, dtype=np.float32) for w in weights_list]
