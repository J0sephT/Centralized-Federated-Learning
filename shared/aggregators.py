"""
Métodos de agregación para Federated Learning
"""
import numpy as np
from typing import List, Tuple


def fedavg(client_weights: List[List[np.ndarray]], 
           client_sizes: List[int]) -> List[np.ndarray]:
    """
    FedAvg (Federated Averaging) - McMahan et al., 2017
    
    Promedio ponderado de los pesos de los clientes según su cantidad de datos.
    
    Args:
        client_weights: Lista de pesos de cada cliente (cada uno es lista de arrays)
        client_sizes: Lista con el número de muestras de cada cliente
    
    Returns:
        Pesos agregados globalmente
    """
    total_samples = sum(client_sizes)
    aggregated_weights = []
    
    # Iterar por cada capa del modelo
    for layer_idx in range(len(client_weights[0])):
        # Promedio ponderado de los pesos de esta capa
        layer_sum = np.zeros_like(client_weights[0][layer_idx], dtype=np.float32)
        
        for client_idx, client_w in enumerate(client_weights):
            weight = client_sizes[client_idx] / total_samples
            layer_sum += client_w[layer_idx] * weight
        
        aggregated_weights.append(layer_sum)
    
    return aggregated_weights


def fedavgm(previous_weights: List[np.ndarray],
            aggregated_weights: List[np.ndarray],
            previous_momentum: List[np.ndarray],
            beta: float = 0.9,
            eta: float = 1.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    FedAvgM (FedAvg with Server Momentum) - Hsu et al., 2019
    
    Aplica momentum en el servidor para acelerar la convergencia.
    
    Args:
        previous_weights: Pesos del modelo en la ronda anterior
        aggregated_weights: Pesos agregados con FedAvg en la ronda actual
        previous_momentum: Momentum de la ronda anterior
        beta: Factor de momentum (típicamente 0.9)
        eta: Learning rate del servidor (típicamente 1.0)
    
    Returns:
        Tupla (nuevos_pesos, nuevo_momentum)
    """
    new_weights = []
    new_momentum = []
    
    for w_prev, w_agg, m_prev in zip(previous_weights, aggregated_weights, previous_momentum):
        # Convertir a float32 para consistencia
        w_prev = np.asarray(w_prev, dtype=np.float32)
        w_agg = np.asarray(w_agg, dtype=np.float32)
        m_prev = np.asarray(m_prev, dtype=np.float32)
        
        # Calcular gradiente (delta)
        gradient = w_agg - w_prev
        
        # Actualizar momentum
        momentum = beta * m_prev + gradient
        
        # Actualizar pesos
        w_new = w_prev + eta * momentum
        
        new_weights.append(w_new)
        new_momentum.append(momentum)
    
    return new_weights, new_momentum


def fednova(previous_weights: List[np.ndarray],
            client_weights: List[List[np.ndarray]],
            client_steps: List[int],
            client_sizes: List[int],
            eta: float = 1.0) -> List[np.ndarray]:
    """
    FedNova (Normalized Averaging) - Wang et al., 2020
    
    Normaliza las actualizaciones locales por el número de pasos de entrenamiento.
    Útil cuando los clientes tienen diferentes cantidades de datos o épocas.
    
    Args:
        previous_weights: Pesos del modelo global de la ronda anterior
        client_weights: Lista de pesos de cada cliente
        client_steps: Lista con el número de pasos de entrenamiento de cada cliente
        client_sizes: Lista con el número de muestras de cada cliente
        eta: Learning rate efectivo (típicamente 1.0)
    
    Returns:
        Pesos agregados globalmente
    """
    total_samples = sum(client_sizes)
    new_weights = []
    
    # Iterar por cada capa
    for layer_idx in range(len(previous_weights)):
        w_prev = np.asarray(previous_weights[layer_idx], dtype=np.float32)
        
        # Sumatoria ponderada de deltas normalizados
        weighted_delta_sum = np.zeros_like(w_prev, dtype=np.float32)
        
        for client_idx, client_w in enumerate(client_weights):
            # Delta del cliente
            delta = client_w[layer_idx] - w_prev
            
            # Normalizar por pasos locales
            normalized_delta = delta / max(client_steps[client_idx], 1)
            
            # Ponderar por tamaño del dataset
            weight = client_sizes[client_idx] / total_samples
            weighted_delta_sum += normalized_delta * weight
        
        # Actualizar peso global
        w_new = w_prev + eta * weighted_delta_sum
        new_weights.append(w_new)
    
    return new_weights


def add_differential_privacy_noise(weights: List[np.ndarray], 
                                   sigma: float = 0.01) -> List[np.ndarray]:
    """
    Agrega ruido gaussiano para privacidad diferencial.
    
    Args:
        weights: Pesos del modelo
        sigma: Desviación estándar del ruido (controla el nivel de privacidad)
    
    Returns:
        Pesos con ruido agregado
    """
    noisy_weights = []
    
    for w in weights:
        noise = np.random.normal(0.0, sigma, size=w.shape).astype(np.float32)
        noisy_weights.append(w + noise)
    
    return noisy_weights


# Mapping de nombres a funciones
AGGREGATION_METHODS = {
    'fedavg': fedavg,
    'fedavgm': fedavgm,
    'fednova': fednova
}


def get_aggregator(method_name: str):
    """
    Obtiene la función de agregación por nombre.
    
    Args:
        method_name: Nombre del método ('fedavg', 'fedavgm', 'fednova')
    
    Returns:
        Función de agregación
    """
    if method_name not in AGGREGATION_METHODS:
        raise ValueError(f"Método de agregación '{method_name}' no reconocido. "
                        f"Opciones: {list(AGGREGATION_METHODS.keys())}")
    
    return AGGREGATION_METHODS[method_name]
