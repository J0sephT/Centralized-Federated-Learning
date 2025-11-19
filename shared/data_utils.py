"""
Utilidades para carga y distribución de datos (IID y Non-IID)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from typing import Dict, List, Tuple
import os


def load_and_preprocess_data(csv_path: str, 
                             test_size: float = 0.2,
                             sample_size: int = None,
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga y preprocesa el dataset CAN-HCRL-OTIDS.
    
    Args:
        csv_path: Ruta al archivo CSV
        test_size: Proporción de datos para test
        sample_size: Número de muestras a usar (None = todas)
        random_state: Semilla para reproducibilidad
    
    Returns:
        Tupla (df_train, df_test)
    """
    print(f"📂 Cargando dataset desde: {csv_path}")
    
    # Cargar CSV
    df = pd.read_csv(csv_path)
    print(f"   Dataset original: {df.shape}")
    
    # Asegurarse de que target sea entero
    if not pd.api.types.is_integer_dtype(df['target']):
        df['target'], _ = pd.factorize(df['target'])
    
    # Muestreo estratificado si se especifica
    if sample_size is not None and sample_size < len(df):
        print(f"   Muestreando {sample_size} filas...")
        df, _ = train_test_split(
            df, 
            train_size=sample_size,
            stratify=df['target'],
            random_state=random_state
        )
    
    # Columnas de datos
    data_cols = [f"DLC{i}" for i in range(8)]
    
    # Normalizar (MinMaxScaler)
    scaler = MinMaxScaler()
    df[data_cols] = df[data_cols].astype('float32')
    df[data_cols] = scaler.fit_transform(df[data_cols])
    
    # Split train/test
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df['target'],
        random_state=random_state
    )
    
    print(f"   Train: {df_train.shape} | Test: {df_test.shape}")
    print(f"   Clases: {sorted(df_train['target'].unique())}")
    
    return df_train, df_test


def distribute_data_iid(df: pd.DataFrame, 
                       num_clients: int,
                       random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Distribución IID (Independiente e Idénticamente Distribuida).
    Cada cliente recibe una muestra estratificada con proporciones similares de cada clase.
    
    Args:
        df: DataFrame con los datos de entrenamiento
        num_clients: Número de clientes
        random_state: Semilla aleatoria
    
    Returns:
        Diccionario {client_id: dataframe}
    """
    print(f"\n📊 Distribuyendo datos IID entre {num_clients} clientes...")
    
    labels = sorted(df['target'].unique().tolist())
    clients_data = {f"client_{i}": [] for i in range(num_clients)}
    
    # Dividir cada clase equitativamente
    for label in labels:
        df_label = df[df['target'] == label]
        df_label = shuffle(df_label, random_state=random_state)
        
        # Split en num_clients partes
        splits = np.array_split(df_label, num_clients)
        
        for client_idx, split_df in enumerate(splits):
            clients_data[f"client_{client_idx}"].append(split_df)
    
    # Concatenar y mezclar
    for client_id in clients_data:
        clients_data[client_id] = pd.concat(clients_data[client_id], ignore_index=True)
        clients_data[client_id] = shuffle(clients_data[client_id], random_state=random_state)
    
    # Mostrar distribución
    _print_distribution(clients_data, labels)
    
    return clients_data


def distribute_data_noniid(df: pd.DataFrame,
                          num_clients: int,
                          alpha: float = 0.5,
                          random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Distribución Non-IID usando Dirichlet distribution.
    Cada cliente tiene una distribución sesgada de clases.
    
    Args:
        df: DataFrame con los datos de entrenamiento
        num_clients: Número de clientes
        alpha: Parámetro de concentración de Dirichlet
               - alpha < 1: Muy heterogéneo (cada cliente tiene pocas clases)
               - alpha = 1: Moderadamente heterogéneo
               - alpha > 1: Más homogéneo (acercándose a IID)
        random_state: Semilla aleatoria
    
    Returns:
        Diccionario {client_id: dataframe}
    """
    print(f"\n📊 Distribuyendo datos Non-IID entre {num_clients} clientes (alpha={alpha})...")
    
    np.random.seed(random_state)
    labels = sorted(df['target'].unique().tolist())
    num_classes = len(labels)
    
    # Generar proporciones con Dirichlet
    # Cada fila = distribución de clases para un cliente
    proportions = np.random.dirichlet([alpha] * num_classes, size=num_clients)
    
    clients_data = {f"client_{i}": [] for i in range(num_clients)}
    
    # Distribuir cada clase según las proporciones
    for label_idx, label in enumerate(labels):
        df_label = df[df['target'] == label]
        df_label = shuffle(df_label, random_state=random_state + label)
        
        # Calcular cuántas muestras va a cada cliente
        total_samples = len(df_label)
        client_samples = (proportions[:, label_idx] * total_samples).astype(int)
        
        # Ajustar para que sumen exactamente total_samples
        diff = total_samples - client_samples.sum()
        client_samples[0] += diff
        
        # Asignar muestras
        start_idx = 0
        for client_idx, n_samples in enumerate(client_samples):
            if n_samples > 0:
                end_idx = start_idx + n_samples
                clients_data[f"client_{client_idx}"].append(
                    df_label.iloc[start_idx:end_idx]
                )
                start_idx = end_idx
    
    # Concatenar y mezclar
    for client_id in clients_data:
        if len(clients_data[client_id]) > 0:
            clients_data[client_id] = pd.concat(clients_data[client_id], ignore_index=True)
            clients_data[client_id] = shuffle(clients_data[client_id], random_state=random_state)
        else:
            # Cliente sin datos (raro pero posible con alpha muy bajo)
            clients_data[client_id] = pd.DataFrame(columns=df.columns)
    
    # Mostrar distribución
    _print_distribution(clients_data, labels)
    
    return clients_data


def _print_distribution(clients_data: Dict[str, pd.DataFrame], labels: List[int]):
    """Imprime la distribución de clases por cliente."""
    print("\n📊 Distribución de clases por cliente:")
    print("-" * 80)
    
    for client_id, df_client in clients_data.items():
        if len(df_client) == 0:
            print(f"{client_id}: Sin datos")
            continue
            
        class_counts = df_client['target'].value_counts().sort_index()
        total = len(df_client)
        
        dist_str = " | ".join([
            f"Clase {label}: {class_counts.get(label, 0):4d} ({100*class_counts.get(label, 0)/total:5.1f}%)"
            for label in labels
        ])
        
        print(f"{client_id}: {total:5d} muestras → {dist_str}")
    
    print("-" * 80)


def save_client_data(clients_data: Dict[str, pd.DataFrame], output_dir: str):
    """
    Guarda los datos de cada cliente en archivos CSV separados.
    
    Args:
        clients_data: Diccionario con datos de clientes
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for client_id, df_client in clients_data.items():
        output_path = os.path.join(output_dir, f"{client_id}_data.csv")
        df_client.to_csv(output_path, index=False)
        print(f"💾 {client_id}: {len(df_client)} muestras → {output_path}")


def load_client_data(client_id: str, data_dir: str) -> pd.DataFrame:
    """
    Carga los datos de un cliente específico.
    
    Args:
        client_id: ID del cliente (ej: "client_0")
        data_dir: Directorio donde están los datos
    
    Returns:
        DataFrame con los datos del cliente
    """
    file_path = os.path.join(data_dir, f"{client_id}_data.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontraron datos para {client_id} en {file_path}")
    
    df = pd.read_csv(file_path)
    return df
