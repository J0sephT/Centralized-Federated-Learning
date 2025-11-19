"""
Cliente para Federated Learning
Se conecta al servidor, entrena localmente y envía actualizaciones.

CAMBIOS EN ESTA VERSIÓN:
- ✅ Eliminado contador local de rondas
- ✅ Obtiene número de ronda del servidor
- ✅ Mejor sincronización con barrera explícita
"""
import os
import sys
import time
import logging
import argparse
import requests
from typing import Dict

import numpy as np

# Agregar directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar módulos compartidos
from shared.model import create_model, get_model_weights, set_model_weights, serialize_weights, deserialize_weights
from shared.data_utils import load_client_data


# Configuración de logging
def setup_logging(client_id: str):
    """Configura logging para el cliente."""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/client_{client_id}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(f'FL-Client-{client_id}')


class FederatedClient:
    """Cliente para Federated Learning Centralizado."""
    
    def __init__(self, 
                 client_id: str,
                 server_url: str,
                 data_dir: str = 'data',
                 local_epochs: int = 2,
                 batch_size: int = 32):
        """
        Inicializa el cliente federado.
        
        Args:
            client_id: ID único del cliente
            server_url: URL del servidor (ej: http://server:5000)
            data_dir: Directorio con los datos del cliente
            local_epochs: Épocas de entrenamiento local por ronda
            batch_size: Tamaño del batch
        """
        self.client_id = client_id
        self.server_url = server_url
        self.data_dir = data_dir
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        
        self.logger = setup_logging(client_id)
        
        # Modelo local
        self.model = None
        self.model_params = None
        
        # Datos locales
        self.X_train = None
        self.y_train = None
        self.num_samples = 0
        
        self.logger.info(f"Cliente inicializado: {client_id}")
    
    def load_data(self):
        """Carga los datos locales del cliente."""
        try:
            client_file_id = f"client_{self.client_id}"
            df = load_client_data(client_file_id, self.data_dir)
            
            # Separar features y target
            data_cols = [f"DLC{i}" for i in range(8)]
            self.X_train = df[data_cols].to_numpy(dtype='float32')
            self.y_train = df['target'].to_numpy()
            self.num_samples = len(self.y_train)
            
            self.logger.info(f"Datos cargados: {self.num_samples} muestras")
            self.logger.info(f"   Distribución: {dict(zip(*np.unique(self.y_train, return_counts=True)))}")
            
        except Exception as e:
            self.logger.error(f"Error cargando datos: {e}")
            raise
    
    def register_with_server(self) -> bool:
        """Registra el cliente con el servidor."""
        url = f"{self.server_url}/register"
        
        max_retries = 10
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json={'client_id': self.client_id}, timeout=5)
                
                if response.status_code == 200:
                    self.logger.info(f"✅ Registrado exitosamente con el servidor")
                    return True
                elif response.status_code == 400:
                    # Ya registrado o error
                    data = response.json()
                    self.logger.warning(f"⚠️ {data.get('error', 'Error desconocido')}")
                    return False
                else:
                    self.logger.warning(f"Intento {attempt+1}/{max_retries}: código {response.status_code}")
            
            except requests.exceptions.ConnectionError:
                self.logger.warning(f"Intento {attempt+1}/{max_retries}: servidor no disponible")
            except Exception as e:
                self.logger.warning(f"Intento {attempt+1}/{max_retries}: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        self.logger.error(f"❌ No se pudo registrar después de {max_retries} intentos")
        return False
    
    def get_global_weights(self) -> bool:
        """Obtiene los pesos del modelo global desde el servidor."""
        url = f"{self.server_url}/get_weights"
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Si es la primera vez, inicializar modelo
                if self.model is None:
                    self.model_params = data['model_params']
                    self.model = create_model(**self.model_params)
                    self.logger.info(f"Modelo inicializado: {self.model_params}")
                
                # Actualizar pesos
                weights = deserialize_weights(data['weights'])
                set_model_weights(self.model, weights)
                
                self.logger.info(f"📥 Pesos globales recibidos (ronda {data['round']})")
                return True
            
            else:
                self.logger.error(f"❌ Error obteniendo pesos: código {response.status_code}")
                return False
        
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo pesos: {e}")
            return False
    
    def get_server_status(self) -> Dict:
        """
        Obtiene el estado actual del servidor.
        
        Returns:
            Diccionario con el estado del servidor
        """
        url = f"{self.server_url}/status"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        
        except Exception as e:
            return {}
    
    def train_local_model(self) -> Dict:
        """
        Entrena el modelo localmente.
        
        Returns:
            Diccionario con métricas de entrenamiento
        """
        self.logger.info(f"🏋️ Entrenando localmente: {self.local_epochs} épocas, batch={self.batch_size}")
        
        start_time = time.time()
        
        # Entrenar
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=0,
            shuffle=True
        )
        
        training_time = time.time() - start_time
        
        # Calcular número de pasos (para FedNova)
        steps_per_epoch = len(self.X_train) // self.batch_size
        total_steps = steps_per_epoch * self.local_epochs
        
        # Métricas finales
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        
        self.logger.info(f"✅ Entrenamiento completado en {training_time:.2f}s")
        self.logger.info(f"   Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
        
        return {
            'training_time': training_time,
            'steps': total_steps,
            'loss': final_loss,
            'accuracy': final_accuracy
        }
    
    def send_update_to_server(self, training_metrics: Dict) -> bool:
        """
        Envía la actualización del modelo al servidor.
        
        Args:
            training_metrics: Métricas del entrenamiento local
        
        Returns:
            True si fue exitoso
        """
        url = f"{self.server_url}/submit_update"
        
        # Obtener pesos actuales
        weights = get_model_weights(self.model)
        weights_serialized = serialize_weights(weights)
        
        # Preparar payload
        payload = {
            'client_id': self.client_id,
            'weights': weights_serialized,
            'num_samples': self.num_samples,
            'training_steps': training_metrics['steps']
        }
        
        try:
            self.logger.info(f"📤 Enviando actualización al servidor...")
            
            response = requests.post(url, json=payload, timeout=180)  # 3 minutos
            
            if response.status_code == 200:
                data = response.json()
                self.logger.info(f"✅ Actualización aceptada")
                self.logger.info(f"   Actualizaciones recibidas: {data['updates_received']}")
                return True
            else:
                self.logger.error(f"❌ Actualización rechazada: código {response.status_code}")
                return False
        
        except Exception as e:
            self.logger.error(f"❌ Error enviando actualización: {e}")
            return False
    
    def wait_for_round_start(self, check_interval: int = 5) -> bool:
        """
        Espera a que el servidor inicie una nueva ronda.
        
        Args:
            check_interval: Segundos entre cada verificación
        
        Returns:
            True si hay una ronda activa, False si terminó el entrenamiento
        """
        url = f"{self.server_url}/status"
        
        while True:
            try:
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Verificar si el entrenamiento terminó
                    if data['current_round'] >= data['total_rounds'] and not data['is_training']:
                        self.logger.info(f"🏁 Entrenamiento completado!")
                        return False
                    
                    # Verificar si hay una ronda activa
                    if data['is_training']:
                        self.logger.info(f"🔔 Ronda {data['current_round']}/{data['total_rounds']} activa")
                        return True
                
                time.sleep(check_interval)
            
            except Exception as e:
                self.logger.warning(f"⚠️ Error verificando estado: {e}")
                time.sleep(check_interval)
    
    def run(self):
        """Ejecuta el ciclo principal del cliente."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🚀 Iniciando Cliente {self.client_id}")
        self.logger.info(f"{'='*60}\n")
        
        # 1. Cargar datos
        self.load_data()
        
        # 2. Registrarse con el servidor
        if not self.register_with_server():
            self.logger.error("❌ No se pudo registrar. Abortando.")
            return
        
        # 3. Esperar a que todos los clientes se registren
        self.logger.info("⏳ Esperando que todos los clientes se registren...")
        time.sleep(10)  # Dar tiempo para que otros clientes se registren
        
        # 4. Ciclo de entrenamiento federado
        # ✅ NO MANTENER contador local - obtener del servidor
        
        while True:
            # Esperar a que inicie una nueva ronda
            if not self.wait_for_round_start():
                break  # Entrenamiento completado
            
            # ✅ OBTENER número de ronda del servidor
            status = self.get_server_status()
            current_round = status.get('current_round', 0)
            total_rounds = status.get('total_rounds', 0)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📍 RONDA {current_round}/{total_rounds}")
            self.logger.info(f"{'='*60}")
            
            # Obtener pesos globales
            if not self.get_global_weights():
                self.logger.error("❌ No se pudieron obtener los pesos. Continuando...")
                continue
            
            # Entrenar localmente
            training_metrics = self.train_local_model()
            
            # Enviar actualización
            if not self.send_update_to_server(training_metrics):
                self.logger.error("❌ No se pudo enviar la actualización. Continuando...")
                continue
            
            self.logger.info(f"✅ Ronda {current_round} completada")
            
            # BARRERA DE SINCRONIZACIÓN: Esperar a que todos completen
            self.logger.info("⏳ Esperando a que todos los clientes completen la ronda...")
            wait_timeout = 300  # 5 minutos máximo
            wait_start = time.time()
            
            while time.time() - wait_start < wait_timeout:
                try:
                    status = self.get_server_status()
                    updates = status.get('updates_received', 0)
                    expected = status.get('expected_clients', 3)
                    server_round = status.get('current_round', 0)
                    is_training = status.get('is_training', True)
                    aggregation_done = status.get('aggregation_done', False)
                    
                    # Si el servidor avanzó de ronda, salir
                    if server_round > current_round:
                        self.logger.info(f"🔄 Servidor avanzó a ronda {server_round}")
                        break
                    
                    # Si todos enviaron actualizaciones Y la agregación terminó
                    if updates >= expected and aggregation_done and not is_training:
                        self.logger.info(f"✅ Agregación completada ({updates}/{expected})")
                        break
                    
                    time.sleep(3)
                except Exception as e:
                    self.logger.error(f"❌ Error verificando sincronización: {e}")
                    time.sleep(5)
            else:
                self.logger.warning(f"⚠️ Timeout esperando sincronización")
            
            # Pausa adicional para asegurar que servidor termine agregación
            time.sleep(5)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🏁 Cliente {self.client_id} finalizó exitosamente")
        self.logger.info(f"{'='*60}\n")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Cliente de Federated Learning')
    parser.add_argument('--client_id', type=str, required=True, help='ID del cliente')
    parser.add_argument('--server_url', type=str, default='http://localhost:5000', help='URL del servidor')
    parser.add_argument('--data_dir', type=str, default='data', help='Directorio de datos')
    parser.add_argument('--local_epochs', type=int, default=2, help='Épocas locales por ronda')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')
    
    args = parser.parse_args()
    
    # Crear y ejecutar cliente
    client = FederatedClient(
        client_id=args.client_id,
        server_url=args.server_url,
        data_dir=args.data_dir,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size
    )
    
    client.run()


if __name__ == '__main__':
    main()
