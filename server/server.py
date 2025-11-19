"""
Servidor Central para Federated Learning
Coordina el entrenamiento distribuido y la agregación de modelos.

CAMBIOS EN ESTA VERSIÓN:
- ✅ Bandera aggregation_done para evitar múltiples agregaciones
- ✅ Guardado automático de métricas después de cada ronda
- ✅ Mejor manejo de sincronización
"""
import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List
import numpy as np

# Flask
from flask import Flask, request, jsonify

# Agregar directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar módulos compartidos
from shared.model import create_model, serialize_weights, deserialize_weights, get_model_weights
from shared.aggregators import fedavg, fedavgm, fednova
from shared.data_utils import load_and_preprocess_data


# Configuración de logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FL-Server')


# Crear aplicación Flask
app = Flask(__name__)


class FederatedServer:
    """Servidor para Federated Learning Centralizado."""
    
    def __init__(self, 
                 num_clients: int = 3,
                 num_rounds: int = 10,
                 aggregation_method: str = 'fedavg',
                 model_params: dict = None):
        """
        Inicializa el servidor federado.
        
        Args:
            num_clients: Número esperado de clientes
            num_rounds: Número de rondas de entrenamiento
            aggregation_method: Método de agregación ('fedavg', 'fedavgm', 'fednova')
            model_params: Parámetros del modelo
        """
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.aggregation_method = aggregation_method
        self.current_round = 0
        self.is_training = False
        
        # ✅ NUEVO: Bandera para evitar múltiples agregaciones
        self.aggregation_done = False
        
        # Parámetros del modelo
        if model_params is None:
            model_params = {
                'input_shape': (8,),
                'num_classes': 4,
                'learning_rate': 0.001
            }
        self.model_params = model_params
        
        # Inicializar modelo global
        self.global_model = create_model(**model_params)
        self.global_weights = get_model_weights(self.global_model)
        
        # Estado de agregación FedAvgM
        self.momentum = None
        if aggregation_method == 'fedavgm':
            self.momentum = [np.zeros_like(w, dtype=np.float32) for w in self.global_weights]
        
        # Registro de clientes
        self.registered_clients = set()
        self.client_updates = {}  # {client_id: {weights, num_samples, steps}}
        
        # Métricas
        self.metrics_history = []
        
        # Dataset de test
        self.test_data = None
        
        logger.info(f"Servidor inicializado: {num_clients} clientes, {num_rounds} rondas, {aggregation_method}")
    
    def load_test_data(self, csv_path: str):
        """Carga el dataset de test para evaluación."""
        try:
            _, df_test = load_and_preprocess_data(csv_path, test_size=0.2)
            
            data_cols = [f"DLC{i}" for i in range(8)]
            self.test_data = {
                'X': df_test[data_cols].to_numpy(dtype='float32'),
                'y': df_test['target'].to_numpy()
            }
            logger.info(f"Test data cargado: {len(self.test_data['y'])} muestras")
        except Exception as e:
            logger.error(f"Error cargando test data: {e}")
    
    def register_client(self, client_id: str) -> bool:
        """Registra un cliente."""
        if client_id in self.registered_clients:
            return False
        
        self.registered_clients.add(client_id)
        logger.info(f"Cliente registrado: {client_id} ({len(self.registered_clients)}/{self.num_clients})")
        return True
    
    def all_clients_registered(self) -> bool:
        """Verifica si todos los clientes están registrados."""
        return len(self.registered_clients) >= self.num_clients
    
    def start_training_round(self):
        """Inicia una nueva ronda de entrenamiento."""
        if self.current_round >= self.num_rounds:
            logger.info("Entrenamiento completado!")
            return False
        
        self.current_round += 1
        self.is_training = True
        self.client_updates = {}
        
        # ✅ NUEVO: Resetear bandera de agregación
        self.aggregation_done = False
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📍 RONDA {self.current_round}/{self.num_rounds}")
        logger.info(f"{'='*60}")
        
        return True
    
    def receive_client_update(self, 
                             client_id: str, 
                             weights_serialized: List,
                             num_samples: int,
                             training_steps: int = 1):
        """
        Recibe actualización de un cliente.
        
        Args:
            client_id: ID del cliente
            weights_serialized: Pesos del modelo (serializados)
            num_samples: Número de muestras usadas
            training_steps: Número de pasos de entrenamiento
        """
        if not self.is_training:
            logger.warning(f"{client_id} envió actualización fuera de ronda")
            return False
        
        # Deserializar pesos
        weights = deserialize_weights(weights_serialized)
        
        # Guardar actualización
        self.client_updates[client_id] = {
            'weights': weights,
            'num_samples': num_samples,
            'steps': training_steps
        }
        
        logger.info(f"Actualización recibida de {client_id}: {num_samples} muestras, {training_steps} pasos")
        
        return True
    
    def all_clients_updated(self) -> bool:
        """Verifica si todos los clientes enviaron sus actualizaciones."""
        return len(self.client_updates) >= self.num_clients
    
    def aggregate_updates(self):
        """Agrega las actualizaciones de los clientes."""
        # ✅ NUEVO: Verificar que no se haya agregado ya
        if self.aggregation_done:
            logger.warning("Agregación ya realizada para esta ronda")
            return
        
        if not self.all_clients_updated():
            logger.warning(f"No todas las actualizaciones recibidas: {len(self.client_updates)}/{self.num_clients}")
            return
        
        # ✅ NUEVO: Marcar como agregado INMEDIATAMENTE para evitar múltiples llamadas
        self.aggregation_done = True
        
        logger.info(f"Agregando {len(self.client_updates)} actualizaciones...")
        
        # Extraer datos
        client_weights = [update['weights'] for update in self.client_updates.values()]
        client_sizes = [update['num_samples'] for update in self.client_updates.values()]
        client_steps = [update['steps'] for update in self.client_updates.values()]
        
        # Agregar según el método seleccionado
        start_time = time.time()
        
        if self.aggregation_method == 'fedavg':
            self.global_weights = fedavg(client_weights, client_sizes)
        
        elif self.aggregation_method == 'fedavgm':
            # Primero hacer FedAvg
            aggregated = fedavg(client_weights, client_sizes)
            # Luego aplicar momentum
            self.global_weights, self.momentum = fedavgm(
                self.global_weights, 
                aggregated, 
                self.momentum,
                beta=0.9,
                eta=1.0
            )
        
        elif self.aggregation_method == 'fednova':
            self.global_weights = fednova(
                self.global_weights,
                client_weights,
                client_steps,
                client_sizes,
                eta=1.0
            )
        
        else:
            raise ValueError(f"Método de agregación desconocido: {self.aggregation_method}")
        
        aggregation_time = time.time() - start_time
        logger.info(f"✅ Agregación completada en {aggregation_time:.3f}s")
        
        # Actualizar modelo global
        self.global_model.set_weights(self.global_weights)
        
        # Evaluar
        if self.test_data is not None:
            self._evaluate_global_model()
        
        # ✅ NUEVO: Guardar métricas después de cada ronda
        self.save_metrics()
        
        self.is_training = False
    
    def _evaluate_global_model(self):
        """Evalúa el modelo global en el dataset de test."""
        try:
            loss, accuracy = self.global_model.evaluate(
                self.test_data['X'], 
                self.test_data['y'],
                verbose=0
            )
            
            metrics = {
                'round': self.current_round,
                'accuracy': float(accuracy),
                'loss': float(loss),
                'timestamp': datetime.now().isoformat()
            }
            
            self.metrics_history.append(metrics)
            
            logger.info(f"📊 Evaluación global - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error en evaluación: {e}")
    
    def get_global_weights(self):
        """Retorna los pesos globales actuales (serializados)."""
        return serialize_weights(self.global_weights)
    
    def save_metrics(self, output_path: str = 'results/metrics.json'):
        """Guarda las métricas del entrenamiento."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            
            logger.info(f"💾 Métricas guardadas en {output_path}")
        except Exception as e:
            logger.error(f"Error guardando métricas: {e}")


# Instancia global del servidor
server = None


# ==================== ENDPOINTS DE LA API ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


@app.route('/register', methods=['POST'])
def register_client():
    """Registra un cliente en el servidor."""
    data = request.json
    client_id = data.get('client_id')
    
    if not client_id:
        return jsonify({'error': 'client_id requerido'}), 400
    
    success = server.register_client(client_id)
    
    if not success:
        return jsonify({'error': 'Cliente ya registrado'}), 400
    
    return jsonify({
        'status': 'registered',
        'client_id': client_id,
        'total_registered': len(server.registered_clients)
    })


@app.route('/status', methods=['GET'])
def get_status():
    """Retorna el estado actual del servidor."""
    return jsonify({
        'current_round': server.current_round,
        'total_rounds': server.num_rounds,
        'is_training': server.is_training,
        'registered_clients': len(server.registered_clients),
        'expected_clients': server.num_clients,
        'updates_received': len(server.client_updates),
        'aggregation_done': server.aggregation_done  # ✅ NUEVO
    })


@app.route('/get_weights', methods=['GET'])
def get_weights():
    """Retorna los pesos del modelo global."""
    weights = server.get_global_weights()
    
    return jsonify({
        'round': server.current_round,
        'weights': weights,
        'model_params': server.model_params
    })


@app.route('/submit_update', methods=['POST'])
def submit_update():
    """Recibe actualización de un cliente."""
    data = request.json
    
    client_id = data.get('client_id')
    weights = data.get('weights')
    num_samples = data.get('num_samples')
    training_steps = data.get('training_steps', 1)
    
    if not all([client_id, weights, num_samples]):
        return jsonify({'error': 'Datos incompletos'}), 400
    
    success = server.receive_client_update(client_id, weights, num_samples, training_steps)
    
    if not success:
        return jsonify({'error': 'Actualización rechazada'}), 400
    
    # ✅ MODIFICADO: Solo agregar si todos han enviado Y no se ha agregado aún
    if server.all_clients_updated() and not server.aggregation_done:
        server.aggregate_updates()
    
    return jsonify({
        'status': 'accepted',
        'round': server.current_round,
        'updates_received': len(server.client_updates)
    })


@app.route('/start_round', methods=['POST'])
def start_round():
    """Inicia una nueva ronda de entrenamiento."""
    if not server.all_clients_registered():
        return jsonify({
            'error': 'No todos los clientes registrados',
            'registered': len(server.registered_clients),
            'expected': server.num_clients
        }), 400
    
    success = server.start_training_round()
    
    if not success:
        return jsonify({'status': 'completed', 'message': 'Entrenamiento finalizado'})
    
    return jsonify({
        'status': 'started',
        'round': server.current_round,
        'total_rounds': server.num_rounds
    })


if __name__ == '__main__':
    # Configuración desde variables de entorno
    num_clients = int(os.getenv('NUM_CLIENTS', 3))
    num_rounds = int(os.getenv('NUM_ROUNDS', 10))
    aggregation_method = os.getenv('AGGREGATION_METHOD', 'fedavg')
    dataset_path = os.getenv('DATASET_PATH', 'data/CAN_HCRL_OTIDS_UB.csv')
    
    # Inicializar servidor
    server = FederatedServer(
        num_clients=num_clients,
        num_rounds=num_rounds,
        aggregation_method=aggregation_method
    )
    
    # Cargar datos de test
    if os.path.exists(dataset_path):
        server.load_test_data(dataset_path)
    else:
        logger.warning(f"⚠️  Dataset no encontrado en {dataset_path}")
    
    # Iniciar servidor Flask
    logger.info(f"🚀 Servidor Flask iniciando en http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
