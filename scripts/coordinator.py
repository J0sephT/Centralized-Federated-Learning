"""
Script Coordinador para Federated Learning
Inicia las rondas de entrenamiento automáticamente.
"""
import time
import requests
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FL-Coordinator')


def wait_for_server(server_url: str, max_retries: int = 30, retry_delay: int = 2):
    """Espera a que el servidor esté disponible."""
    url = f"{server_url}/health"
    
    logger.info(f"⏳ Esperando que el servidor esté disponible...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ Servidor disponible!")
                return True
        except:
            pass
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    logger.error(f"❌ Servidor no disponible después de {max_retries} intentos")
    return False


def wait_for_clients(server_url: str, expected_clients: int, timeout: int = 120):
    """Espera a que todos los clientes se registren."""
    url = f"{server_url}/status"
    
    logger.info(f"⏳ Esperando {expected_clients} clientes...")
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                registered = data['registered_clients']
                
                if registered >= expected_clients:
                    logger.info(f"✅ Todos los clientes registrados! ({registered}/{expected_clients})")
                    return True
                
                if int(time.time() - start_time) % 10 == 0:
                    logger.info(f"   Clientes registrados: {registered}/{expected_clients}")
        except:
            pass
        
        time.sleep(2)
    
    logger.error(f"❌ Timeout esperando clientes")
    return False


def start_training_round(server_url: str) -> bool:
    """Inicia una ronda de entrenamiento."""
    url = f"{server_url}/start_round"
    
    try:
        response = requests.post(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['status'] == 'completed':
                logger.info(f"🏁 Entrenamiento completado!")
                return False
            
            logger.info(f"🚀 Ronda {data['round']}/{data['total_rounds']} iniciada")
            return True
        else:
            logger.error(f"❌ Error iniciando ronda: código {response.status_code}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error iniciando ronda: {e}")
        return False


def wait_for_aggregation(server_url: str, expected_clients: int, timeout: int = 300):
    """Espera a que se complete la agregación."""
    url = f"{server_url}/status"
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Si ya no está entrenando, la agregación terminó
                if not data['is_training']:
                    logger.info(f"✅ Agregación completada!")
                    return True
                
                # Mostrar progreso
                if int(time.time() - start_time) % 10 == 0:
                    updates = data['updates_received']
                    logger.info(f"   Esperando agregación... ({updates}/{expected_clients} actualizaciones)")
        except:
            pass
        
        time.sleep(2)
    
    logger.error(f"❌ Timeout esperando agregación")
    return False


def run_coordinator(server_url: str, num_clients: int, num_rounds: int):
    """Ejecuta el coordinador de FL."""
    logger.info(f"\n{'='*60}")
    logger.info(f"🌐 COORDINADOR DE FEDERATED LEARNING")
    logger.info(f"{'='*60}\n")
    logger.info(f"Servidor: {server_url}")
    logger.info(f"Clientes: {num_clients}")
    logger.info(f"Rondas: {num_rounds}")
    logger.info(f"{'='*60}\n")
    
    # 1. Esperar servidor
    if not wait_for_server(server_url):
        return
    
    # 2. Esperar clientes
    if not wait_for_clients(server_url, num_clients):
        return
    
    # 3. Ejecutar rondas de entrenamiento
    for round_num in range(num_rounds):
        logger.info(f"\n{'='*60}")
        logger.info(f"📍 INICIANDO RONDA {round_num + 1}/{num_rounds}")
        logger.info(f"{'='*60}")
        
        # Iniciar ronda
        if not start_training_round(server_url):
            break  # Entrenamiento completado o error
        
        # Esperar agregación
        if not wait_for_aggregation(server_url, num_clients):
            logger.error(f"❌ Error en ronda {round_num + 1}")
            break
        
        logger.info(f"✅ Ronda {round_num + 1} completada\n")
        
        # Pequeña pausa entre rondas
        time.sleep(3)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 ENTRENAMIENTO FEDERADO COMPLETADO")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Coordinador de Federated Learning')
    parser.add_argument('--server_url', type=str, default='http://localhost:5000',
                       help='URL del servidor')
    parser.add_argument('--num_clients', type=int, default=3,
                       help='Número de clientes esperados')
    parser.add_argument('--num_rounds', type=int, default=10,
                       help='Número de rondas de entrenamiento')
    
    args = parser.parse_args()
    
    run_coordinator(args.server_url, args.num_clients, args.num_rounds)


if __name__ == '__main__':
    main()
