# 📝 Ejemplos de Uso

## 1. Simulación Local (Sin Docker)

### Paso 1: Preparar datos

```bash
python scripts/prepare_data.py \
    --csv_path data/CAN_HCRL_OTIDS_UB.csv \
    --num_clients 3 \
    --distribution noniid \
    --alpha 0.5
```

### Paso 2: Iniciar servidor

```bash
# Terminal 1
export NUM_CLIENTS=3
export NUM_ROUNDS=10
export AGGREGATION_METHOD=fedavg
export DATASET_PATH=data/CAN_HCRL_OTIDS_UB.csv

python server/server.py
```

### Paso 3: Iniciar clientes

```bash
# Terminal 2
python client/client.py \
    --client_id 0 \
    --server_url http://localhost:5000 \
    --data_dir data/

# Terminal 3
python client/client.py \
    --client_id 1 \
    --server_url http://localhost:5000 \
    --data_dir data/

# Terminal 4
python client/client.py \
    --client_id 2 \
    --server_url http://localhost:5000 \
    --data_dir data/
```

### Paso 4: Iniciar coordinador

```bash
# Terminal 5
python scripts/coordinator.py \
    --server_url http://localhost:5000 \
    --num_clients 3 \
    --num_rounds 10
```

### Paso 5: Visualizar resultados

```bash
python scripts/visualize_results.py --metrics_file results/metrics.json
```

---

## 2. Deployment Real - Raspberry Pi

### En el servidor (PC principal)

```bash
# 1. Obtener IP
hostname -I  # Ejemplo: 192.168.1.100

# 2. Abrir firewall
sudo ufw allow 5000/tcp

# 3. Preparar datos
python scripts/prepare_data.py \
    --csv_path data/CAN_HCRL_OTIDS_UB.csv \
    --num_clients 3 \
    --distribution noniid \
    --alpha 0.5

# 4. Iniciar servidor
python server/server.py
```

### En Raspberry Pi (Cliente 0)

```bash
# 1. Instalar dependencias
pip3 install tensorflow-cpu requests numpy pandas scikit-learn flask

# 2. Copiar proyecto
scp -r usuario@192.168.1.100:/path/to/federated-learning-project .

# 3. Ejecutar cliente
python3 client/client.py \
    --client_id 0 \
    --server_url http://192.168.1.100:5000 \
    --data_dir data/
```

### Iniciar coordinador

```bash
python scripts/coordinator.py \
    --server_url http://192.168.1.100:5000 \
    --num_clients 3 \
    --num_rounds 10
```

---

## 3. Comparar Métodos de Agregación

### FedAvg

```bash
export AGGREGATION_METHOD=fedavg
python server/server.py
```

### FedAvgM

```bash
export AGGREGATION_METHOD=fedavgm
python server/server.py
```

### FedNova

```bash
export AGGREGATION_METHOD=fednova
python server/server.py
```

---

## 4. Distribución de Datos

### IID (uniforme)

```bash
python scripts/prepare_data.py \
    --csv_path data/CAN_HCRL_OTIDS_UB.csv \
    --num_clients 3 \
    --distribution iid
```

### Non-IID Suave (alpha=1.0)

```bash
python scripts/prepare_data.py \
    --csv_path data/CAN_HCRL_OTIDS_UB.csv \
    --num_clients 3 \
    --distribution noniid \
    --alpha 1.0
```

### Non-IID Extremo (alpha=0.1)

```bash
python scripts/prepare_data.py \
    --csv_path data/CAN_HCRL_OTIDS_UB.csv \
    --num_clients 3 \
    --distribution noniid \
    --alpha 0.1
```

---

## 5. Monitoreo en Tiempo Real

### Ver logs del servidor

```bash
tail -f logs/server.log
```

### Ver logs de clientes

```bash
tail -f logs/client_0.log
tail -f logs/client_1.log
tail -f logs/client_2.log
```

### Verificar estado

```bash
curl http://localhost:5000/status | python -m json.tool
```

---

## 6. Docker Personalizado

### Cambiar número de rondas

Edita `docker-compose.yml`:

```yaml
environment:
  - NUM_ROUNDS=20  # En lugar de 10
```

### Cambiar método de agregación

```yaml
environment:
  - AGGREGATION_METHOD=fedavgm  # En lugar de fedavg
```

### Agregar más clientes

Añade en `docker-compose.yml`:

```yaml
client_3:
  build:
    context: .
    dockerfile: Dockerfile.client
  container_name: fl_client_3
  depends_on:
    - server
  command: >
    sh -c "sleep 25 && python client/client.py --client_id 3 --server_url http://server:5000"
```

Y actualiza:

```yaml
environment:
  - NUM_CLIENTS=4  # En server
```

---

## 7. Troubleshooting

### Cliente no se conecta

```bash
# Verificar servidor
curl http://<IP_SERVIDOR>:5000/health

# Ping
ping <IP_SERVIDOR>

# Firewall
sudo ufw status
```

### Memoria insuficiente

```bash
# Reducir batch size
python client/client.py --batch_size 16

# Reducir épocas
python client/client.py --local_epochs 1
```

### Desincronización

Ya corregido en esta versión ✅

---

## 8. Experimentos para Paper

### Experimento 1: IID vs Non-IID

```bash
# Test 1: IID
python scripts/prepare_data.py --distribution iid
# Ejecutar FL y guardar resultados

# Test 2: Non-IID (alpha=0.5)
python scripts/prepare_data.py --distribution noniid --alpha 0.5
# Ejecutar FL y guardar resultados

# Test 3: Non-IID (alpha=0.1)
python scripts/prepare_data.py --distribution noniid --alpha 0.1
# Ejecutar FL y guardar resultados

# Comparar gráficas
```

### Experimento 2: Métodos de Agregación

```bash
# Test FedAvg
export AGGREGATION_METHOD=fedavg
# Ejecutar y guardar

# Test FedAvgM
export AGGREGATION_METHOD=fedavgm
# Ejecutar y guardar

# Test FedNova
export AGGREGATION_METHOD=fednova
# Ejecutar y guardar

# Comparar convergencia
```

### Experimento 3: Número de Clientes

```bash
# Test con 2 clientes
python scripts/prepare_data.py --num_clients 2
export NUM_CLIENTS=2
# Ejecutar

# Test con 5 clientes
python scripts/prepare_data.py --num_clients 5
export NUM_CLIENTS=5
# Ejecutar

# Comparar escalabilidad
```
