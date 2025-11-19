# 🌐 Deployment en Dispositivos Reales

Guía completa para ejecutar Federated Learning en dispositivos físicos heterogéneos.

---

## 📋 Requisitos Previos

### Hardware Soportado

| Dispositivo | Rol | Requisitos Mínimos |
|-------------|-----|-------------------|
| PC/Laptop | Servidor | 4GB RAM, Python 3.8+ |
| Raspberry Pi 4 | Cliente | 2GB RAM, Raspbian/Ubuntu |
| PC/Laptop | Cliente | 2GB RAM, Python 3.8+ |
| Android (Termux) | Cliente | 2GB RAM, Termux instalado |

### Software Necesario

**Todos los dispositivos:**
- Python 3.8+
- pip
- Conexión a la misma red (WiFi/LAN)

---

## 🔧 Setup por Dispositivo

### 1️⃣ Servidor (PC Principal)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Preparar y distribuir datos
python scripts/prepare_data.py \
    --csv_path data/CAN_HCRL_OTIDS_UB.csv \
    --num_clients 3 \
    --distribution noniid \
    --alpha 0.5 \
    --output_dir data/

# 3. Verificar IP del servidor
hostname -I  # Linux/Mac
ipconfig     # Windows

# Ejemplo: 192.168.1.100

# 4. Configurar firewall (permitir puerto 5000)
sudo ufw allow 5000/tcp  # Linux
# Windows: Agregar regla en Windows Firewall

# 5. Iniciar servidor
export NUM_CLIENTS=3
export NUM_ROUNDS=10
export AGGREGATION_METHOD=fedavg
export DATASET_PATH=data/CAN_HCRL_OTIDS_UB.csv

python server/server.py

# El servidor estará en: http://192.168.1.100:5000
```

---

### 2️⃣ Cliente en Raspberry Pi

```bash
# 1. Actualizar sistema
sudo apt update && sudo apt upgrade -y

# 2. Instalar Python 3.8+ (si no está instalado)
sudo apt install python3 python3-pip -y

# 3. Instalar dependencias ligeras
pip3 install tensorflow-cpu==2.13.0  # Versión CPU
pip3 install flask requests numpy pandas scikit-learn

# 4. Copiar proyecto (desde el servidor)
scp -r usuario@192.168.1.100:/path/to/federated-learning-project .
cd federated-learning-project

# O clonar desde Git:
git clone <tu-repo>
cd federated-learning-project

# 5. Copiar datos del cliente
# Opción A: Desde el servidor
scp usuario@192.168.1.100:/path/to/federated-learning-project/data/client_0_data.csv data/

# Opción B: Usar script de deployment
./scripts/deploy_client.sh --server_ip 192.168.1.100 --client_id 0

# 6. Ejecutar cliente
python3 client/client.py \
    --client_id 0 \
    --server_url http://192.168.1.100:5000 \
    --data_dir data/ \
    --local_epochs 2 \
    --batch_size 32
```

---

### 3️⃣ Cliente en PC/Laptop

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Copiar datos del cliente
scp usuario@192.168.1.100:/path/to/federated-learning-project/data/client_1_data.csv data/

# 3. Ejecutar cliente
python client/client.py \
    --client_id 1 \
    --server_url http://192.168.1.100:5000 \
    --data_dir data/
```

---

### 4️⃣ Cliente en Android (Termux)

```bash
# 1. Instalar Termux desde F-Droid o Google Play

# 2. Actualizar paquetes
pkg update && pkg upgrade

# 3. Instalar Python
pkg install python

# 4. Instalar dependencias
pip install tensorflow requests numpy pandas scikit-learn flask

# 5. Descargar proyecto
pkg install git
git clone <tu-repo>
cd federated-learning-project

# 6. Copiar datos
# Usar wget o curl desde servidor HTTP:
wget http://192.168.1.100:8000/client_2_data.csv -O data/client_2_data.csv

# 7. Ejecutar cliente
python client/client.py \
    --client_id 2 \
    --server_url http://192.168.1.100:5000 \
    --data_dir data/
```

---

## 🚀 Ejecución Completa

### Paso 1: Iniciar Servidor

**En PC principal (192.168.1.100):**

```bash
python server/server.py
```

Esperar ver:
```
2025-11-18 20:00:00,000 - FL-Server - INFO - Servidor inicializado: 3 clientes, 10 rondas, fedavg
2025-11-18 20:00:00,001 - FL-Server - INFO - Servidor Flask iniciando en http://0.0.0.0:5000
```

---

### Paso 2: Iniciar Clientes

**En Raspberry Pi (Cliente 0):**
```bash
python3 client/client.py --client_id 0 --server_url http://192.168.1.100:5000 --data_dir data/
```

**En PC/Laptop (Cliente 1):**
```bash
python client/client.py --client_id 1 --server_url http://192.168.1.100:5000 --data_dir data/
```

**En Android/Termux (Cliente 2):**
```bash
python client/client.py --client_id 2 --server_url http://192.168.1.100:5000 --data_dir data/
```

Cada cliente mostrará:
```
2025-11-18 20:00:05,000 - FL-Client-0 - INFO - Cliente inicializado: 0
2025-11-18 20:00:05,100 - FL-Client-0 - INFO - Datos cargados: 14783 muestras
2025-11-18 20:00:05,200 - FL-Client-0 - INFO - Registrado exitosamente con el servidor
```

---

### Paso 3: Iniciar Coordinador

**En cualquier dispositivo (puede ser el servidor):**

```bash
python scripts/coordinator.py \
    --server_url http://192.168.1.100:5000 \
    --num_clients 3 \
    --num_rounds 10
```

El coordinador iniciará automáticamente las rondas de entrenamiento.

---

## 📊 Monitoreo

### Verificar Estado del Sistema

```bash
# Desde cualquier dispositivo
curl http://192.168.1.100:5000/status

# Respuesta:
{
  "current_round": 1,
  "total_rounds": 10,
  "is_training": true,
  "registered_clients": 3,
  "expected_clients": 3,
  "updates_received": 2
}
```

### Ver Logs en Tiempo Real

**Servidor:**
```bash
tail -f logs/server.log
```

**Clientes:**
```bash
tail -f logs/client_0.log
tail -f logs/client_1.log
tail -f logs/client_2.log
```

---

## 🐛 Troubleshooting

### ❌ Error: "Connection refused"

**Causa:** Cliente no puede conectarse al servidor.

**Soluciones:**
1. Verificar IP del servidor: `hostname -I`
2. Verificar que el servidor esté corriendo: `curl http://<IP>:5000/health`
3. Verificar firewall: `sudo ufw status`
4. Hacer ping: `ping 192.168.1.100`

---

### ❌ Error: "No se encontraron datos para client_X"

**Causa:** Datos del cliente no están en el dispositivo.

**Soluciones:**
```bash
# Copiar datos desde servidor
scp usuario@192.168.1.100:/path/to/data/client_0_data.csv data/

# O servir datos desde servidor vía HTTP
# En servidor:
cd data/
python3 -m http.server 8000

# En cliente:
wget http://192.168.1.100:8000/client_0_data.csv -O data/client_0_data.csv
```

---

### ❌ Error: "TensorFlow no compatible" (Raspberry Pi)

**Causa:** TensorFlow 2.x completo es pesado para ARM.

**Solución:**
```bash
# Usar versión CPU optimizada
pip3 install tensorflow-cpu==2.13.0

# O versión lite
pip3 install tflite-runtime
```

---

### ❌ Error: "Timeout esperando sincronización"

**Causa:** Un cliente se desconectó o está muy lento.

**Soluciones:**
1. Verificar que todos los clientes estén activos
2. Reducir `local_epochs` en clientes lentos:
   ```bash
   python client/client.py --client_id 2 --local_epochs 1
   ```
3. Aumentar timeout en `client.py` (línea 235):
   ```python
   wait_timeout = 600  # 10 minutos
   ```

---

### ❌ Error: "Memory error" (Android/Raspberry Pi)

**Causa:** Modelo muy grande para dispositivo.

**Soluciones:**
1. Reducir batch size:
   ```bash
   python client/client.py --client_id 0 --batch_size 16
   ```
2. Usar modelo más pequeño (editar `shared/model.py`):
   ```python
   Dense(32, activation='relu'),  # En lugar de 64
   Dense(16, activation='relu'),  # En lugar de 32
   ```

---

## 📸 Captura de Evidencia para Paper

### Screenshots Recomendados

1. **Setup completo:**
   - Foto de todos los dispositivos conectados
   - Screenshot de cada terminal activa

2. **Registro de clientes:**
   ```
   2025-11-18 20:00:10,000 - FL-Server - INFO - Cliente registrado: 0 (Raspberry Pi)
   2025-11-18 20:00:11,000 - FL-Server - INFO - Cliente registrado: 1 (PC)
   2025-11-18 20:00:12,000 - FL-Server - INFO - Cliente registrado: 2 (Android)
   ```

3. **Entrenamiento activo:**
   - Logs de servidor mostrando agregación
   - Logs de cada cliente mostrando entrenamiento local

4. **Métricas finales:**
   - Gráficas de accuracy/loss
   - Tabla de resultados

---

## 🎯 Mejores Prácticas

### Red Estable

- Usar conexión por cable (Ethernet) cuando sea posible
- Si usas WiFi, evitar WiFi público
- Todos en la misma red local

### Sincronización de Tiempo

```bash
# Sincronizar reloj en todos los dispositivos
sudo apt install ntp
sudo systemctl start ntp
```

### Backup de Datos

```bash
# Antes de cada experimento
cp -r data/ data_backup/
cp -r logs/ logs_backup/
```

---

## 📝 Checklist de Deployment

### Antes de Empezar

- [ ] Todos los dispositivos en la misma red
- [ ] IP del servidor conocida y accesible
- [ ] Firewall configurado (puerto 5000 abierto)
- [ ] Python 3.8+ instalado en todos los dispositivos
- [ ] Dependencias instaladas
- [ ] Datos distribuidos a cada cliente
- [ ] Suficiente espacio en disco (500MB mínimo)

### Durante Ejecución

- [ ] Servidor corriendo y accesible
- [ ] Todos los clientes conectados
- [ ] Logs guardándose correctamente
- [ ] Monitor de recursos (RAM/CPU) activo

### Después de Terminar

- [ ] Métricas guardadas en `results/metrics.json`
- [ ] Logs respaldados
- [ ] Screenshots/fotos capturadas
- [ ] Resultados visualizados

---

## 🚀 Próximos Pasos

Después del deployment exitoso:

1. **Visualizar resultados:**
   ```bash
   python scripts/visualize_results.py
   ```

2. **Comparar métodos:**
   - Ejecutar con `fedavg`, `fedavgm`, `fednova`
   - Comparar convergencia

3. **Documentar en paper:**
   - Ver [PAPER_GUIDE.md](PAPER_GUIDE.md)

---

## 💡 Tips para Presentación/Examen

1. **Demo en vivo:** Tener todo preparado y probado antes
2. **Backup:** Tener logs y screenshots por si falla la demo
3. **Explicar arquitectura:** Mostrar diagrama de red
4. **Destacar heterogeneidad:** Raspberry Pi + PC + Android demuestra FL real
5. **Métricas:** Mostrar gráficas de convergencia

---

¿Preguntas? Ver [README.md](README.md) o crear un issue en GitHub.
