# 🌐 Federated Learning - Proyecto Completo

Sistema de Federated Learning implementado desde cero con soporte para:
- ✅ Simulación con Docker (desarrollo/testing)
- ✅ Deployment real en dispositivos heterogéneos (PCs, Raspberry Pi, móviles)
- ✅ Múltiples métodos de agregación (FedAvg, FedAvgM, FedNova)
- ✅ Distribución Non-IID con Dirichlet

---

## 🚀 Quick Start

### Opción 1: Simulación con Docker (Recomendado para Testing)

```bash
# 1. Clonar y preparar datos
./quickstart.sh

# 2. Iniciar con Docker
docker-compose up --build
```

**¡Eso es todo!** El sistema comenzará automáticamente 🎉

---

### Opción 2: Deployment Real en Dispositivos Físicos

Ver **[DEPLOYMENT.md](DEPLOYMENT.md)** para instrucciones detalladas.

**Resumen rápido:**

```bash
# En el servidor (PC principal)
python scripts/prepare_data.py --num_clients 3 --distribution noniid
python server/server.py

# En cada cliente (Raspberry Pi, PC, etc.)
python client/client.py --client_id 0 --server_url http://<IP_SERVIDOR>:5000 --data_dir data/

# En el coordinador (puede ser el mismo servidor)
python scripts/coordinator.py --server_url http://localhost:5000 --num_clients 3 --num_rounds 10
```

---

## 📁 Estructura del Proyecto

```
federated-learning-project/
├── 📄 README.md              # Este archivo
├── 📄 DEPLOYMENT.md         # Guía para deployment real
├── 📄 requirements.txt      # Dependencias Python
├── 🚀 quickstart.sh         # Script de inicio rápido
│
├── 🐳 Docker (Simulación)
│   ├── docker-compose.yml
│   ├── Dockerfile.server
│   └── Dockerfile.client
│
├── 🖥️  server/
│   ├── server.py            # Servidor Flask (CORREGIDO)
│   └── Dockerfile
│
├── 💻 client/
│   ├── client.py            # Cliente FL (CORREGIDO)
│   └── Dockerfile
│
├── 📦 shared/
│   ├── model.py             # Red neuronal
│   ├── aggregators.py       # FedAvg/FedAvgM/FedNova
│   └── data_utils.py        # Distribución IID/Non-IID
│
├── 🔧 scripts/
│   ├── prepare_data.py      # Preparar y distribuir datos
│   ├── coordinator.py       # Coordinador de rondas
│   ├── deploy_server.sh     # Deploy servidor en dispositivo
│   ├── deploy_client.sh     # Deploy cliente en dispositivo
│   └── visualize_results.py # Visualizar métricas
│
├── 📊 data/                 # Datos de clientes
├── 📝 logs/                 # Logs del sistema
└── 📈 results/              # Métricas y resultados
```

---

## 🔧 Instalación Manual

### Requisitos

- Python 3.8+
- TensorFlow 2.x
- Flask
- NumPy, Pandas, scikit-learn

### Instalación

```bash
# 1. Descomprimir proyecto
unzip federated-learning-project.zip
cd federated-learning-project/

# 2. Crear entorno virtual (RECOMENDADO ✅)
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Preparar datos
python scripts/prepare_data.py \
    --csv_path data/CAN_HCRL_OTIDS_UB.csv \
    --num_clients 3 \
    --distribution noniid \
    --alpha 0.5
```

**⚠️ IMPORTANTE:** Siempre activa el entorno virtual antes de trabajar:
```bash
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

Ver **[SETUP.md](SETUP.md)** para guía detallada de instalación y troubleshooting.

---

## 🎯 Modos de Uso

### Modo 1: Simulación Local con Docker

**Ventajas:**
- ✅ Setup automático
- ✅ Aislamiento de dependencias
- ✅ Ideal para desarrollo y testing

**Uso:**
```bash
docker-compose up --build
```

---

### Modo 2: Simulación Local sin Docker

**Ventajas:**
- ✅ Más rápido (sin overhead de containers)
- ✅ Fácil debugging

**Uso:**

```bash
# Terminal 1: Servidor
python server/server.py

# Terminal 2-4: Clientes
python client/client.py --client_id 0 --server_url http://localhost:5000 --data_dir data/
python client/client.py --client_id 1 --server_url http://localhost:5000 --data_dir data/
python client/client.py --client_id 2 --server_url http://localhost:5000 --data_dir data/

# Terminal 5: Coordinator
python scripts/coordinator.py --server_url http://localhost:5000 --num_clients 3 --num_rounds 10
```

---

### Modo 3: Deployment Real en Dispositivos Heterogéneos

**Ventajas:**
- ✅ Federated Learning REAL
- ✅ Demuestra comunicación entre dispositivos
- ✅ Ideal para paper/presentación

**Ver [DEPLOYMENT.md](DEPLOYMENT.md) para guía completa.**

---

## 📊 Visualizar Resultados

Después del entrenamiento:

```bash
python scripts/visualize_results.py --metrics_file results/metrics.json
```

Genera gráficas:
- Accuracy vs Rounds
- Loss vs Rounds
- Comparación de métodos

---

## ⚙️ Configuración

### Variables de Entorno

```bash
# Servidor
export NUM_CLIENTS=3
export NUM_ROUNDS=10
export AGGREGATION_METHOD=fedavg  # fedavg, fedavgm, fednova
export DATASET_PATH=data/CAN_HCRL_OTIDS_UB.csv

# Cliente
export LOCAL_EPOCHS=2
export BATCH_SIZE=32
```

### Métodos de Agregación

| Método | Descripción | Uso Recomendado |
|--------|-------------|-----------------|
| `fedavg` | FedAvg clásico | Baseline, clientes homogéneos |
| `fedavgm` | FedAvg + Momentum | Convergencia más rápida |
| `fednova` | Normalización por pasos | Clientes heterogéneos (datos/hardware) |

### Distribución de Datos

| Tipo | Alpha | Descripción |
|------|-------|-------------|
| `iid` | - | Distribución uniforme (todas las clases en todos los clientes) |
| `noniid` | 0.1-0.5 | Muy heterogéneo (pocas clases por cliente) |
| `noniid` | 0.5-1.0 | Moderadamente heterogéneo |
| `noniid` | >1.0 | Casi IID |

---

## 🐛 Troubleshooting

### Problema: Clientes no se conectan

```bash
# Verificar que el servidor esté corriendo
curl http://localhost:5000/health

# Verificar firewall (en el servidor)
sudo ufw allow 5000/tcp
```

### Problema: "No se pudieron obtener los pesos"

- Verificar que el servidor haya cargado los datos correctamente
- Revisar logs en `logs/server.log`

### Problema: Desincronización de rondas

- **Ya corregido en esta versión**
- Los clientes obtienen el número de ronda del servidor

---

## 📚 Documentación Adicional

- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Guía detallada de deployment real
- **[PAPER_GUIDE.md](PAPER_GUIDE.md)**: Guía para escribir el paper IEEE
- **[API.md](API.md)**: Documentación de la API REST del servidor

---

## 🤝 Contribuir

Este proyecto es para fines académicos. Si encuentras bugs o mejoras:

1. Crea un issue
2. Fork el proyecto
3. Crea un Pull Request

---

## 📝 Licencia

MIT License - Ver LICENSE para detalles

---

## 🎓 Créditos

Proyecto desarrollado como demostración académica de Federated Learning.

**Algoritmos implementados:**
- FedAvg: McMahan et al., 2017
- FedAvgM: Hsu et al., 2019
- FedNova: Wang et al., 2020

**Dataset:**
- CAN-HCRL-OTIDS: Dataset de detección de intrusiones en redes vehiculares
