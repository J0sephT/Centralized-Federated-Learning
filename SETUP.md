# 🔧 Setup Completo - Guía de Instalación

## ✅ Pasos Recomendados (con Virtual Environment)

### 1️⃣ **Descomprimir el proyecto**

```bash
unzip federated-learning-project.zip
cd federated-learning-project/
```

---

### 2️⃣ **Crear entorno virtual** (RECOMENDADO ✅)

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Verificar activación:**
- Tu terminal debe mostrar: `(.venv)` al inicio de la línea

---

### 3️⃣ **Instalar dependencias**

```bash
# Con el entorno virtual activado:
pip install --upgrade pip
pip install -r requirements.txt
```

**Verificar instalación:**
```bash
pip list
```

Deberías ver:
- tensorflow==2.13.0
- flask==2.3.2
- numpy, pandas, scikit-learn, etc.

---

### 4️⃣ **Preparar datos**

```bash
# Asegúrate de tener el dataset en data/
python scripts/prepare_data.py \
    --csv_path data/CAN_HCRL_OTIDS_UB.csv \
    --num_clients 3 \
    --distribution noniid \
    --alpha 0.5
```

**Salida esperada:**
```
📦 PREPARANDO DATOS PARA FEDERATED LEARNING
Dataset: data/CAN_HCRL_OTIDS_UB.csv
Clientes: 3
Distribución: noniid
...
✅ Datos preparados exitosamente!
```

---

### 5️⃣ **Ejecutar el sistema**

**Opción A: Con el script coordinador (RECOMENDADO)**

```bash
# Terminal 1: Servidor
python server/server.py

# Terminal 2-4: Clientes (en nuevas terminales con .venv activado)
python client/client.py --client_id 0 --server_url http://localhost:5000 --data_dir data/
python client/client.py --client_id 1 --server_url http://localhost:5000 --data_dir data/
python client/client.py --client_id 2 --server_url http://localhost:5000 --data_dir data/

# Terminal 5: Coordinator
python scripts/coordinator.py --server_url http://localhost:5000 --num_clients 3 --num_rounds 10
```

**Opción B: Con Docker (sin necesidad de .venv)**

```bash
docker-compose up --build
```

---

### 6️⃣ **Visualizar resultados**

```bash
python scripts/visualize_results.py --metrics_file results/metrics.json
```

---

## 🔄 **Activar/Desactivar el entorno virtual**

### Activar (cada vez que abras una nueva terminal)

**Linux/Mac:**
```bash
cd federated-learning-project/
source .venv/bin/activate
```

**Windows:**
```bash
cd federated-learning-project
.venv\Scripts\activate
```

### Desactivar

```bash
deactivate
```

---

## 🐛 **Troubleshooting - Setup**

### ❌ Error: "python3: command not found"

**Solución:**
```bash
# Linux/Mac
sudo apt install python3 python3-pip python3-venv

# Mac (con Homebrew)
brew install python3

# Windows
# Descargar desde: https://www.python.org/downloads/
```

---

### ❌ Error: "No module named 'venv'"

**Solución:**
```bash
# Ubuntu/Debian
sudo apt install python3-venv

# Fedora/RHEL
sudo dnf install python3-virtualenv
```

---

### ❌ Error: "pip: command not found"

**Solución:**
```bash
# Linux
python3 -m ensurepip --upgrade

# Windows
python -m ensurepip --upgrade
```

---

### ❌ Error: "tensorflow installation failed"

**Solución para Raspberry Pi:**
```bash
# Usar versión CPU
pip install tensorflow-cpu==2.13.0

# O versión lite
pip install tflite-runtime
```

**Solución para Mac M1/M2:**
```bash
# Usar versión para Apple Silicon
pip install tensorflow-macos==2.13.0
pip install tensorflow-metal
```

---

### ❌ Error: "No such file or directory: data/CAN_HCRL_OTIDS_UB.csv"

**Solución:**
1. Descargar el dataset
2. Colocarlo en: `federated-learning-project/data/CAN_HCRL_OTIDS_UB.csv`

---

## 📋 **Checklist de Setup**

Usa esta lista para verificar que todo esté listo:

- [ ] Proyecto descomprimido
- [ ] Entorno virtual creado (`.venv/` existe)
- [ ] Entorno virtual activado (`(.venv)` visible en terminal)
- [ ] Dependencias instaladas (`pip list` muestra TensorFlow, Flask, etc.)
- [ ] Dataset colocado en `data/CAN_HCRL_OTIDS_UB.csv`
- [ ] Datos de clientes generados (`data/client_*_data.csv` existen)
- [ ] Directorios `logs/` y `results/` creados

---

## 🎯 **Diferentes Configuraciones**

### Para Desarrollo/Testing (Local)

```bash
# Setup con entorno virtual
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Pros:**
- ✅ Control total
- ✅ Fácil debugging
- ✅ Rápido de iniciar

**Cons:**
- ❌ Dependencias en tu sistema
- ❌ Múltiples terminales

---

### Para Simulación (Docker)

```bash
# No necesita entorno virtual
docker-compose up --build
```

**Pros:**
- ✅ Aislamiento completo
- ✅ Reproducible
- ✅ Un solo comando

**Cons:**
- ❌ Más lento para iterar
- ❌ Overhead de containers

---

### Para Deployment Real (Dispositivos)

```bash
# En cada dispositivo:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Pros:**
- ✅ FL real entre dispositivos
- ✅ Ideal para demo

**Cons:**
- ❌ Setup en múltiples dispositivos
- ❌ Requiere red configurada

---

## 💡 **Tips Adicionales**

### 1. Actualizar dependencias

```bash
pip install --upgrade -r requirements.txt
```

### 2. Exportar dependencias actuales

```bash
pip freeze > requirements-frozen.txt
```

### 3. Limpiar entorno virtual

```bash
# Desactivar primero
deactivate

# Eliminar y recrear
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Versión mínima de Python

```bash
python3 --version
# Debe ser Python 3.8 o superior
```

### 5. Guardar el .venv en .gitignore

Ya está incluido en `.gitignore`, pero verifica:
```bash
cat .gitignore | grep venv
# Debe mostrar: .venv
```

---

## 🚀 **Quick Start Completo**

Para tu conveniencia, aquí está el flujo completo:

```bash
# 1. Descomprimir
unzip federated-learning-project.zip
cd federated-learning-project/

# 2. Setup entorno
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# O: .venv\Scripts\activate  # Windows

# 3. Instalar
pip install --upgrade pip
pip install -r requirements.txt

# 4. Preparar datos
python scripts/prepare_data.py \
    --csv_path data/CAN_HCRL_OTIDS_UB.csv \
    --num_clients 3 \
    --distribution noniid \
    --alpha 0.5

# 5. Ejecutar (en terminales separadas)
# Terminal 1:
python server/server.py

# Terminal 2-4:
python client/client.py --client_id 0 --server_url http://localhost:5000 --data_dir data/
python client/client.py --client_id 1 --server_url http://localhost:5000 --data_dir data/
python client/client.py --client_id 2 --server_url http://localhost:5000 --data_dir data/

# Terminal 5:
python scripts/coordinator.py --num_clients 3 --num_rounds 10

# 6. Visualizar
python scripts/visualize_results.py
```

---

## 📝 **Para Recordar**

1. **SIEMPRE** activa el entorno virtual antes de trabajar:
   ```bash
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

2. **NO COMMITEAR** el directorio `.venv/` a Git (ya está en `.gitignore`)

3. Si trabajas en **múltiples dispositivos**, cada uno necesita su propio `.venv`

4. Para **compartir** el proyecto, solo comparte el código, NO el `.venv/`

---

## ✅ **Verificación Final**

Ejecuta este comando para verificar que todo esté listo:

```bash
python -c "
import tensorflow as tf
import flask
import numpy as np
import pandas as pd
import sklearn
print('✅ Todas las dependencias instaladas correctamente!')
print(f'TensorFlow: {tf.__version__}')
print(f'Python: {tf.test.is_built_with_cuda()}')
"
```

**Salida esperada:**
```
✅ Todas las dependencias instaladas correctamente!
TensorFlow: 2.13.0
Python: False
```

---

🎉 **¡Listo para empezar!**

Si tienes algún problema, revisa la sección de Troubleshooting o consulta:
- README.md para documentación general
- DEPLOYMENT.md para deployment en dispositivos reales
- EXAMPLE_USAGE.md para ejemplos específicos
