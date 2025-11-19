# 📦 Guía de Instalación - Dependencias

## ⚠️ Problema Común: Versión de TensorFlow

Si ves este error:
```
ERROR: Could not find a version that satisfies the requirement tensorflow==2.13.0
```

**Causa:** Tu versión de Python es incompatible con TensorFlow 2.13.0

---

## ✅ Solución Automática

### Opción 1: Usar versiones flexibles (RECOMENDADO)

```bash
# Instalar con versiones compatibles
pip install -r requirements.txt
```

Este archivo usa versiones **flexibles** (`>=`) que funcionan con:
- ✅ Python 3.8
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11
- ✅ Python 3.12+

---

### Opción 2: Usar versiones legacy (Python 3.8-3.11)

Si prefieres versiones exactas:

```bash
# Solo para Python 3.8 - 3.11
pip install -r requirements-legacy.txt
```

---

## 🔍 Verificar tu versión de Python

```bash
python3 --version
# o
python --version
```

**Según tu versión:**

| Versión Python | Archivo a usar |
|----------------|----------------|
| 3.8 - 3.11 | `requirements.txt` o `requirements-legacy.txt` |
| 3.12+ | `requirements.txt` (actualizado) |

---

## 📋 Contenido de `requirements.txt` (actualizado)

```
# Compatible con Python 3.8 - 3.12+
tensorflow>=2.16.0
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.3.0
flask>=3.0.0
requests>=2.31.0
matplotlib>=3.8.0
seaborn>=0.13.0
tqdm>=4.66.1
colorama>=0.4.6
joblib>=1.3.2
python-json-logger>=2.0.7
```

---

## 🚀 Instalación Paso a Paso

### 1. Crear entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows
```

### 2. Actualizar pip

```bash
pip install --upgrade pip
```

### 3. Instalar dependencias

**Para Python 3.12+:**
```bash
pip install -r requirements.txt
```

**Para Python 3.8-3.11:**
```bash
# Opción A: Versiones flexibles (recomendado)
pip install -r requirements.txt

# Opción B: Versiones exactas
pip install -r requirements-legacy.txt
```

### 4. Verificar instalación

```bash
python -c "
import tensorflow as tf
import flask
import numpy as np
import pandas as pd
print('✅ TensorFlow:', tf.__version__)
print('✅ Flask:', flask.__version__)
print('✅ NumPy:', np.__version__)
print('✅ Pandas:', pd.__version__)
"
```

---

## 🐛 Troubleshooting

### ❌ Error: "No matching distribution found for tensorflow"

**Solución 1: Actualizar pip**
```bash
pip install --upgrade pip setuptools wheel
```

**Solución 2: Usar versión específica**
```bash
# Para Python 3.12
pip install tensorflow>=2.16.0

# Para Python 3.11
pip install tensorflow>=2.15.0

# Para Python 3.10
pip install tensorflow>=2.13.0
```

**Solución 3: Instalar desde requirements actualizado**
```bash
pip install -r requirements.txt
```

---

### ❌ Error: "Could not build wheels for numpy"

**Solución:**
```bash
# Linux
sudo apt-get install python3-dev build-essential

# Mac
xcode-select --install

# Windows
# Instalar Visual Studio Build Tools
```

---

### ❌ Error en Raspberry Pi / ARM

**Solución: Usar versión CPU**
```bash
# Desinstalar versión estándar
pip uninstall tensorflow

# Instalar versión CPU/ARM
pip install tensorflow-cpu>=2.15.0

# O versión lite
pip install tflite-runtime
```

**Archivo especial para Raspberry Pi:**
```bash
cat > requirements-raspi.txt << 'END'
tensorflow-cpu>=2.15.0
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.3.0
flask>=3.0.0
requests>=2.31.0
matplotlib>=3.8.0
END

pip install -r requirements-raspi.txt
```

---

### ❌ Error en Mac M1/M2

**Solución: Usar versiones optimizadas para Apple Silicon**
```bash
# Instalar versión para Apple Silicon
pip install tensorflow-macos>=2.15.0
pip install tensorflow-metal

# O usar requirements estándar (debería funcionar)
pip install -r requirements.txt
```

---

## 🎯 Instalación por Sistema Operativo

### Ubuntu/Debian

```bash
# Instalar dependencias del sistema
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip python3-venv build-essential

# Crear entorno virtual e instalar
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS

```bash
# Con Homebrew
brew install python3

# Crear entorno virtual e instalar
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```powershell
# Crear entorno virtual e instalar
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔄 Actualizar Dependencias

```bash
# Actualizar pip
pip install --upgrade pip

# Actualizar todas las dependencias
pip install --upgrade -r requirements.txt

# O actualizar paquetes específicos
pip install --upgrade tensorflow numpy pandas
```

---

## 📊 Comparación de Versiones

| Paquete | Python 3.8-3.11 (legacy) | Python 3.12+ (actualizado) |
|---------|--------------------------|----------------------------|
| tensorflow | 2.13.0 | 2.16.0+ |
| numpy | 1.24.3 | 1.26.0+ |
| pandas | 2.0.3 | 2.1.0+ |
| flask | 2.3.2 | 3.0.0+ |
| matplotlib | 3.7.2 | 3.8.0+ |

---

## ✅ Verificación Final

Ejecuta este script para verificar que todo funciona:

```bash
python << 'PYTHON'
import sys
print(f"Python: {sys.version}")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow: {e}")

try:
    import flask
    print(f"✅ Flask: {flask.__version__}")
except ImportError as e:
    print(f"❌ Flask: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas: {e}")

print("\n🎉 Si ves ✅ en todos, estás listo!")
PYTHON
```

---

## 💡 Recomendaciones

1. **Usa siempre entorno virtual** (`.venv`)
2. **requirements.txt usa `>=`** para compatibilidad futura
3. **requirements-legacy.txt usa `==`** para reproducibilidad exacta
4. **Actualiza pip primero**: `pip install --upgrade pip`
5. **Para producción**, usa `pip freeze > requirements-frozen.txt` después de probar

---

## 📝 Resumen Rápido

```bash
# Setup completo en 4 comandos:
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verificar
python -c "import tensorflow; print(tensorflow.__version__)"
```

---

🎉 **¡Listo!** Ahora puedes continuar con el resto del setup en [SETUP.md](SETUP.md)
