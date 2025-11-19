#!/bin/bash

echo "===================================="
echo "🚀 Federated Learning - Quick Start"
echo "===================================="

# Colores
GREEN='\033[0.32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Modo de ejecución
MODE="${1:-local}"  # Opciones: local, docker

if [ "$MODE" = "docker" ]; then
    echo -e "\n${YELLOW}Modo: Docker${NC}"
    
    # Verificar Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker no está instalado${NC}"
        echo "   Instálalo desde: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose no está instalado${NC}"
        echo "   Instálalo desde: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Docker disponible${NC}"
    
    # Verificar dataset
    if [ ! -f "data/CAN_HCRL_OTIDS_UB.csv" ]; then
        echo -e "${RED}❌ Dataset no encontrado en data/CAN_HCRL_OTIDS_UB.csv${NC}"
        exit 1
    fi
    
    # Preparar datos si no existen
    if [ ! -f "data/client_0_data.csv" ]; then
        echo -e "\n${YELLOW}Preparando datos de clientes...${NC}"
        python scripts/prepare_data.py \
            --csv_path data/CAN_HCRL_OTIDS_UB.csv \
            --num_clients 3 \
            --distribution noniid \
            --alpha 0.5
    fi
    
    echo -e "\n${YELLOW}Iniciando Docker Compose...${NC}"
    docker-compose up --build
    
else
    echo -e "\n${YELLOW}Modo: Local (con entorno virtual)${NC}"
    
    # 1. Verificar Python
    echo -e "\n${YELLOW}1. Verificando Python...${NC}"
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 no está instalado${NC}"
        echo "   Instálalo desde: https://www.python.org/downloads/"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo -e "${GREEN}✅ Python $PYTHON_VERSION disponible${NC}"
    
    # 2. Crear entorno virtual
    echo -e "\n${YELLOW}2. Configurando entorno virtual...${NC}"
    if [ ! -d ".venv" ]; then
        echo "   Creando .venv..."
        python3 -m venv .venv
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ Error creando entorno virtual${NC}"
            echo "   Intenta: sudo apt install python3-venv"
            exit 1
        fi
        echo -e "${GREEN}✅ Entorno virtual creado${NC}"
    else
        echo -e "${GREEN}✅ Entorno virtual ya existe${NC}"
    fi
    
    # 3. Activar entorno virtual
    echo -e "\n${YELLOW}3. Activando entorno virtual...${NC}"
    source .venv/bin/activate
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Error activando entorno virtual${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Entorno virtual activado${NC}"
    
    # 4. Instalar dependencias
    echo -e "\n${YELLOW}4. Instalando dependencias...${NC}"
    pip install --upgrade pip > /dev/null 2>&1
    
    # Detectar versión de Python
    PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
    echo "   Detectado: Python 3.$PYTHON_MINOR"
    
    # Instalar requirements apropiado
    echo "   Instalando paquetes..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Error instalando dependencias${NC}"
        echo "   Intenta: pip install -r requirements-legacy.txt"
        echo "   Ver INSTALL.md para más detalles"
        exit 1
    fi
    echo -e "${GREEN}✅ Dependencias instaladas${NC}"
    
    # 5. Verificar dataset
    echo -e "\n${YELLOW}5. Verificando dataset...${NC}"
    if [ ! -f "data/CAN_HCRL_OTIDS_UB.csv" ]; then
        echo -e "${RED}❌ Dataset no encontrado en data/CAN_HCRL_OTIDS_UB.csv${NC}"
        echo "   Descarga el dataset y colócalo en data/"
        exit 1
    fi
    echo -e "${GREEN}✅ Dataset encontrado${NC}"
    
    # 6. Preparar datos de clientes
    echo -e "\n${YELLOW}6. Preparando datos de clientes...${NC}"
    if [ ! -f "data/client_0_data.csv" ]; then
        python scripts/prepare_data.py \
            --csv_path data/CAN_HCRL_OTIDS_UB.csv \
            --num_clients 3 \
            --distribution noniid \
            --alpha 0.5
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ Error preparando datos${NC}"
            exit 1
        fi
    else
        echo "   Datos ya preparados"
    fi
    echo -e "${GREEN}✅ Datos preparados${NC}"
    
    # 7. Crear directorios
    echo -e "\n${YELLOW}7. Creando directorios...${NC}"
    mkdir -p logs results
    echo -e "${GREEN}✅ Directorios creados${NC}"
    
    # 8. Instrucciones finales
    echo -e "\n${GREEN}═══════════════════════════════════════${NC}"
    echo -e "${GREEN}🎉 Setup completado exitosamente!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}Ahora ejecuta en terminales separadas:${NC}"
    echo ""
    echo "Terminal 1 - Servidor:"
    echo "  source .venv/bin/activate"
    echo "  python server/server.py"
    echo ""
    echo "Terminal 2-4 - Clientes:"
    echo "  source .venv/bin/activate"
    echo "  python client/client.py --client_id 0 --server_url http://localhost:5000 --data_dir data/"
    echo "  python client/client.py --client_id 1 --server_url http://localhost:5000 --data_dir data/"
    echo "  python client/client.py --client_id 2 --server_url http://localhost:5000 --data_dir data/"
    echo ""
    echo "Terminal 5 - Coordinator:"
    echo "  source .venv/bin/activate"
    echo "  python scripts/coordinator.py --num_clients 3 --num_rounds 10"
    echo ""
    echo -e "${YELLOW}O usa el modo Docker:${NC}"
    echo "  ./quickstart.sh docker"
    echo ""
fi
