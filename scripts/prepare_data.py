"""
Script para preparar y distribuir datos a los clientes
"""
import os
import sys
import argparse

# Agregar directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_utils import (
    load_and_preprocess_data,
    distribute_data_iid,
    distribute_data_noniid,
    save_client_data
)


def main():
    parser = argparse.ArgumentParser(description='Preparar y distribuir datos para FL')
    parser.add_argument('--csv_path', type=str, default='data/CAN_HCRL_OTIDS_UB.csv',
                       help='Ruta al dataset CSV')
    parser.add_argument('--num_clients', type=int, default=3,
                       help='Número de clientes')
    parser.add_argument('--distribution', type=str, default='noniid',
                       choices=['iid', 'noniid'],
                       help='Tipo de distribución de datos')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Parámetro alpha para Dirichlet (solo non-IID)')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Directorio de salida')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proporción de datos para test')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Número de muestras a usar (None = todas)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("📦 PREPARANDO DATOS PARA FEDERATED LEARNING")
    print("="*60)
    print(f"Dataset: {args.csv_path}")
    print(f"Clientes: {args.num_clients}")
    print(f"Distribución: {args.distribution}")
    if args.distribution == 'noniid':
        print(f"Alpha: {args.alpha}")
    print("="*60)
    
    # 1. Cargar y preprocesar
    df_train, df_test = load_and_preprocess_data(
        args.csv_path,
        test_size=args.test_size,
        sample_size=args.sample_size
    )
    
    # 2. Distribuir datos
    if args.distribution == 'iid':
        clients_data = distribute_data_iid(df_train, args.num_clients)
    else:
        clients_data = distribute_data_noniid(
            df_train, 
            args.num_clients,
            alpha=args.alpha
        )
    
    # 3. Guardar datos de clientes
    print(f"\n💾 Guardando datos de clientes en {args.output_dir}/...")
    save_client_data(clients_data, args.output_dir)
    
    print("\n✅ Datos preparados exitosamente!")
    print(f"📁 Archivos generados en: {args.output_dir}/")
    for i in range(args.num_clients):
        print(f"   - client_{i}_data.csv")


if __name__ == '__main__':
    main()
