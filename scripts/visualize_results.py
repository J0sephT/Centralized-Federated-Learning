"""
Script para visualizar resultados del entrenamiento federado
"""
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(metrics_file: str, output_dir: str = 'results'):
    """
    Genera gráficas de loss y accuracy vs rondas.
    
    Args:
        metrics_file: Ruta al archivo metrics.json
        output_dir: Directorio para guardar gráficas
    """
    # Cargar métricas
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    if not metrics:
        print("❌ No hay métricas para visualizar")
        return
    
    # Convertir a DataFrame
    df = pd.DataFrame(metrics)
    
    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Accuracy
    ax1.plot(df['round'], df['accuracy'], marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Ronda', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Global vs Ronda', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Loss
    ax2.plot(df['round'], df['loss'], marker='s', color='red', linewidth=2, markersize=8)
    ax2.set_xlabel('Ronda', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Global vs Ronda', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    output_path = f"{output_dir}/metrics_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfica guardada en: {output_path}")
    
    # Mostrar
    plt.show()
    
    # Imprimir resumen
    print("\n📊 RESUMEN DE MÉTRICAS:")
    print("="*60)
    print(f"Rondas totales: {len(df)}")
    print(f"Accuracy inicial: {df['accuracy'].iloc[0]:.4f}")
    print(f"Accuracy final: {df['accuracy'].iloc[-1]:.4f}")
    print(f"Mejora: {(df['accuracy'].iloc[-1] - df['accuracy'].iloc[0]):.4f}")
    print(f"Loss inicial: {df['loss'].iloc[0]:.4f}")
    print(f"Loss final: {df['loss'].iloc[-1]:.4f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Visualizar resultados de FL')
    parser.add_argument('--metrics_file', type=str, default='results/metrics.json',
                       help='Ruta al archivo de métricas')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directorio para guardar gráficas')
    
    args = parser.parse_args()
    
    plot_metrics(args.metrics_file, args.output_dir)


if __name__ == '__main__':
    main()
