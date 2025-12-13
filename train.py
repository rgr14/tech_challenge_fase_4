"""
Script principal de treinamento do modelo LSTM
"""
import argparse
import json
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from config import (
    DEFAULT_TICKER, DEFAULT_START_DATE, EPOCHS, BATCH_SIZE,
    MODELS_DIR, DATA_DIR, FEATURES
)
from data_collector import StockDataCollector
from preprocessor import StockDataPreprocessor
from model import StockLSTMModel, calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    ticker: str = DEFAULT_TICKER,
    start_date: str = DEFAULT_START_DATE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    save_plots: bool = True
) -> dict:
    """
    Pipeline completo de treinamento do modelo.
    
    Args:
        ticker: Símbolo da ação
        start_date: Data inicial dos dados
        epochs: Número de épocas
        batch_size: Tamanho do batch
        save_plots: Se deve salvar gráficos
    
    Returns:
        Dicionário com métricas e informações do treinamento
    """
    logger.info(f"=" * 60)
    logger.info(f"Iniciando treinamento para {ticker}")
    logger.info(f"=" * 60)
    
    # 1. Coleta de dados
    logger.info("\n[1/6] Coletando dados...")
    collector = StockDataCollector(ticker)
    df = collector.fetch_historical_data(start_date=start_date)
    collector.save_data()
    stock_info = collector.get_stock_info()
    
    logger.info(f"Coletados {len(df)} registros de {df.index.min()} até {df.index.max()}")
    
    # 2. Pré-processamento
    logger.info("\n[2/6] Preparando features com validação de dados...")
    preprocessor = StockDataPreprocessor(
        features=FEATURES,
        validate_data=True,
        auto_clean=True,
        scaler_type="minmax"
    )
    
    # prepare_features agora inclui validação automática
    df_features = preprocessor.prepare_features(df, ticker=ticker)
    
    logger.info(f"Features criadas: {len(df_features.columns)} colunas")
    logger.info(f"Dados após feature engineering: {len(df_features)} registros")
    
    # Mostrar estatísticas de limpeza
    if preprocessor.cleaning_stats:
        stats = preprocessor.cleaning_stats
        logger.info(f"\n📊 Qualidade dos dados:")
        logger.info(f"   Registros removidos: {stats['removed_records']} ({stats['removal_percentage']:.2f}%)")
        logger.info(f"   Avisos: {stats['warnings']}, Erros: {stats['errors']}")
    
    # 3. Normalização e criação de sequências
    logger.info("\n[3/6] Normalizando dados e criando sequências...")
    preprocessor.fit_scalers(df_features)
    features_scaled, target_scaled = preprocessor.transform(df_features)
    
    X, y = preprocessor.create_sequences(features_scaled, target_scaled)
    logger.info(f"Sequências criadas: X={X.shape}, y={y.shape}")
    
    # 4. Split dos dados
    logger.info("\n[4/6] Dividindo dados em treino/validação/teste...")
    
    # 70% treino, 15% validação, 15% teste
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # 5. Treinamento
    logger.info("\n[5/6] Treinando modelo...")
    model = StockLSTMModel(
        sequence_length=preprocessor.sequence_length,
        n_features=X_train.shape[2]
    )
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        ticker=ticker,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # 6. Avaliação
    logger.info("\n[6/6] Avaliando modelo...")
    
    # Previsões
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Reverter normalização
    y_train_real = preprocessor.inverse_transform_predictions(y_train)
    y_train_pred_real = preprocessor.inverse_transform_predictions(y_train_pred)
    y_val_real = preprocessor.inverse_transform_predictions(y_val)
    y_val_pred_real = preprocessor.inverse_transform_predictions(y_val_pred)
    y_test_real = preprocessor.inverse_transform_predictions(y_test)
    y_test_pred_real = preprocessor.inverse_transform_predictions(y_test_pred)
    
    # Calcular métricas
    train_metrics = calculate_metrics(y_train_real, y_train_pred_real)
    val_metrics = calculate_metrics(y_val_real, y_val_pred_real)
    test_metrics = calculate_metrics(y_test_real, y_test_pred_real)
    
    logger.info("\n" + "=" * 60)
    logger.info("MÉTRICAS DE AVALIAÇÃO")
    logger.info("=" * 60)
    logger.info(f"\nTreino:")
    logger.info(f"  MAE: ${train_metrics['mae']:.2f}")
    logger.info(f"  RMSE: ${train_metrics['rmse']:.2f}")
    logger.info(f"  MAPE: {train_metrics['mape']:.2f}%")
    logger.info(f"  R²: {train_metrics['r2']:.4f}")
    logger.info(f"  Direction Accuracy: {train_metrics['direction_accuracy']:.2f}%")
    
    logger.info(f"\nValidação:")
    logger.info(f"  MAE: ${val_metrics['mae']:.2f}")
    logger.info(f"  RMSE: ${val_metrics['rmse']:.2f}")
    logger.info(f"  MAPE: {val_metrics['mape']:.2f}%")
    logger.info(f"  R²: {val_metrics['r2']:.4f}")
    logger.info(f"  Direction Accuracy: {val_metrics['direction_accuracy']:.2f}%")
    
    logger.info(f"\nTeste:")
    logger.info(f"  MAE: ${test_metrics['mae']:.2f}")
    logger.info(f"  RMSE: ${test_metrics['rmse']:.2f}")
    logger.info(f"  MAPE: {test_metrics['mape']:.2f}%")
    logger.info(f"  R²: {test_metrics['r2']:.4f}")
    logger.info(f"  Direction Accuracy: {test_metrics['direction_accuracy']:.2f}%")
    
    # Salvar modelo e scalers
    model.save_model(ticker)
    preprocessor.save_scalers(ticker)
    
    # Gerar gráficos
    if save_plots:
        _save_training_plots(
            history, 
            y_test_real, 
            y_test_pred_real, 
            ticker
        )
    
    # Resultados
    results = {
        "ticker": ticker,
        "stock_info": stock_info,
        "training_date": datetime.now().isoformat(),
        "data_range": {
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "total_records": len(df)
        },
        "data_quality": {
            "original_records": preprocessor.cleaning_stats.get("original_records", len(df)),
            "final_records": preprocessor.cleaning_stats.get("final_records", len(df_features)),
            "records_removed": preprocessor.cleaning_stats.get("removed_records", 0),
            "removal_percentage": preprocessor.cleaning_stats.get("removal_percentage", 0),
            "is_valid": preprocessor.cleaning_stats.get("is_valid", True),
            "warnings": preprocessor.cleaning_stats.get("warnings", 0),
            "errors": preprocessor.cleaning_stats.get("errors", 0)
        },
        "model_config": {
            "sequence_length": preprocessor.sequence_length,
            "n_features": X_train.shape[2],
            "features_used": preprocessor.fitted_features,
            "epochs_trained": len(history['loss']),
            "batch_size": batch_size,
            "scaler_type": preprocessor.scaler_type
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics
        },
        "final_predictions": {
            "last_actual": float(y_test_real[-1][0]),
            "last_predicted": float(y_test_pred_real[-1][0])
        }
    }
    
    # Salvar resultados
    results_path = MODELS_DIR / f"{ticker}_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResultados salvos em {results_path}")
    logger.info(f"Modelo salvo em {MODELS_DIR / f'{ticker}_lstm_model.keras'}")
    
    return results


def _save_training_plots(
    history: dict,
    y_test_real: np.ndarray,
    y_test_pred: np.ndarray,
    ticker: str
) -> None:
    """Gera e salva gráficos de treinamento."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss durante treinamento
    axes[0, 0].plot(history['loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[0, 0].set_title('Loss Durante Treinamento')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. MAE durante treinamento
    axes[0, 1].plot(history['mae'], label='Train MAE', color='blue')
    axes[0, 1].plot(history['val_mae'], label='Val MAE', color='orange')
    axes[0, 1].set_title('MAE Durante Treinamento')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Previsões vs Real (Teste)
    axes[1, 0].plot(y_test_real, label='Real', color='blue', alpha=0.7)
    axes[1, 0].plot(y_test_pred, label='Previsto', color='red', alpha=0.7)
    axes[1, 0].set_title(f'Previsões vs Valores Reais ({ticker})')
    axes[1, 0].set_xlabel('Amostras')
    axes[1, 0].set_ylabel('Preço ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Scatter plot
    axes[1, 1].scatter(y_test_real, y_test_pred, alpha=0.5, color='blue')
    min_val = min(y_test_real.min(), y_test_pred.min())
    max_val = max(y_test_real.max(), y_test_pred.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfeito')
    axes[1, 1].set_title('Correlação Real vs Previsto')
    axes[1, 1].set_xlabel('Valor Real ($)')
    axes[1, 1].set_ylabel('Valor Previsto ($)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = MODELS_DIR / f"{ticker}_training_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráficos salvos em {plot_path}")


def main():
    """Entry point do script de treinamento."""
    parser = argparse.ArgumentParser(
        description='Treinar modelo LSTM para previsão de ações'
    )
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        default=DEFAULT_TICKER,
        help=f'Símbolo da ação (default: {DEFAULT_TICKER})'
    )
    parser.add_argument(
        '--start-date', '-s',
        type=str,
        default=DEFAULT_START_DATE,
        help=f'Data inicial dos dados (default: {DEFAULT_START_DATE})'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=EPOCHS,
        help=f'Número de épocas (default: {EPOCHS})'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=BATCH_SIZE,
        help=f'Tamanho do batch (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Não salvar gráficos'
    )
    
    args = parser.parse_args()
    
    results = train_model(
        ticker=args.ticker,
        start_date=args.start_date,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_plots=not args.no_plots
    )
    
    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print(f"\nAção: {results['ticker']} - {results['stock_info']['name']}")
    print(f"MAPE no teste: {results['metrics']['test']['mape']:.2f}%")
    print(f"Direction Accuracy: {results['metrics']['test']['direction_accuracy']:.2f}%")


if __name__ == "__main__":
    main()