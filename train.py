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
    MODELS_DIR, DATA_DIR, FEATURES, TRAIN_TEST_SPLIT
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
    
    logger.info(f"=" * 60)
    logger.info(f"Iniciando treinamento para {ticker}")
    logger.info(f"=" * 60)
    
    # 1. Coleta de dados
    logger.info("\n[1/5] Coletando dados...")
    collector = StockDataCollector(ticker)
    df = collector.fetch_historical_data(start_date=start_date)
    collector.save_data()
    stock_info = collector.get_stock_info()
    
    logger.info(f"Coletados {len(df)} registros de {df.index.min()} até {df.index.max()}")
    
    # 2. Pré-processamento
    logger.info("\n[2/5] Preparando features...")
    preprocessor = StockDataPreprocessor(
        features=FEATURES,
        validate_data=True,
        auto_clean=True,
        scaler_type="minmax"
    )
    
    # Feature Engineering (Simplificado se features=['Close'])
    df_features = preprocessor.prepare_features(df, ticker=ticker)
    
    logger.info(f"Features usadas: {df_features.columns.tolist()}")
    
    # 3. Normalização e Sequenciamento
    logger.info("\n[3/5] Normalizando e criando sequências...")
    
    # --- CORREÇÃO IMPORTANTE 1: Fit no dataset COMPLETO ---
    # Isso garante que o scaler aprenda o preço máximo histórico ($140+)
    preprocessor.fit_scalers(df_features)
    
    features_scaled, target_scaled = preprocessor.transform(df_features)
    X, y = preprocessor.create_sequences(features_scaled, target_scaled)
    logger.info(f"Sequências criadas: X={X.shape}, y={y.shape}")
    
    # 4. Split dos dados (CORREÇÃO DE LÓGICA)
    logger.info("\n[4/5] Dividindo dados (Respeitando config.py)...")
    
    # --- CORREÇÃO IMPORTANTE 2: Respeitar TRAIN_TEST_SPLIT do config ---
    # Usamos divisão simples Treino/Teste para maximizar dados de treino
    split_idx = int(len(X) * TRAIN_TEST_SPLIT)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    logger.info(f"Train: {X_train.shape[0]} amostras ({TRAIN_TEST_SPLIT*100}%)")
    logger.info(f"Test:  {X_test.shape[0]} amostras ({(1-TRAIN_TEST_SPLIT)*100:.0f}%)")
    
    # 5. Treinamento
    logger.info("\n[5/5] Treinando modelo...")
    model = StockLSTMModel(
        sequence_length=preprocessor.sequence_length,
        n_features=X_train.shape[2]
    )
    
    # Usamos X_test como validação para monitorar overfitting em tempo real
    history = model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        ticker=ticker,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # 6. Avaliação Final
    logger.info("\n" + "=" * 60)
    logger.info("AVALIAÇÃO FINAL")
    logger.info("=" * 60)
    
    # Previsões
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Reverter escala (Voltar para Dólares)
    y_train_real = preprocessor.inverse_transform_predictions(y_train)
    y_train_pred_real = preprocessor.inverse_transform_predictions(y_train_pred)
    y_test_real = preprocessor.inverse_transform_predictions(y_test)
    y_test_pred_real = preprocessor.inverse_transform_predictions(y_test_pred)
    
    # Calcular Métricas
    train_metrics = calculate_metrics(y_train_real, y_train_pred_real)
    test_metrics = calculate_metrics(y_test_real, y_test_pred_real)
    
    # Exibir Console
    _print_metrics("Treino", train_metrics)
    _print_metrics("Teste", test_metrics)
    
    # Salvar artefatos
    model.save_model(ticker)
    preprocessor.save_scalers(ticker)
    
    if save_plots:
        _save_training_plots(history, y_test_real, y_test_pred_real, ticker)
    
    # Retornar dict de resultados
    return {
        "ticker": ticker,
        "metrics": {"train": train_metrics, "test": test_metrics}
    }

def _print_metrics(label: str, metrics: dict):
    logger.info(f"\n{label}:")
    logger.info(f"  MAE: ${metrics['mae']:.2f}")
    logger.info(f"  RMSE: ${metrics['rmse']:.2f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    logger.info(f"  Direction Accuracy: {metrics['direction_accuracy']:.2f}%")

def _save_training_plots(history, y_real, y_pred, ticker):
    """Gera gráficos simplificados e diretos."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss
    axes[0].plot(history['loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Test Loss')
    axes[0].set_title('Curva de Aprendizado (Loss)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Previsão vs Real
    axes[1].plot(y_real, label='Real', color='blue')
    axes[1].plot(y_pred, label='Previsto', color='red', linestyle='--')
    axes[1].set_title(f'Teste: Real vs Previsto ({ticker})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plot_path = MODELS_DIR / f"{ticker}_training_results.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Gráfico salvo em {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default=DEFAULT_TICKER)
    args = parser.parse_args()
    
    train_model(ticker=args.ticker)

if __name__ == "__main__":
    main()