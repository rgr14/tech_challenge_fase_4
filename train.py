"""
Script principal de treinamento do modelo LSTM com Grid Search
==============================================================
Suporta treinamento simples e Grid Search com TimeSeriesSplit.
"""
import argparse
import json
import logging
import random
from datetime import datetime
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

from config import (
    DEFAULT_TICKER, DEFAULT_START_DATE, EPOCHS, BATCH_SIZE,
    MODELS_DIR, DATA_DIR, FEATURES, TRAIN_TEST_SPLIT,
    SEQUENCE_LENGTH, LSTM_UNITS_1, DROPOUT_RATE
)
from data_collector import StockDataCollector
from preprocessor import StockDataPreprocessor
from model import StockLSTMModel, calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACAO DO GRID SEARCH
# ============================================================
GRID_SEARCH_PARAMS = {
    "sequence_length": [15, 30, 60],
    "lstm_units": [50, 100, 150],
    "dropout_rate": [0.1, 0.2, 0.3],
    "batch_size": [16, 32, 64],
    "epochs": [50, 100],
    "learning_rate": [0.001, 0.0001],
    "n_layers": [1, 2, 3]
}

# Numero de splits para TimeSeriesSplit
N_SPLITS = 3

# Limite maximo de combinacoes (para evitar estouro de memoria)
MAX_COMBINATIONS = 100


def create_model_with_params(
    sequence_length: int,
    n_features: int,
    lstm_units: int,
    dropout_rate: float,
    n_layers: int,
    learning_rate: float
):
    """
    Cria um modelo LSTM com parametros customizados.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    model.add(Input(shape=(sequence_length, n_features)))

    for i in range(n_layers):
        return_sequences = (i < n_layers - 1)
        model.add(LSTM(units=lstm_units, return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def train_single_fold(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    verbose: int = 0
) -> dict:
    """
    Treina o modelo em um unico fold e retorna metricas.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )

    # Previsoes
    y_val_pred = model.predict(X_val, verbose=0)

    # Metricas (em escala normalizada)
    val_loss = min(history.history['val_loss'])

    return {
        "val_loss": val_loss,
        "epochs_trained": len(history.history['loss']),
        "y_val_pred": y_val_pred,
        "y_val_true": y_val
    }


def run_grid_search(
    ticker: str = DEFAULT_TICKER,
    start_date: str = DEFAULT_START_DATE,
    n_splits: int = N_SPLITS,
    save_results: bool = True
) -> dict:
    """
    Executa Grid Search com TimeSeriesSplit.
    """
    logger.info("=" * 70)
    logger.info("INICIANDO GRID SEARCH COM TIMESERIESSPLIT")
    logger.info("=" * 70)

    # 1. Coleta de dados
    logger.info("\n[1/4] Coletando dados...")
    collector = StockDataCollector(ticker)
    df = collector.fetch_historical_data(start_date=start_date)
    collector.save_data()

    logger.info(f"Coletados {len(df)} registros")

    # 2. Preparar dados base (sem sequencias ainda - faremos por config)
    logger.info("\n[2/4] Preparando dados...")

    # Calcular total de combinacoes
    param_keys = list(GRID_SEARCH_PARAMS.keys())
    param_values = list(GRID_SEARCH_PARAMS.values())
    all_combinations = list(product(*param_values))
    total_combinations = len(all_combinations)

    # Limitar combinacoes se necessario (Random Search)
    if total_combinations > MAX_COMBINATIONS:
        logger.info(f"Total de combinacoes possiveis: {total_combinations}")
        logger.info(f"Limitando para {MAX_COMBINATIONS} combinacoes (Random Search)")
        random.seed(42)  # Reproducibilidade
        all_combinations = random.sample(all_combinations, MAX_COMBINATIONS)
        total_combinations = len(all_combinations)

    logger.info(f"Total de combinacoes a testar: {total_combinations}")
    logger.info(f"Folds por combinacao: {n_splits}")
    logger.info(f"Total de treinamentos: {total_combinations * n_splits}")

    # 3. Grid Search
    logger.info("\n[3/4] Executando Grid Search...")
    results = []
    best_result = None
    best_score = float('inf')

    for idx, combination in enumerate(all_combinations, 1):
        params = dict(zip(param_keys, combination))

        logger.info(f"\n--- Combinacao {idx}/{total_combinations} ---")
        logger.info(f"Params: {params}")

        try:
            # Preprocessar com sequence_length especifico
            preprocessor = StockDataPreprocessor(
                sequence_length=params["sequence_length"],
                features=FEATURES,
                validate_data=True,
                auto_clean=True,
                scaler_type="minmax"
            )

            df_features = preprocessor.prepare_features(df.copy(), ticker=ticker)
            preprocessor.fit_scalers(df_features)
            features_scaled, target_scaled = preprocessor.transform(df_features)
            X, y = preprocessor.create_sequences(features_scaled, target_scaled)

            if len(X) < n_splits * 10:
                logger.warning(f"Dados insuficientes para {n_splits} splits. Pulando...")
                continue

            # TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=n_splits)
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Criar modelo
                model = create_model_with_params(
                    sequence_length=params["sequence_length"],
                    n_features=X.shape[2],
                    lstm_units=params["lstm_units"],
                    dropout_rate=params["dropout_rate"],
                    n_layers=params["n_layers"],
                    learning_rate=params["learning_rate"]
                )

                # Treinar fold
                fold_result = train_single_fold(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=params["epochs"],
                    batch_size=params["batch_size"],
                    verbose=0
                )

                fold_scores.append(fold_result["val_loss"])

                # Limpar memoria
                del model
                import tensorflow as tf
                tf.keras.backend.clear_session()

            # Media dos folds
            mean_val_loss = np.mean(fold_scores)
            std_val_loss = np.std(fold_scores)

            result = {
                **params,
                "mean_val_loss": float(mean_val_loss),
                "std_val_loss": float(std_val_loss),
                "fold_scores": [float(s) for s in fold_scores],
                "n_samples": len(X)
            }
            results.append(result)

            logger.info(f"Val Loss: {mean_val_loss:.6f} (+/- {std_val_loss:.6f})")

            # Atualizar melhor resultado
            if mean_val_loss < best_score:
                best_score = mean_val_loss
                best_result = result.copy()
                logger.info(">>> NOVO MELHOR MODELO! <<<")

        except Exception as e:
            logger.error(f"Erro na combinacao {idx}: {e}")
            continue

    # 4. Salvar resultados
    logger.info("\n[4/4] Salvando resultados...")

    if save_results and results:
        # Ordenar por performance
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("mean_val_loss")

        # Salvar CSV
        csv_path = MODELS_DIR / f"{ticker}_grid_search_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Resultados salvos em: {csv_path}")

        # Salvar JSON com melhor resultado
        if best_result:
            json_path = MODELS_DIR / f"{ticker}_best_params.json"
            with open(json_path, "w") as f:
                json.dump(best_result, f, indent=2)
            logger.info(f"Melhores parametros salvos em: {json_path}")

    # Exibir resumo
    logger.info("\n" + "=" * 70)
    logger.info("RESUMO DO GRID SEARCH")
    logger.info("=" * 70)

    if best_result:
        logger.info(f"\nMelhores parametros encontrados:")
        for key, value in best_result.items():
            if key not in ["fold_scores"]:
                logger.info(f"  {key}: {value}")

    logger.info(f"\nTotal de combinacoes testadas: {len(results)}/{total_combinations}")

    return {
        "best_params": best_result,
        "all_results": results,
        "ticker": ticker
    }


def train_with_best_params(
    ticker: str = DEFAULT_TICKER,
    start_date: str = DEFAULT_START_DATE,
    save_plots: bool = True
) -> dict:
    """
    Treina o modelo final usando os melhores parametros do Grid Search.
    """
    # Carregar melhores parametros
    json_path = MODELS_DIR / f"{ticker}_best_params.json"

    if not json_path.exists():
        logger.warning(f"Arquivo de melhores parametros nao encontrado: {json_path}")
        logger.info("Usando parametros padrao do config.py")
        params = {
            "sequence_length": SEQUENCE_LENGTH,
            "lstm_units": LSTM_UNITS_1,
            "dropout_rate": DROPOUT_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": 0.001,
            "n_layers": 2
        }
    else:
        with open(json_path, "r") as f:
            params = json.load(f)
        logger.info(f"Parametros carregados de: {json_path}")

    logger.info("=" * 60)
    logger.info(f"TREINAMENTO FINAL - {ticker}")
    logger.info("=" * 60)
    logger.info(f"Parametros: {params}")

    # 1. Coleta de dados
    logger.info("\n[1/5] Coletando dados...")
    collector = StockDataCollector(ticker)
    df = collector.fetch_historical_data(start_date=start_date)
    collector.save_data()

    logger.info(f"Coletados {len(df)} registros")

    # 2. Pre-processamento
    logger.info("\n[2/5] Preparando features...")
    preprocessor = StockDataPreprocessor(
        sequence_length=params["sequence_length"],
        features=FEATURES,
        validate_data=True,
        auto_clean=True,
        scaler_type="minmax"
    )

    df_features = preprocessor.prepare_features(df, ticker=ticker)
    logger.info(f"Features usadas: {df_features.columns.tolist()}")

    # 3. Normalizacao e Sequenciamento
    logger.info("\n[3/5] Normalizando e criando sequencias...")
    preprocessor.fit_scalers(df_features)
    features_scaled, target_scaled = preprocessor.transform(df_features)
    X, y = preprocessor.create_sequences(features_scaled, target_scaled)
    logger.info(f"Sequencias criadas: X={X.shape}, y={y.shape}")

    # 4. Split dos dados
    logger.info("\n[4/5] Dividindo dados...")
    split_idx = int(len(X) * TRAIN_TEST_SPLIT)

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    logger.info(f"Train: {X_train.shape[0]} amostras ({TRAIN_TEST_SPLIT*100}%)")
    logger.info(f"Test:  {X_test.shape[0]} amostras ({(1-TRAIN_TEST_SPLIT)*100:.0f}%)")

    # 5. Treinamento
    logger.info("\n[5/5] Treinando modelo final...")

    model = create_model_with_params(
        sequence_length=params["sequence_length"],
        n_features=X_train.shape[2],
        lstm_units=params["lstm_units"],
        dropout_rate=params["dropout_rate"],
        n_layers=params["n_layers"],
        learning_rate=params["learning_rate"]
    )

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{ticker}_best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=callbacks,
        verbose=1
    )

    # 6. Avaliacao Final
    logger.info("\n" + "=" * 60)
    logger.info("AVALIACAO FINAL")
    logger.info("=" * 60)

    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)

    # Reverter escala
    y_train_real = preprocessor.inverse_transform_predictions(y_train)
    y_train_pred_real = preprocessor.inverse_transform_predictions(y_train_pred)
    y_test_real = preprocessor.inverse_transform_predictions(y_test)
    y_test_pred_real = preprocessor.inverse_transform_predictions(y_test_pred)

    # Calcular Metricas
    train_metrics = calculate_metrics(y_train_real, y_train_pred_real)
    test_metrics = calculate_metrics(y_test_real, y_test_pred_real)

    _print_metrics("Treino", train_metrics)
    _print_metrics("Teste", test_metrics)

    # Salvar modelo final
    model.save(str(MODELS_DIR / f"{ticker}_lstm_model.keras"))
    preprocessor.save_scalers(ticker)

    # Salvar parametros usados
    final_params_path = MODELS_DIR / f"{ticker}_final_params.json"
    with open(final_params_path, "w") as f:
        json.dump({
            **params,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "trained_at": datetime.now().isoformat()
        }, f, indent=2)

    if save_plots:
        _save_training_plots(history.history, y_test_real, y_test_pred_real, ticker)

    return {
        "ticker": ticker,
        "params": params,
        "metrics": {"train": train_metrics, "test": test_metrics}
    }


def train_model(
    ticker: str = DEFAULT_TICKER,
    start_date: str = DEFAULT_START_DATE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    save_plots: bool = True
) -> dict:
    """
    Treinamento simples (sem Grid Search) - mantido para compatibilidade.
    """
    logger.info(f"=" * 60)
    logger.info(f"Iniciando treinamento para {ticker}")
    logger.info(f"=" * 60)

    # 1. Coleta de dados
    logger.info("\n[1/5] Coletando dados...")
    collector = StockDataCollector(ticker)
    df = collector.fetch_historical_data(start_date=start_date)
    collector.save_data()
    stock_info = collector.get_stock_info()

    logger.info(f"Coletados {len(df)} registros de {df.index.min()} ate {df.index.max()}")

    # 2. Pre-processamento
    logger.info("\n[2/5] Preparando features...")
    preprocessor = StockDataPreprocessor(
        features=FEATURES,
        validate_data=True,
        auto_clean=True,
        scaler_type="minmax"
    )

    df_features = preprocessor.prepare_features(df, ticker=ticker)
    logger.info(f"Features usadas: {df_features.columns.tolist()}")

    # 3. Normalizacao e Sequenciamento
    logger.info("\n[3/5] Normalizando e criando sequencias...")
    preprocessor.fit_scalers(df_features)
    features_scaled, target_scaled = preprocessor.transform(df_features)
    X, y = preprocessor.create_sequences(features_scaled, target_scaled)
    logger.info(f"Sequencias criadas: X={X.shape}, y={y.shape}")

    # 4. Split dos dados
    logger.info("\n[4/5] Dividindo dados...")
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

    history = model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        ticker=ticker,
        epochs=epochs,
        batch_size=batch_size
    )

    # 6. Avaliacao Final
    logger.info("\n" + "=" * 60)
    logger.info("AVALIACAO FINAL")
    logger.info("=" * 60)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_real = preprocessor.inverse_transform_predictions(y_train)
    y_train_pred_real = preprocessor.inverse_transform_predictions(y_train_pred)
    y_test_real = preprocessor.inverse_transform_predictions(y_test)
    y_test_pred_real = preprocessor.inverse_transform_predictions(y_test_pred)

    train_metrics = calculate_metrics(y_train_real, y_train_pred_real)
    test_metrics = calculate_metrics(y_test_real, y_test_pred_real)

    _print_metrics("Treino", train_metrics)
    _print_metrics("Teste", test_metrics)

    model.save_model(ticker)
    preprocessor.save_scalers(ticker)

    if save_plots:
        _save_training_plots(history, y_test_real, y_test_pred_real, ticker)

    return {
        "ticker": ticker,
        "metrics": {"train": train_metrics, "test": test_metrics}
    }


def _print_metrics(label: str, metrics: dict):
    logger.info(f"\n{label}:")
    logger.info(f"  MAE: ${metrics['mae']:.2f}")
    logger.info(f"  RMSE: ${metrics['rmse']:.2f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    logger.info(f"  R2: {metrics['r2']:.4f}")

def _save_training_plots(history, y_real, y_pred, ticker):
    """Gera graficos de treinamento."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Loss
    axes[0].plot(history['loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_title('Curva de Aprendizado (Loss)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Previsao vs Real
    axes[1].plot(y_real, label='Real', color='blue')
    axes[1].plot(y_pred, label='Previsto', color='red', linestyle='--')
    axes[1].set_title(f'Teste: Real vs Previsto ({ticker})')
    axes[1].set_xlabel('Amostras')
    axes[1].set_ylabel('Preco ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = MODELS_DIR / f"{ticker}_training_results.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Grafico salvo em {plot_path}")


def _save_grid_search_plots(results: list, ticker: str):
    """Gera graficos do Grid Search."""
    if not results:
        return

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Val Loss por sequence_length
    ax = axes[0, 0]
    grouped = df.groupby('sequence_length')['mean_val_loss'].agg(['mean', 'std'])
    ax.bar(grouped.index.astype(str), grouped['mean'], yerr=grouped['std'], capsize=5)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Val Loss (mean)')
    ax.set_title('Val Loss por Sequence Length')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Val Loss por lstm_units
    ax = axes[0, 1]
    grouped = df.groupby('lstm_units')['mean_val_loss'].agg(['mean', 'std'])
    ax.bar(grouped.index.astype(str), grouped['mean'], yerr=grouped['std'], capsize=5)
    ax.set_xlabel('LSTM Units')
    ax.set_ylabel('Val Loss (mean)')
    ax.set_title('Val Loss por LSTM Units')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Val Loss por n_layers
    ax = axes[0, 2]
    grouped = df.groupby('n_layers')['mean_val_loss'].agg(['mean', 'std'])
    ax.bar(grouped.index.astype(str), grouped['mean'], yerr=grouped['std'], capsize=5)
    ax.set_xlabel('N Layers')
    ax.set_ylabel('Val Loss (mean)')
    ax.set_title('Val Loss por N Layers')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Val Loss por dropout_rate
    ax = axes[1, 0]
    grouped = df.groupby('dropout_rate')['mean_val_loss'].agg(['mean', 'std'])
    ax.bar(grouped.index.astype(str), grouped['mean'], yerr=grouped['std'], capsize=5)
    ax.set_xlabel('Dropout Rate')
    ax.set_ylabel('Val Loss (mean)')
    ax.set_title('Val Loss por Dropout Rate')
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Val Loss por learning_rate
    ax = axes[1, 1]
    grouped = df.groupby('learning_rate')['mean_val_loss'].agg(['mean', 'std'])
    ax.bar([f"{lr:.0e}" for lr in grouped.index], grouped['mean'], yerr=grouped['std'], capsize=5)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Val Loss (mean)')
    ax.set_title('Val Loss por Learning Rate')
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Top 10 melhores configuracoes
    ax = axes[1, 2]
    top10 = df.nsmallest(10, 'mean_val_loss')
    labels = [f"SL={r['sequence_length']}, U={r['lstm_units']}, L={r['n_layers']}"
              for _, r in top10.iterrows()]
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.3, 10))
    ax.barh(range(10), top10['mean_val_loss'], color=colors)
    ax.set_yticks(range(10))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Val Loss')
    ax.set_title('Top 10 Configuracoes')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plot_path = MODELS_DIR / f"{ticker}_grid_search_analysis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Grafico do Grid Search salvo em {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Treinamento de modelo LSTM para previsao de acoes'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        default=DEFAULT_TICKER,
        help='Simbolo da acao (ex: NVDA, AAPL)'
    )
    parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Executar Grid Search para encontrar melhores parametros'
    )
    parser.add_argument(
        '--train-best',
        action='store_true',
        help='Treinar modelo final usando melhores parametros do Grid Search'
    )
    parser.add_argument(
        '--splits',
        type=int,
        default=N_SPLITS,
        help=f'Numero de splits para TimeSeriesSplit (default: {N_SPLITS})'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=DEFAULT_START_DATE,
        help=f'Data inicial para coleta de dados (default: {DEFAULT_START_DATE})'
    )

    args = parser.parse_args()

    if args.grid_search:
        # Executar Grid Search
        result = run_grid_search(
            ticker=args.ticker,
            start_date=args.start_date,
            n_splits=args.splits
        )

        # Salvar graficos do Grid Search
        if result["all_results"]:
            _save_grid_search_plots(result["all_results"], args.ticker)

        logger.info("\n" + "=" * 70)
        logger.info("Grid Search concluido!")
        logger.info(f"Para treinar o modelo final, execute:")
        logger.info(f"  python train.py --ticker {args.ticker} --train-best")
        logger.info("=" * 70)

    elif args.train_best:
        # Treinar com melhores parametros
        train_with_best_params(
            ticker=args.ticker,
            start_date=args.start_date
        )

    else:
        # Treinamento simples (compatibilidade)
        train_model(ticker=args.ticker)


if __name__ == "__main__":
    main()
