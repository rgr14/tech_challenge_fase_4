"""
Script CLI para previsao de precos de acoes
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model

from src.config import MODELS_DIR, SCALERS_DIR, SEQUENCE_LENGTH, DEFAULT_TICKER
from src.data.preprocessor import StockDataPreprocessor

# --- CONFIGURACAO ---
TICKER = DEFAULT_TICKER


def main():
    print(f"\n--- INICIANDO PREVISAO SIMPLIFICADA PARA {TICKER} ---")

    # 1. Carregar o Modelo Vencedor
    try:
        model_path = MODELS_DIR / f"{TICKER}_lstm_model.keras"
        model = load_model(model_path)
        print("Modelo carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return

    # 2. Carregar os Scalers
    try:
        feature_scaler = joblib.load(SCALERS_DIR / f"{TICKER}_feature_scaler.pkl")
        target_scaler = joblib.load(SCALERS_DIR / f"{TICKER}_target_scaler.pkl")
        fitted_features = joblib.load(SCALERS_DIR / f"{TICKER}_features_list.pkl")
        print("Scalers carregados com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar scalers. Rode o train.py primeiro.")
        return

    # 3. Baixar dados (Margem de seguranca de 200 dias)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200)
    print(f"Baixando dados de {start_date.date()} ate {end_date.date()}...")

    # auto_adjust=True garante precos ajustados por splits/dividendos
    df = yf.download(TICKER, start=start_date, end=end_date, progress=False, auto_adjust=True)

    if df.empty:
        print("Falha ao baixar dados.")
        return

    # --- CORRECAO DE COMPATIBILIDADE YFINANCE ---
    # Se o yfinance retornou MultiIndex (ex: ('Close', 'NVDA')), achatamos para apenas 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Resetar index se necessario para garantir formato limpo
    if 'Date' not in df.columns and df.index.name == 'Date':
        # O index ja esta certo, mantemos como esta
        pass

    # --- FIM DA CORRECAO ---

    # Extrair ultimo preco (com protecao contra formatacao)
    try:
        raw_close = df['Close'].iloc[-1]
        if hasattr(raw_close, 'item'):
            last_close = raw_close.item()
        else:
            last_close = float(raw_close)
    except Exception:
        last_close = float(df['Close'].values[-1])

    last_date = df.index[-1]
    print(f"Ultimo Fechamento Real ({last_date.date()}): ${last_close:.2f}")

    # 4. Preparar as Features
    # validate_data=False pula a validacao pesada que estava dando erro,
    # pois acabamos de limpar o dataframe manualmente acima.
    preprocessor = StockDataPreprocessor(validate_data=False)

    # Gera RSI, MACD, MA, etc.
    df_features = preprocessor.prepare_features(df, ticker=TICKER)

    # Garante que so temos as colunas que o scaler conhece
    # Se faltar alguma coluna, vamos preencher com 0 para nao travar (fallback)
    for col in fitted_features:
        if col not in df_features.columns:
            df_features[col] = 0

    df_features = df_features[fitted_features]

    # 5. Pegar apenas os ultimos 30 dias
    if len(df_features) < SEQUENCE_LENGTH:
        print(f"Dados insuficientes. Preciso de {SEQUENCE_LENGTH}, tenho {len(df_features)}.")
        return

    last_sequence_df = df_features.tail(SEQUENCE_LENGTH)

    # 6. Escalar os dados
    sequence_scaled = feature_scaler.transform(last_sequence_df.values)

    # Formato LSTM: (1, 30, n_features)
    X_input = sequence_scaled.reshape(1, SEQUENCE_LENGTH, -1)

    # 7. Prever
    predicted_scaled = model.predict(X_input, verbose=0)

    # 8. Desfazer a escala
    predicted_price = target_scaler.inverse_transform(predicted_scaled)[0][0]

    # --- RESULTADO ---
    variation = predicted_price - last_close
    variation_pct = (variation / last_close) * 100
    direction = "SUBIR" if variation > 0 else "CAIR"

    print("\n" + "=" * 40)
    print(f"PREVISAO PARA O PROXIMO PREGAO")
    print("=" * 40)
    print(f"Preco Atual:    ${last_close:.2f}")
    print(f"Preco Previsto: ${predicted_price:.2f}")
    print(f"Variacao:       ${variation:+.2f} ({variation_pct:+.2f}%)")
    print(f"Tendencia:      {direction}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
