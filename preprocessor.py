"""
Módulo de pré-processamento de dados para LSTM
==============================================
Inclui validação, limpeza, feature engineering e normalização.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import Tuple, Optional, List, Dict
import joblib
import logging
import warnings

from config import (
    SEQUENCE_LENGTH, TRAIN_TEST_SPLIT, FEATURES, TARGET,
    SCALERS_DIR
)
from data_validator import DataValidator, DataCleaner, validate_and_clean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataPreprocessor:
    """Classe para pré-processamento de dados de ações para LSTM."""
    
    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        features: List[str] = FEATURES,
        target: str = TARGET,
        scaler_type: str = "minmax",  # 'minmax' ou 'robust'
        validate_data: bool = True,
        auto_clean: bool = True
    ):
        self.sequence_length = sequence_length
        self.features = features
        self.target = target
        self.validate_data = validate_data
        self.auto_clean = auto_clean
        
        # Escolher tipo de scaler
        if scaler_type == "robust":
            # RobustScaler é menos sensível a outliers
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
        else:
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.scaler_type = scaler_type
        self.is_fitted = False
        self.data_quality_report = None
        self.cleaning_stats: Dict = {}
    
    def validate_and_clean_data(
        self,
        df: pd.DataFrame,
        ticker: str = "UNKNOWN"
    ) -> pd.DataFrame:
        """
        Valida e limpa os dados antes do processamento.
        
        Args:
            df: DataFrame com dados brutos
            ticker: Símbolo da ação
        
        Returns:
            DataFrame validado e limpo
        """
        original_len = len(df)
        
        if not self.validate_data:
            logger.info("Validação desabilitada, pulando...")
            return df
        
        logger.info("=" * 50)
        logger.info("VALIDAÇÃO E LIMPEZA DE DADOS")
        logger.info("=" * 50)
        
        # Validar e limpar
        df_clean, report = validate_and_clean(
            df,
            ticker=ticker,
            auto_clean=self.auto_clean,
            verbose=True
        )
        
        self.data_quality_report = report
        
        # Estatísticas de limpeza
        self.cleaning_stats = {
            "original_records": original_len,
            "final_records": len(df_clean),
            "removed_records": original_len - len(df_clean),
            "removal_percentage": ((original_len - len(df_clean)) / original_len) * 100,
            "is_valid": report.is_valid,
            "warnings": len(report.warnings),
            "errors": len(report.errors)
        }
        
        if not report.is_valid:
            logger.warning("⚠️ ATENÇÃO: Dados têm problemas de qualidade!")
            logger.warning("O modelo pode ter performance reduzida.")
        
        return df_clean
    
    def prepare_features(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> pd.DataFrame:
        """
        Versão SIMPLIFICADA: Apenas retorna os dados limpos, sem criar indicadores.
        Isso imita o comportamento do código 'vencedor' original.
        """
        # 1. Validar e limpar (mantemos isso pois é segurança básica)
        df = self.validate_and_clean_data(df, ticker)
        
        # 2. Se a configuração pede apenas 'Close', retornamos apenas isso
        # sem calcular RSI, MACD, etc.
        if len(self.features) == 1 and 'Close' in self.features:
            logger.info("Modo Simplificado: Usando apenas coluna Close (Univariado)")
            return df[['Close']]
            
        # --- CÓDIGO ANTIGO FICARIA AQUI, MAS VAMOS PULAR ---
        # Se quiser usar as features complexas no futuro, a lógica original entraria aqui.
        # Por enquanto, retornamos apenas as colunas solicitadas no config.
        
        available_features = [col for col in self.features if col in df.columns]
        return df[available_features]
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula o RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26
    ) -> pd.Series:
        """Calcula o MACD."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow
    
    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """Calcula as Bandas de Bollinger."""
        ma = prices.rolling(window=period, min_periods=1).mean()
        std = prices.rolling(window=period, min_periods=1).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calcula o ATR (Average True Range).
        Mede a volatilidade do mercado.
        """
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def _calculate_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calcula o Stochastic Oscillator.
        %K e %D indicam sobrecompra/sobrevenda.
        """
        low_min = df['Low'].rolling(window=k_period, min_periods=1).min()
        high_max = df['High'].rolling(window=k_period, min_periods=1).max()
        
        # Evitar divisão por zero
        denominator = high_max - low_min
        denominator = denominator.replace(0, np.nan)
        
        stoch_k = ((df['Close'] - low_min) / denominator) * 100
        stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
        
        return stoch_k, stoch_d
    
    def _handle_feature_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata problemas comuns em features calculadas.
        
        - Valores infinitos
        - NaN isolados
        - Valores extremos
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Substituir infinitos por NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Interpolar NaN isolados (não nas bordas)
            if df[col].isna().sum() > 0 and df[col].isna().sum() < len(df) * 0.1:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            # Clipar valores extremos (além de 5 desvios padrão)
            if col not in ['Close', 'Open', 'High', 'Low', 'Volume']:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    lower = mean - 5 * std
                    upper = mean + 5 * std
                    df[col] = df[col].clip(lower, upper)
        
        return df
    
    def fit_scalers(self, df: pd.DataFrame) -> None:
        """
        Ajusta os scalers com os dados de treino.
        
        Args:
            df: DataFrame com dados de treino
        """
        # Selecionar features disponíveis
        available_features = [f for f in self.features if f in df.columns]
        
        # Fit do scaler de features
        self.feature_scaler.fit(df[available_features].values)
        
        # Fit do scaler do target
        self.target_scaler.fit(df[[self.target]].values)
        
        self.is_fitted = True
        self.fitted_features = available_features
        logger.info(f"Scalers ajustados com {len(available_features)} features")
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforma os dados usando os scalers.
        
        Args:
            df: DataFrame com dados
        
        Returns:
            Tuple com features e target normalizados
        """
        if not self.is_fitted:
            raise ValueError("Scalers não ajustados. Execute fit_scalers primeiro.")
        
        features_scaled = self.feature_scaler.transform(
            df[self.fitted_features].values
        )
        target_scaled = self.target_scaler.transform(
            df[[self.target]].values
        )
        
        return features_scaled, target_scaled
    
    def create_sequences(
        self,
        features: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria sequências para treinamento do LSTM.
        
        Args:
            features: Array de features normalizadas
            target: Array de target normalizado
        
        Returns:
            Tuple com X (sequências) e y (targets)
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i - self.sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = TRAIN_TEST_SPLIT
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide os dados em treino e teste (sem shuffle para séries temporais).
        
        Args:
            X: Features
            y: Target
            train_ratio: Proporção de dados para treino
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * train_ratio)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_predictions(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Reverte a normalização das previsões.
        
        Args:
            predictions: Previsões normalizadas
        
        Returns:
            Previsões na escala original
        """
        return self.target_scaler.inverse_transform(predictions)
    
    def prepare_prediction_data(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:
        """
        Prepara dados para fazer previsões (últimos N dias).
        
        Args:
            df: DataFrame com dados recentes
        
        Returns:
            Array formatado para previsão
        """
        if not self.is_fitted:
            raise ValueError("Scalers não ajustados. Carregue um modelo treinado.")
        
        # Garantir que temos dados suficientes
        if len(df) < self.sequence_length:
            raise ValueError(
                f"Necessário pelo menos {self.sequence_length} registros, "
                f"recebido {len(df)}"
            )
        
        # Usar os últimos sequence_length registros
        recent_data = df.tail(self.sequence_length)
        
        # Transformar
        features_scaled = self.feature_scaler.transform(
            recent_data[self.fitted_features].values
        )
        
        # Formatar para LSTM: (1, sequence_length, n_features)
        return features_scaled.reshape(1, self.sequence_length, -1)
    
    def save_scalers(self, ticker: str) -> None:
        """Salva os scalers treinados."""
        joblib.dump(
            self.feature_scaler,
            SCALERS_DIR / f"{ticker}_feature_scaler.pkl"
        )
        joblib.dump(
            self.target_scaler,
            SCALERS_DIR / f"{ticker}_target_scaler.pkl"
        )
        joblib.dump(
            self.fitted_features,
            SCALERS_DIR / f"{ticker}_features_list.pkl"
        )
        logger.info(f"Scalers salvos em {SCALERS_DIR}")
    
    def load_scalers(self, ticker: str) -> None:
        """Carrega scalers salvos."""
        self.feature_scaler = joblib.load(
            SCALERS_DIR / f"{ticker}_feature_scaler.pkl"
        )
        self.target_scaler = joblib.load(
            SCALERS_DIR / f"{ticker}_target_scaler.pkl"
        )
        self.fitted_features = joblib.load(
            SCALERS_DIR / f"{ticker}_features_list.pkl"
        )
        self.is_fitted = True
        logger.info(f"Scalers carregados de {SCALERS_DIR}")


def main():
    """Exemplo de uso do preprocessador com validação."""
    from data_collector import StockDataCollector
    
    print("=" * 60)
    print("DEMONSTRAÇÃO DO PREPROCESSADOR COM VALIDAÇÃO")
    print("=" * 60)
    
    # Coletar dados
    collector = StockDataCollector("NVDA")
    df = collector.fetch_historical_data()
    
    print(f"\n📥 Dados coletados: {len(df)} registros")
    print(f"📅 Período: {df.index.min()} até {df.index.max()}")
    
    # Preprocessar COM validação
    preprocessor = StockDataPreprocessor(
        features=FEATURES,
        validate_data=True,
        auto_clean=True,
        scaler_type="minmax"
    )
    
    # Preparar features (inclui validação automática)
    df_features = preprocessor.prepare_features(df, ticker="NVDA")
    
    print(f"\n📊 Features criadas: {len(df_features.columns)} colunas")
    print(f"📋 Lista de features:")
    for i, col in enumerate(df_features.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Estatísticas de limpeza
    if preprocessor.cleaning_stats:
        stats = preprocessor.cleaning_stats
        print(f"\n📈 Estatísticas de Limpeza:")
        print(f"   Registros originais: {stats['original_records']}")
        print(f"   Registros finais: {stats['final_records']}")
        print(f"   Removidos: {stats['removed_records']} ({stats['removal_percentage']:.2f}%)")
    
    # Fit e transform
    preprocessor.fit_scalers(df_features)
    features_scaled, target_scaled = preprocessor.transform(df_features)
    
    print(f"\n🔢 Dados normalizados:")
    print(f"   Features shape: {features_scaled.shape}")
    print(f"   Target shape: {target_scaled.shape}")
    print(f"   Features range: [{features_scaled.min():.4f}, {features_scaled.max():.4f}]")
    
    # Criar sequências
    X, y = preprocessor.create_sequences(features_scaled, target_scaled)
    print(f"\n🔄 Sequências criadas:")
    print(f"   X shape: {X.shape} (samples, timesteps, features)")
    print(f"   y shape: {y.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)
    print(f"\n✂️ Train/Test Split:")
    print(f"   Train: X={X_train.shape}, y={y_train.shape}")
    print(f"   Test: X={X_test.shape}, y={y_test.shape}")
    
    print("\n" + "=" * 60)
    print("✅ Preprocessamento concluído com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()