"""
Módulo de definição do modelo LSTM para previsão de ações
"""
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from typing import Tuple, List, Optional
import logging

from config import (
    LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS, DROPOUT_RATE,
    EPOCHS, BATCH_SIZE, EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE,
    MODELS_DIR, SEQUENCE_LENGTH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockLSTMModel:
    """Classe para construção e treinamento do modelo LSTM."""
    
    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        n_features: int = 5,
        lstm_units_1: int = LSTM_UNITS_1,
        lstm_units_2: int = LSTM_UNITS_2,
        dense_units: int = DENSE_UNITS,
        dropout_rate: float = DROPOUT_RATE,
        learning_rate: float = 0.001
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model: Optional[Sequential] = None
        self.history = None
    
    def build_model(self) -> Sequential:
        """
        Constrói a arquitetura do modelo LSTM.
        
        Returns:
            Modelo Keras compilado
        """
        model = Sequential([
            # Input layer
            Input(shape=(self.sequence_length, self.n_features)),
            
            # Primeira camada LSTM
            LSTM(
                units=self.lstm_units_1,
                return_sequences=True,
                kernel_regularizer=l2(0.001)
            ),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Segunda camada LSTM
            LSTM(
                units=self.lstm_units_2,
                return_sequences=True,
                kernel_regularizer=l2(0.001)
            ),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Terceira camada LSTM
            LSTM(
                units=self.lstm_units_2 // 2,
                return_sequences=False,
                kernel_regularizer=l2(0.001)
            ),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Camadas densas
            Dense(self.dense_units, activation='relu'),
            Dropout(self.dropout_rate / 2),
            
            Dense(self.dense_units // 2, activation='relu'),
            
            # Output layer
            Dense(1, activation='linear')
        ])
        
        # Compilar modelo
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Mais robusto a outliers que MSE
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info("Modelo LSTM construído com sucesso")
        self._print_model_summary()
        
        return model
    
    def _print_model_summary(self) -> None:
        """Imprime o resumo do modelo."""
        if self.model:
            self.model.summary()
    
    def get_callbacks(self, ticker: str) -> List:
        """
        Retorna callbacks para treinamento.
        
        Args:
            ticker: Símbolo da ação para nomear checkpoints
        
        Returns:
            Lista de callbacks
        """
        callbacks = [
            # Early stopping para evitar overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduzir learning rate quando estabilizar
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Salvar melhor modelo
            ModelCheckpoint(
                filepath=str(MODELS_DIR / f"{ticker}_best_model.keras"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        ticker: str,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE
    ) -> dict:
        """
        Treina o modelo LSTM.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação
            y_val: Target de validação
            ticker: Símbolo da ação
            epochs: Número de épocas
            batch_size: Tamanho do batch
        
        Returns:
            Histórico de treinamento
        """
        if self.model is None:
            self.build_model()
        
        # Atualizar n_features baseado nos dados
        self.n_features = X_train.shape[2]
        self.model = None  # Resetar para reconstruir com features corretos
        self.build_model()
        
        logger.info(f"Iniciando treinamento: {epochs} épocas, batch_size={batch_size}")
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
        
        callbacks = self.get_callbacks(ticker)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Avalia o modelo.
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
        
        Returns:
            Tuple com (loss, mae, mse)
        """
        if self.model is None:
            raise ValueError("Modelo não treinado. Execute train primeiro.")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        loss, mae, mse = results
        
        logger.info(f"Avaliação - Loss: {loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
        
        return loss, mae, mse
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz previsões.
        
        Args:
            X: Features para previsão
        
        Returns:
            Array de previsões
        """
        if self.model is None:
            raise ValueError("Modelo não treinado ou carregado.")
        
        return self.model.predict(X, verbose=0)
    
    def save_model(self, ticker: str) -> str:
        """
        Salva o modelo treinado.
        
        Args:
            ticker: Símbolo da ação
        
        Returns:
            Caminho do arquivo salvo
        """
        if self.model is None:
            raise ValueError("Nenhum modelo para salvar.")
        
        filepath = MODELS_DIR / f"{ticker}_lstm_model.keras"
        self.model.save(filepath)
        logger.info(f"Modelo salvo em {filepath}")
        
        return str(filepath)
    
    def load_model(self, ticker: str) -> None:
        """
        Carrega um modelo salvo.
        
        Args:
            ticker: Símbolo da ação
        """
        filepath = MODELS_DIR / f"{ticker}_lstm_model.keras"
        
        if not filepath.exists():
            # Tentar carregar o best model
            filepath = MODELS_DIR / f"{ticker}_best_model.keras"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Modelo não encontrado para {ticker}")
        
        self.model = load_model(filepath)
        logger.info(f"Modelo carregado de {filepath}")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Calcula métricas de avaliação.
    
    Args:
        y_true: Valores reais
        y_pred: Valores previstos
    
    Returns:
        Dicionário com métricas
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Direction Accuracy (acertou a direção do movimento?)
    if len(y_true) > 1:
        direction_true = np.diff(y_true.flatten()) > 0
        direction_pred = np.diff(y_pred.flatten()) > 0
        direction_accuracy = np.mean(direction_true == direction_pred) * 100
    else:
        direction_accuracy = None
    
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "direction_accuracy": float(direction_accuracy) if direction_accuracy else None
    }


def main():
    """Exemplo de uso do modelo."""
    # Criar modelo
    model = StockLSTMModel(
        sequence_length=60,
        n_features=5
    )
    
    # Build
    model.build_model()
    
    # Dados sintéticos para teste
    X_dummy = np.random.rand(100, 60, 5)
    y_dummy = np.random.rand(100, 1)
    
    print(f"\nFormato dos dados: X={X_dummy.shape}, y={y_dummy.shape}")


if __name__ == "__main__":
    main()
