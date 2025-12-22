"""
Módulo de definição do modelo LSTM - Réplica do Original
"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, List, Optional, Dict
import logging
from pathlib import Path

from config import (
    LSTM_UNITS_1, LSTM_UNITS_2, DROPOUT_RATE,
    EPOCHS, BATCH_SIZE, EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE,
    MODELS_DIR, SEQUENCE_LENGTH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockLSTMModel:
    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        n_features: int = 1, # Univariado
        lstm_units_1: int = LSTM_UNITS_1,
        lstm_units_2: int = LSTM_UNITS_2,
        dropout_rate: float = DROPOUT_RATE,
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dropout_rate = dropout_rate
        self.model: Optional[Sequential] = None
        self.history = None
    
    def build_model(self) -> Sequential:
        """
        Replicando a função build_model do arquivo original:
        - 2 Camadas LSTM
        - Dropout após cada uma
        - Loss: MEAN SQUARED ERROR (Crucial!)
        """
        model = Sequential()
        
        # 1ª Camada
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        model.add(LSTM(
            units=self.lstm_units_1,
            return_sequences=True, # Necessário pois tem a 2ª camada
        ))
        model.add(Dropout(self.dropout_rate))
        
        # 2ª Camada (Original tinha n_layers=2)
        model.add(LSTM(
            units=self.lstm_units_2,
            return_sequences=False 
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Saída
        model.add(Dense(1))
        
        # --- DIFERENÇA CRÍTICA AQUI ---
        # O original usava 'mean_squared_error'. 
        # O novo estava usando 'huber'. Voltamos para o MSE.
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        self.model = model
        logger.info(f"Modelo construído (Réplica Original): MSE Loss, 2 Layers, {self.lstm_units_1} units")
        return model
    
    def get_callbacks(self, ticker: str) -> List:
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            # O original não tinha ReduceLR explicitamente no grid search, 
            # mas é boa prática manter.
            ModelCheckpoint(
                filepath=str(MODELS_DIR / f"{ticker}_best_model.keras"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, ticker, epochs=EPOCHS, batch_size=BATCH_SIZE):
        if self.model is None:
            self.build_model()
            
        callbacks = self.get_callbacks(ticker)
        
        # Validação usando o próprio teste (X_val = X_test), igual ao original
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0)
    
    def save_model(self, ticker: str) -> str:
        filepath = MODELS_DIR / f"{ticker}_lstm_model.keras"
        self.model.save(filepath)
        return str(filepath)
    
    def load_model(self, ticker: str) -> None:
        filepath = MODELS_DIR / f"{ticker}_lstm_model.keras"
        if not filepath.exists():
             filepath = MODELS_DIR / f"{ticker}_best_model.keras"
        self.model = load_model(filepath)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    
    direction_accuracy = 0.0
    if len(y_true) > 1:
        diff_true = np.diff(y_true)
        diff_pred = y_pred[1:] - y_true[:-1]
        correct = (np.sign(diff_true) == np.sign(diff_pred))
        direction_accuracy = np.mean(correct) * 100

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "direction_accuracy": float(direction_accuracy)
    }