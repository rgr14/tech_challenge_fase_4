"""
Módulo de definição do modelo LSTM para previsão de ações
"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# Removemos o L2 import
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, List, Optional, Dict
import logging
from pathlib import Path

from config import (
    LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS, DROPOUT_RATE,
    EPOCHS, BATCH_SIZE, EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE,
    MODELS_DIR, SEQUENCE_LENGTH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockLSTMModel:
    
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
        Constrói a arquitetura do modelo (SEM Regularização L2).
        Voltamos para a versão que funcionava, focada em capturar tendência.
        """
        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),

            # 1ª Camada LSTM (Potente)
            LSTM(
                units=self.lstm_units_1,
                return_sequences=True,
                activation='tanh'
                # Sem kernel_regularizer
            ),
            Dropout(self.dropout_rate),

            # 2ª Camada LSTM (Potente)
            LSTM(
                units=self.lstm_units_2,
                return_sequences=False,
                activation='tanh'
                # Sem kernel_regularizer
            ),
            Dropout(self.dropout_rate),

            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info(f"Modelo construído: 2 Camadas LSTM ({self.lstm_units_1}, {self.lstm_units_2}) - High Performance")
        return model
    
    def get_callbacks(self, ticker: str) -> List:
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            ),
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
        
        if X_train.shape[2] != self.n_features:
            self.n_features = X_train.shape[2]
            self.build_model()
        
        callbacks = self.get_callbacks(ticker)
        
        logger.info(f"Iniciando treinamento para {ticker} por {epochs} épocas...")
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
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        metrics = self.model.evaluate(X, y, verbose=0)
        return dict(zip(self.model.metrics_names, metrics))

    def save_model(self, ticker: str) -> str:
        if self.model is None:
            raise ValueError("Nenhum modelo para salvar.")
        filepath = MODELS_DIR / f"{ticker}_lstm_model.keras"
        self.model.save(filepath)
        return str(filepath)
    
    def load_model(self, ticker: str) -> None:
        filepath = MODELS_DIR / f"{ticker}_lstm_model.keras"
        if not filepath.exists():
             filepath = MODELS_DIR / f"{ticker}_best_model.keras"
        if not filepath.exists():
            raise FileNotFoundError(f"Nenhum modelo encontrado para {ticker}") 
        self.model = load_model(filepath)
        self.n_features = self.model.input_shape[-1]
        self.sequence_length = self.model.input_shape[1]
        logger.info(f"Modelo carregado de: {filepath}")

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan

    direction_accuracy = 0.0
    if len(y_true) > 1:
        diff_true = np.diff(y_true)
        diff_pred = y_pred[1:] - y_true[:-1]
        correct_direction = (np.sign(diff_true) == np.sign(diff_pred))
        direction_accuracy = np.mean(correct_direction) * 100

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "direction_accuracy": float(direction_accuracy)
    }