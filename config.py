"""
Configurações centralizadas do projeto LSTM Stock Predictor
"""
from pathlib import Path

# Diretórios
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SCALERS_DIR = BASE_DIR / "scalers"

# Criar diretórios se não existirem
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
SCALERS_DIR.mkdir(exist_ok=True)

# Configurações de dados
DEFAULT_TICKER = "NVDA"  # NVIDIA
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = None  # None = hoje

# Configurações do modelo LSTM
SEQUENCE_LENGTH = 60  # Dias de histórico para prever o próximo
TRAIN_TEST_SPLIT = 0.8
FEATURES = ["Close", "Volume", "High", "Low", "Open"]
TARGET = "Close"

# Hiperparâmetros do modelo
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.2

# Treinamento
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
