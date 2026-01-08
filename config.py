"""
Configurações centralizadas - CLONE DO MODELO ORIGINAL VENCEDOR
"""
from pathlib import Path

# Diretórios
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SCALERS_DIR = BASE_DIR / "scalers"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
SCALERS_DIR.mkdir(exist_ok=True)

# Configurações de dados
DEFAULT_TICKER = "NVDA"
DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = None

# --- PARÂMETROS DO GRID SEARCH VENCEDOR ---
SEQUENCE_LENGTH = 30           # look_back=30
LSTM_UNITS_1 = 100             # lstm_units=100
LSTM_UNITS_2 = 100             # (Camada 2 igual a 1)
DENSE_UNITS = 32               # (Não usado explicitamente no original, mas mantemos)
DROPOUT_RATE = 0.2             # dropout=0.2

# Treinamento
EPOCHS = 100                   # epochs=100
BATCH_SIZE = 32                # batch=32
TRAIN_TEST_SPLIT = 0.8         # 80% treino

# --- CRÍTICO: MODO UNIVARIADO ---
# O original usava apenas df[['Close']]
FEATURES = ["Close"] 
TARGET = "Close"

EARLY_STOPPING_PATIENCE = 10   # O original usava 10
REDUCE_LR_PATIENCE = 5

API_HOST = "0.0.0.0"
API_PORT = 8000