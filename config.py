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
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = None

SEQUENCE_LENGTH = 30  

TRAIN_TEST_SPLIT = 0.8
FEATURES = ["Close", "Volume", "High", "Low", "Open"]
TARGET = "Close"

# --- VOLTANDO PARA A CONFIGURAÇÃO FORTE ---
LSTM_UNITS_1 = 100  # <--- De volta para 100
LSTM_UNITS_2 = 100  # <--- De volta para 100
DENSE_UNITS = 32

# Mantemos o Dropout em 0.2 (Padrão de mercado)
DROPOUT_RATE = 0.2

# Treinamento
EPOCHS = 100
BATCH_SIZE = 32

EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5

API_HOST = "0.0.0.0"
API_PORT = 8000