# 📈 Stock LSTM Predictor

Sistema completo de previsão de preços de ações utilizando redes neurais **Long Short-Term Memory (LSTM)**, com validação robusta de dados e API REST para deploy em produção.

## 🎯 Objetivo

Prever o preço de fechamento de ações utilizando dados históricos e técnicas de Deep Learning, disponibilizando as previsões através de uma API RESTful.

## 🏗️ Arquitetura do Projeto

```
stock_lstm_predictor/
├── config.py              # Configurações centralizadas
├── data_collector.py      # Coleta de dados via Yahoo Finance
├── data_validator.py      # 🆕 Validação e limpeza de dados
├── preprocessor.py        # Feature engineering + normalização
├── model.py               # Arquitetura do modelo LSTM
├── train.py               # Pipeline de treinamento
├── predict.py             # Script de previsão
├── api.py                 # FastAPI para deploy
├── requirements.txt       # Dependências
├── data/                  # Dados históricos (gerado)
├── models/                # Modelos treinados (gerado)
└── scalers/               # Scalers salvos (gerado)
```

## 🔍 Pipeline de Dados

```
┌─────────────────────────────────────────────────────────────┐
│                    1. COLETA DE DADOS                       │
│                    (data_collector.py)                      │
│         Yahoo Finance API → Dados OHLCV históricos          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 2. VALIDAÇÃO DE DADOS 🆕                    │
│                   (data_validator.py)                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Dados       │ │ Duplicatas  │ │ Integridade         │   │
│  │ Faltantes   │ │ e Gaps      │ │ (High>=Low, etc)    │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Outliers    │ │ Valores     │ │ Variações           │   │
│  │ Detection   │ │ Infinitos   │ │ Extremas            │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    3. LIMPEZA DE DADOS 🆕                   │
│                     (data_validator.py)                     │
│  • Interpolação temporal de valores faltantes               │
│  • Clipping de outliers                                     │
│  • Correção de integridade OHLCV                            │
│  • Remoção de duplicatas                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 4. FEATURE ENGINEERING                      │
│                    (preprocessor.py)                        │
│  • Indicadores técnicos (RSI, MACD, Bollinger, ATR, etc)   │
│  • Médias móveis e razões                                   │
│  • Volatilidade e momentum                                  │
│  • Features temporais (dia da semana, mês)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    5. NORMALIZAÇÃO                          │
│                    (preprocessor.py)                        │
│           MinMaxScaler ou RobustScaler (0-1)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  6. CRIAÇÃO DE SEQUÊNCIAS                   │
│                    (preprocessor.py)                        │
│              60 dias histórico → 1 previsão                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    7. MODELO LSTM                           │
│                      (model.py)                             │
│                 3 camadas LSTM + Dense                      │
└─────────────────────────────────────────────────────────────┘
```

## 🔎 Validação de Dados (Novo!)

O módulo `data_validator.py` garante a qualidade dos dados antes do treinamento:

### Verificações Realizadas

| Verificação | Descrição |
|-------------|-----------|
| **Dados Faltantes** | Conta NaN por coluna, calcula percentual |
| **Duplicatas** | Detecta datas e linhas duplicadas |
| **Gaps Temporais** | Identifica dias de negociação faltando |
| **Outliers** | Detecta via IQR, Z-Score ou MAD |
| **Integridade** | Valida High >= Low, preços > 0, Volume >= 0 |
| **Valores Infinitos** | Detecta `inf` e `-inf` |
| **Variações Extremas** | Alerta variações > 50% em um dia |

### Métodos de Detecção de Outliers

```python
# IQR (Interquartile Range) - Padrão
outlier se: valor < Q1 - 3*IQR  ou  valor > Q3 + 3*IQR

# Z-Score
outlier se: |valor - média| / std > threshold

# MAD (Median Absolute Deviation) - Mais robusto
outlier se: |valor - mediana| / MAD > threshold
```

### Métodos de Tratamento

| Dados Faltantes | Outliers |
|-----------------|----------|
| `drop` - Remove linhas | `keep` - Mantém |
| `interpolate` - Interpola temporal | `clip` - Limita aos bounds |
| `ffill` - Propaga anterior | `remove` - Remove linha |
| `bfill` - Propaga posterior | `median` - Substitui pela mediana |

### Exemplo de Relatório

```
============================================================
📊 RELATÓRIO DE QUALIDADE - NVDA
============================================================

📅 Período: 2019-01-02 até 2025-12-05
📈 Total de registros: 1,489

--- Dados Faltantes ---
✅ Nenhum dado faltante encontrado

--- Duplicatas ---
✅ Nenhuma duplicata encontrada

--- Gaps Temporais ---
⚠️ 12 dias de negociação potencialmente faltando

--- Outliers ---
⚠️ Close: 3 outliers
⚠️ Volume: 7 outliers

--- Integridade dos Dados ---
✅ Dados íntegros (High >= Low, etc.)

------------------------------------------------------------
Status Final: ✅ VÁLIDO
============================================================
```

## 🧠 Modelo LSTM

### Arquitetura

```
Input (60 timesteps, N features)
    │
    ▼
LSTM (128 units, return_sequences=True) + L2 Regularization
BatchNormalization + Dropout(0.2)
    │
    ▼
LSTM (64 units, return_sequences=True) + L2 Regularization
BatchNormalization + Dropout(0.2)
    │
    ▼
LSTM (32 units, return_sequences=False) + L2 Regularization
BatchNormalization + Dropout(0.2)
    │
    ▼
Dense (32 units, ReLU)
Dropout(0.1)
    │
    ▼
Dense (16 units, ReLU)
    │
    ▼
Dense (1 unit, Linear) → Preço previsto
```

### Features Utilizadas (35+ indicadores)

**Features Básicas:**
- Open, High, Low, Close, Volume

**Features de Retorno:**
- Returns (retorno diário percentual)
- Log Returns (log-retorno)

**Médias Móveis:**
- MA_7, MA_21, MA_50 (Médias móveis simples)
- MA_7_21_Ratio (Razão entre MAs - tendência)
- Price_MA21_Ratio (Preço relativo à MA)

**Volatilidade:**
- Volatility_7, Volatility_21 (Desvio padrão dos retornos)
- ATR (Average True Range)
- Daily_Range (Range intraday normalizado)

**Indicadores de Momentum:**
- RSI (Relative Strength Index)
- MACD, MACD_Signal, MACD_Hist
- Momentum_7, Momentum_14
- ROC_7 (Rate of Change)
- Stoch_K, Stoch_D (Stochastic Oscillator)

**Bollinger Bands:**
- BB_Upper, BB_Lower (Bandas)
- BB_Width (Largura normalizada)
- BB_Position (Posição do preço 0-1)

**Volume:**
- Volume_MA_7 (Média móvel de volume)
- Volume_Ratio (Volume relativo à média)
- Volume_Change (Variação percentual)

**Features Temporais:**
- Day_of_Week (Dia da semana normalizado)
- Month (Mês normalizado)

## 🚀 Quick Start

### 1. Instalação

```bash
# Clone o projeto
cd stock_lstm_predictor

# Crie ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instale dependências
pip install -r requirements.txt
```

### 2. Treinamento

```bash
# Treinar modelo para NVIDIA (padrão)
python train.py

# Treinar para outra ação
python train.py --ticker AAPL

# Opções avançadas
python train.py --ticker GOOGL --epochs 150 --batch-size 64
```

### 3. Fazer Previsões (CLI)

```bash
# Prever próximo dia
python predict.py --ticker NVDA

# Prever próximos 5 dias
python predict.py --ticker NVDA --days 5
```

### 4. Iniciar API

```bash
python api.py
```

A API estará disponível em `http://localhost:8000`

## 📡 API Endpoints

### Health Check
```http
GET /health
```

### Prever Próximo Dia
```http
GET /predict/{ticker}
```

**Exemplo:**
```bash
curl http://localhost:8000/predict/NVDA
```

**Resposta:**
```json
{
  "ticker": "NVDA",
  "prediction_date": "2024-01-15",
  "last_close": 547.89,
  "last_close_date": "2024-01-12",
  "predicted_close": 552.34,
  "expected_change": 4.45,
  "expected_change_pct": 0.81,
  "direction": "UP",
  "generated_at": "2024-01-14T10:30:00"
}
```

### Prever Múltiplos Dias
```http
GET /predict/{ticker}/days/{n_days}
```

**Exemplo:**
```bash
curl http://localhost:8000/predict/NVDA/days/5
```

### Informações do Modelo
```http
GET /model/{ticker}
```

### Listar Modelos Disponíveis
```http
GET /models
```

## 📊 Métricas de Avaliação

| Métrica | Descrição |
|---------|-----------|
| MAE | Mean Absolute Error (erro médio em $) |
| RMSE | Root Mean Square Error |
| MAPE | Mean Absolute Percentage Error |
| R² | Coeficiente de determinação |
| Direction Accuracy | Precisão na direção do movimento |

## ⚙️ Configurações

Edite `config.py` para personalizar:

```python
# Dados
DEFAULT_TICKER = "NVDA"
DEFAULT_START_DATE = "2019-01-01"
SEQUENCE_LENGTH = 60  # Dias de histórico

# Modelo
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.2

# Treinamento
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15
```

## 📈 Exemplo de Uso em Python

```python
# Validar dados manualmente
from data_validator import validate_and_clean
from data_collector import StockDataCollector

collector = StockDataCollector("NVDA")
df = collector.fetch_historical_data()

# Validar e limpar
df_clean, report = validate_and_clean(df, ticker="NVDA")
print(report.summary())

# Fazer previsões
from predict import StockPredictor

predictor = StockPredictor("NVDA")
result = predictor.predict_next_day()

print(f"Preço previsto: ${result['predicted_close']:.2f}")
print(f"Direção: {result['direction']}")
```

## 🐳 Docker (Opcional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "api.py"]
```

```bash
docker build -t stock-predictor .
docker run -p 8000:8000 stock-predictor
```

## ⚠️ Disclaimer

**IMPORTANTE**: Este projeto é apenas para fins educacionais e de demonstração técnica.

- Não constitui recomendação de investimento
- O mercado de ações é inerentemente imprevisível
- Performance passada não garante resultados futuros
- Use por sua própria conta e risco

## 📚 Tecnologias

| Tecnologia | Uso |
|------------|-----|
| **Python 3.10+** | Linguagem principal |
| **TensorFlow/Keras** | Deep Learning / LSTM |
| **FastAPI** | API REST |
| **yfinance** | Coleta de dados de mercado |
| **scikit-learn** | Pré-processamento e métricas |
| **Pandas/NumPy** | Manipulação de dados |
| **Pydantic** | Validação de schemas da API |
| **Uvicorn** | Servidor ASGI |

## 📄 Licença

MIT License

---

Desenvolvido para demonstrar o pipeline completo de Deep Learning aplicado a séries temporais financeiras: desde a coleta e validação de dados até o deploy em produção.