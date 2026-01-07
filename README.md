# Stock LSTM Predictor

Sistema de previsao de precos de acoes utilizando redes neurais **Long Short-Term Memory (LSTM)**, com validacao de dados e API REST para deploy.

## Objetivo

Prever o preco de fechamento de acoes utilizando dados historicos e Deep Learning, disponibilizando as previsoes atraves de uma API RESTful.

## Arquitetura do Projeto

```
stock_lstm_predictor/
├── config.py              # Configuracoes centralizadas (parametros do modelo vencedor)
├── data_collector.py      # Coleta de dados via Yahoo Finance
├── data_validator.py      # Validacao e limpeza de dados
├── preprocessor.py        # Normalizacao e criacao de sequencias
├── model.py               # Arquitetura do modelo LSTM
├── train.py               # Pipeline de treinamento
├── predict.py             # Script de previsao (CLI)
├── api.py                 # FastAPI para deploy
├── requirements.txt       # Dependencias
├── data/                  # Dados historicos (gerado)
├── models/                # Modelos treinados (gerado)
└── scalers/               # Scalers salvos (gerado)
```

## Pipeline de Dados

```
┌─────────────────────────────────────────────────────────────┐
│                    1. COLETA DE DADOS                       │
│                    (data_collector.py)                      │
│         Yahoo Finance API → Dados OHLCV historicos          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 2. VALIDACAO DE DADOS                       │
│                   (data_validator.py)                       │
│  • Dados faltantes e duplicatas                             │
│  • Gaps temporais e outliers                                │
│  • Integridade dos dados (High>=Low, etc)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 3. PRE-PROCESSAMENTO                        │
│                    (preprocessor.py)                        │
│  • Extracao da feature Close (modo univariado)              │
│  • Normalizacao MinMaxScaler (0-1)                          │
│  • Criacao de sequencias (30 dias → 1 previsao)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    4. MODELO LSTM                           │
│                      (model.py)                             │
│             2 camadas LSTM + Dense (MSE Loss)               │
└─────────────────────────────────────────────────────────────┘
```

## Modelo LSTM

### Arquitetura (Replica do Modelo Vencedor)

O modelo utiliza os parametros otimizados via Grid Search:

```
Input (30 timesteps, 1 feature - Close)
    │
    ▼
LSTM (100 units, return_sequences=True)
Dropout (0.2)
    │
    ▼
LSTM (100 units, return_sequences=False)
Dropout (0.2)
    │
    ▼
Dense (1 unit, Linear) → Preco previsto
```

### Parametros do Grid Search Vencedor

| Parametro | Valor |
|-----------|-------|
| Sequence Length | 30 dias |
| LSTM Units | 100 |
| Dropout Rate | 0.2 |
| Loss Function | Mean Squared Error |
| Epochs | 100 |
| Batch Size | 32 |
| Train/Test Split | 80/20 |

### Modo de Operacao

O modelo opera em **modo univariado**, utilizando apenas o preco de fechamento (`Close`) como feature. Isso replica o comportamento do modelo original que obteve os melhores resultados.

## Quick Start

### 1. Instalacao

```bash
# Crie ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instale dependencias
pip install -r requirements.txt
```

### 2. Treinamento

```bash
# Treinar modelo para NVIDIA (padrao)
python train.py

# Treinar para outra acao
python train.py --ticker AAPL
```

### 2.1 Grid Search (Opcional - Recomendado)

O script suporta **Grid Search com TimeSeriesSplit** para encontrar os melhores hiperparametros:

```bash
# Executar Grid Search completo
python train.py --grid-search --ticker NVDA

# Apos o Grid Search, treinar com os melhores parametros
python train.py --train-best --ticker NVDA
```

#### Parametros do Grid Search

| Parametro | Valores Testados |
|-----------|------------------|
| sequence_length | 15, 30, 60 |
| lstm_units | 50, 100, 150 |
| dropout_rate | 0.1, 0.2, 0.3 |
| batch_size | 16, 32, 64 |
| epochs | 50, 100 |
| learning_rate | 0.001, 0.0001 |
| n_layers | 1, 2, 3 |

#### Opcoes do Grid Search

```bash
# Personalizar numero de splits para TimeSeriesSplit
python train.py --grid-search --splits 5

# Personalizar data inicial
python train.py --grid-search --start-date 2020-01-01
```

#### Arquivos Gerados

Apos o Grid Search, os seguintes arquivos sao gerados na pasta `models/`:

| Arquivo | Descricao |
|---------|-----------|
| `{ticker}_grid_search_results.csv` | Resultados de todas as combinacoes |
| `{ticker}_best_params.json` | Melhores parametros encontrados |
| `{ticker}_grid_search_analysis.png` | Graficos de analise |
| `{ticker}_final_params.json` | Parametros + metricas do modelo final |

### 3. Fazer Previsoes (CLI)

```bash
# Prever proximo dia
python predict.py
```

O script exibe:
- Preco atual
- Preco previsto
- Variacao esperada ($ e %)
- Tendencia (SUBIR/CAIR)

### 4. Iniciar API

```bash
python api.py
```

A API estara disponivel em `http://localhost:8000`

## API Endpoints

### Health Check
```http
GET /health
```
Retorna status da API e modelos disponiveis.

### Prever Proximo Dia
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
  "generated_at": "2024-01-14T10:30:00"
}
```

### Prever Multiplos Dias
```http
GET /predict/{ticker}/days/{n_days}
```

**Exemplo:**
```bash
curl http://localhost:8000/predict/NVDA/days/5
```

**Nota:** Previsoes mais distantes tem incerteza significativamente maior.

### Documentacao Interativa
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Metricas de Avaliacao

| Metrica | Descricao |
|---------|-----------|
| MAE | Mean Absolute Error (erro medio em $) |
| RMSE | Root Mean Square Error |
| MAPE | Mean Absolute Percentage Error |
| R² | Coeficiente de determinacao |
| Direction Accuracy | Precisao na direcao do movimento |

## Configuracoes

Edite `config.py` para personalizar:

```python
# Dados
DEFAULT_TICKER = "NVDA"
DEFAULT_START_DATE = "2019-01-01"
SEQUENCE_LENGTH = 30  # Dias de historico

# Modelo
LSTM_UNITS_1 = 100
LSTM_UNITS_2 = 100
DROPOUT_RATE = 0.2

# Treinamento
EPOCHS = 100
BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.8

# Modo Univariado
FEATURES = ["Close"]
TARGET = "Close"
```

## Validacao de Dados

O modulo `data_validator.py` garante a qualidade dos dados antes do treinamento:

### Verificacoes Realizadas

| Verificacao | Descricao |
|-------------|-----------|
| Dados Faltantes | Conta NaN por coluna, calcula percentual |
| Duplicatas | Detecta datas e linhas duplicadas |
| Gaps Temporais | Identifica dias de negociacao faltando |
| Outliers | Detecta via IQR, Z-Score ou MAD |
| Integridade | Valida High >= Low, precos > 0, Volume >= 0 |
| Valores Infinitos | Detecta `inf` e `-inf` |
| Variacoes Extremas | Alerta variacoes > 50% em um dia |

## Tecnologias

| Tecnologia | Uso |
|------------|-----|
| Python 3.10+ | Linguagem principal |
| TensorFlow/Keras | Deep Learning / LSTM |
| FastAPI | API REST |
| yfinance | Coleta de dados de mercado |
| scikit-learn | Pre-processamento e metricas |
| Pandas/NumPy | Manipulacao de dados |
| Pydantic | Validacao de schemas da API |
| Uvicorn | Servidor ASGI |

## Disclaimer

**IMPORTANTE**: Este projeto e apenas para fins educacionais e de demonstracao tecnica.

- Nao constitui recomendacao de investimento
- O mercado de acoes e inerentemente imprevisivel
- Performance passada nao garante resultados futuros
- Use por sua propria conta e risco

## Licenca

MIT License

---

Desenvolvido para o Tech Challenge FIAP - Demonstracao de pipeline de Deep Learning aplicado a series temporais financeiras.
