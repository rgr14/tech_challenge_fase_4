# Stock LSTM Predictor

Sistema de previsao de precos de acoes utilizando redes neurais **Long Short-Term Memory (LSTM)**, com validacao de dados e API REST para deploy.

## Objetivo

Prever o preco de fechamento de acoes utilizando dados historicos e Deep Learning, disponibilizando as previsoes atraves de uma API RESTful.

## Arquitetura do Projeto

```
stock_lstm_predictor/
├── main.py                    # Ponto de entrada principal (CLI)
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuracoes centralizadas
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py       # Coleta de dados via Yahoo Finance
│   │   ├── validator.py       # Validacao e limpeza de dados
│   │   └── preprocessor.py    # Normalizacao e criacao de sequencias
│   ├── models/
│   │   ├── __init__.py
│   │   └── lstm.py            # Arquitetura do modelo LSTM
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py             # FastAPI para deploy
│   └── cli/
│       ├── __init__.py
│       ├── train.py           # Pipeline de treinamento
│       └── predict.py         # Script de previsao
├── data/                      # Dados historicos (gerado)
├── artifacts/
│   ├── models/                # Modelos treinados (gerado)
│   └── scalers/               # Scalers salvos (gerado)
├── requirements.txt           # Dependencias
└── README.md
```

## Pipeline de Dados

```
┌─────────────────────────────────────────────────────────────┐
│                    1. COLETA DE DADOS                       │
│                  (src/data/collector.py)                    │
│         Yahoo Finance API → Dados OHLCV historicos          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 2. VALIDACAO DE DADOS                       │
│                 (src/data/validator.py)                     │
│  • Dados faltantes e duplicatas                             │
│  • Outliers                                │
│  • Integridade dos dados (High>=Low, etc)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 3. PRE-PROCESSAMENTO                        │
│                (src/data/preprocessor.py)                   │
│  • Extracao da feature Close (modo univariado)              │
│  • Normalizacao MinMaxScaler (0-1)                          │
│  • Criacao de sequencias (30 dias → 1 previsao)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    4. MODELO LSTM                           │
│                   (src/models/lstm.py)                      │
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

O modelo opera em **modo univariado**, utilizando apenas o preco de fechamento (`Close`) como feature.

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
python main.py train

# Treinar para outra acao
python main.py train --ticker AAPL
```

### 2.1 Grid Search (Opcional - Recomendado)

O script suporta **Grid Search com TimeSeriesSplit** para encontrar os melhores hiperparametros:

```bash
# Executar Grid Search completo
python main.py train --grid-search --ticker NVDA

# Apos o Grid Search, treinar com os melhores parametros
python main.py train --train-best --ticker NVDA
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
python main.py train --grid-search --splits 5

# Personalizar data inicial
python main.py train --grid-search --start-date 2020-01-01
```

#### Arquivos Gerados

Apos o Grid Search, os seguintes arquivos sao gerados na pasta `artifacts/models/`:

| Arquivo | Descricao |
|---------|-----------|
| `{ticker}_grid_search_results.csv` | Resultados de todas as combinacoes |
| `{ticker}_best_params.json` | Melhores parametros encontrados |
| `{ticker}_grid_search_analysis.png` | Graficos de analise |
| `{ticker}_final_params.json` | Parametros + metricas do modelo final |

### 3. Fazer Previsoes (CLI)

```bash
# Prever proximo dia
python main.py predict

# Prever para outra acao
python main.py predict --ticker AAPL
```

O script exibe:
- Preco atual
- Preco previsto
- Variacao esperada ($ e %)
- Tendencia (SUBIR/CAIR)

### 4. Iniciar API

```bash
python main.py api
```

A API estara disponivel em `http://localhost:8000`

## Comandos Disponiveis

```bash
# Ver ajuda geral
python main.py --help

# Ver ajuda do comando train
python main.py train --help

# Ver ajuda do comando predict
python main.py predict --help

# Ver ajuda do comando api
python main.py api --help
```

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

## Configuracoes

Edite `src/config.py` para personalizar:

```python
# Dados
DEFAULT_TICKER = "NVDA"
DEFAULT_START_DATE = "2022-01-01"
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

O modulo `src/data/validator.py` garante a qualidade dos dados antes do treinamento:

### Verificacoes Realizadas

| Verificacao | Descricao |
|-------------|-----------|
| Dados Faltantes | Conta NaN por coluna, calcula percentual |
| Duplicatas | Detecta datas e linhas duplicadas |
| Outliers | Detecta via IQR, Z-Score ou MAD |
| Integridade | Valida High >= Low, precos > 0, Volume >= 0 |
| Valores Infinitos | Detecta `inf` e `-inf` |
| Variacoes Extremas | Alerta variacoes > 50% em um dia |

## Analise Exploratoria (EDA)

O modulo `src/data/validator.py` tambem oferece funcoes para analise exploratoria de dados com foco em deteccao de outliers.

### Gerando Boxplots

**Via linha de comando:**
```bash
# Gerar boxplots para NVDA (salva em artifacts/models/)
python -m src.data.validator --ticker NVDA --boxplot

# Gerar e exibir na tela
python -m src.data.validator --ticker NVDA --boxplot --show

# Para outra acao
python -m src.data.validator --ticker AAPL --boxplot --show
```

**Via Python:**
```python
from src.data.collector import StockDataCollector
from src.data.validator import generate_boxplots, print_outlier_summary

# Coletar dados
collector = StockDataCollector("NVDA")
df = collector.fetch_historical_data()

# Gerar boxplots (salva em artifacts/models/NVDA_boxplots_eda.png)
generate_boxplots(df, ticker="NVDA")

# Imprimir resumo de outliers no terminal
print_outlier_summary(df, ticker="NVDA")
```

### Arquivos Gerados

| Arquivo | Descricao |
|---------|-----------|
| `artifacts/models/{ticker}_boxplots_eda.png` | Boxplots de todas as variaveis quantitativas |

### Variaveis Analisadas

| Variavel | Descricao |
|----------|-----------|
| Open | Preco de abertura |
| High | Preco maximo do dia |
| Low | Preco minimo do dia |
| Close | Preco de fechamento |
| Volume | Volume de negociacao |

O grafico separa as variaveis de preco (escala em USD) do Volume (escala em milhoes) para melhor visualizacao. Cada boxplot exibe a contagem de outliers detectados via metodo IQR (1.5 * IQR).

## Estrutura de Imports

Para usar os modulos em seu codigo:

```python
# Configuracoes
from src.config import DEFAULT_TICKER, MODELS_DIR, FEATURES

# Coleta de dados
from src.data.collector import StockDataCollector

# Validacao
from src.data.validator import DataValidator, validate_and_clean

# Pre-processamento
from src.data.preprocessor import StockDataPreprocessor

# Modelo
from src.models.lstm import StockLSTMModel, calculate_metrics

# API
from src.api.app import app
```

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
