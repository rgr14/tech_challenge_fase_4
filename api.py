"""
API FastAPI para previsão de preços de ações com LSTM
"""
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import os

from config import DEFAULT_TICKER, API_HOST, API_PORT, MODELS_DIR, SEQUENCE_LENGTH, FEATURES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================
# StockPredictor (inline)
# ==========================
# Mantemos o predictor dentro deste arquivo para evitar erros de import
# quando o módulo `predict.py` não estiver presente/atualizado.
import numpy as np  # noqa: F401

from data_collector import StockDataCollector
from preprocessor import StockDataPreprocessor
from model import StockLSTMModel


class StockPredictor:
    """Classe para fazer previsões de ações."""

    def __init__(self, ticker: str = DEFAULT_TICKER):
        self.ticker = ticker.upper()
        self.model = StockLSTMModel()
        self.preprocessor = StockDataPreprocessor(
            features=FEATURES,
            validate_data=False,  # Desabilitado por padrão para previsões
            auto_clean=False
        )
        self.collector = StockDataCollector(self.ticker)
        self._is_loaded = False

    def load(self) -> None:
        """Carrega modelo e scalers treinados."""
        try:
            self.model.load_model(self.ticker)
            self.preprocessor.load_scalers(self.ticker)
            self._is_loaded = True
            logger.info(f"Modelo e scalers carregados para {self.ticker}")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Modelo não encontrado para {self.ticker}. "
                "Execute train.py primeiro."
            ) from e

    def predict_next_day(self) -> dict:
        """Prevê o preço de fechamento do próximo dia útil."""
        if not self._is_loaded:
            self.load()

        logger.info("Coletando dados recentes...")
        df = self.collector.get_latest_data(days=SEQUENCE_LENGTH + 30)

        self.preprocessor.validate_data = False
        df_features = self.preprocessor.prepare_features(df, ticker=self.ticker)

        if len(df_features) < SEQUENCE_LENGTH:
            raise ValueError(
                f"Dados insuficientes. Necessário: {SEQUENCE_LENGTH}, "
                f"Disponível: {len(df_features)}"
            )

        X = self.preprocessor.prepare_prediction_data(df_features)
        prediction_scaled = self.model.predict(X)
        prediction = self.preprocessor.inverse_transform_predictions(prediction_scaled)

        last_close = float(df_features['Close'].iloc[-1])
        predicted_price = float(prediction[0][0])
        change = predicted_price - last_close
        change_pct = (change / last_close) * 100

        last_date = df_features.index[-1]
        next_date = self._get_next_business_day(last_date)

        return {
            "ticker": self.ticker,
            "prediction_date": next_date.strftime("%Y-%m-%d"),
            "last_close": round(last_close, 2),
            "last_close_date": last_date.strftime("%Y-%m-%d"),
            "predicted_close": round(predicted_price, 2),
            "expected_change": round(change, 2),
            "expected_change_pct": round(change_pct, 2),
            "generated_at": datetime.now().isoformat()
        }

    def predict_n_days(self, n_days: int = 5) -> list:
        """Prevê os próximos N dias (previsão iterativa)."""
        if not self._is_loaded:
            self.load()

        df = self.collector.get_latest_data(days=SEQUENCE_LENGTH + 30)

        self.preprocessor.validate_data = False
        df_features = self.preprocessor.prepare_features(df, ticker=self.ticker)

        predictions = []
        current_features = df_features.copy()
        last_date = current_features.index[-1]

        for i in range(n_days):
            X = self.preprocessor.prepare_prediction_data(current_features)
            pred_scaled = self.model.predict(X)
            pred = self.preprocessor.inverse_transform_predictions(pred_scaled)[0][0]

            next_date = self._get_next_business_day(last_date)

            predictions.append({
                "day": i + 1,
                "date": next_date.strftime("%Y-%m-%d"),
                "predicted_close": round(float(pred), 2)
            })

            # Atualizar para próxima previsão (heurística simples)
            new_row = current_features.iloc[-1].copy()
            new_row['Close'] = pred
            new_row['Open'] = current_features['Close'].iloc[-1]
            new_row['High'] = max(pred, new_row['Open']) * 1.01
            new_row['Low'] = min(pred, new_row['Open']) * 0.99
            new_row.name = next_date

            current_features = self.preprocessor.prepare_features(
                current_features._append(new_row),
                ticker=self.ticker
            )
            last_date = next_date

        return predictions

    def _get_next_business_day(self, date) -> datetime:
        """Retorna o próximo dia útil."""
        next_day = date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day

    def get_model_info(self) -> dict:
        """Retorna informações sobre o modelo carregado."""
        if not self._is_loaded:
            self.load()

        stock_info = self.collector.get_stock_info()

        return {
            "ticker": self.ticker,
            "stock_info": stock_info,
            "model_loaded": True,
            "sequence_length": getattr(self.preprocessor, "sequence_length", SEQUENCE_LENGTH),
            "features_used": getattr(self.preprocessor, "fitted_features", FEATURES)
        }

# Inicializar FastAPI
app = FastAPI(
    title="Stock Price Predictor API",
    description="""
    API para previsão de preços de ações usando redes neurais LSTM.
    
    ## Funcionalidades
    
    * **Previsão de próximo dia**: Prevê o preço de fechamento do próximo dia útil
    * **Previsão de múltiplos dias**: Prevê os próximos N dias (com incerteza crescente)
    
    ## Disclaimer
    
    ⚠️ Esta API é apenas para fins educacionais e não constitui recomendação de investimento.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache de preditores
predictors = {}


# Modelos Pydantic
class PredictionResponse(BaseModel):
    """Resposta de previsão de um dia."""
    ticker: str = Field(..., description="Símbolo da ação")
    prediction_date: str = Field(..., description="Data da previsão")
    last_close: float = Field(..., description="Último preço de fechamento")
    last_close_date: str = Field(..., description="Data do último fechamento")
    predicted_close: float = Field(..., description="Preço previsto")
    expected_change: float = Field(..., description="Variação esperada em $")
    expected_change_pct: float = Field(..., description="Variação esperada em %")
    generated_at: str = Field(..., description="Timestamp da previsão")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class MultiDayPrediction(BaseModel):
    """Previsão para um dia específico."""
    day: int = Field(..., description="Número do dia (1 = amanhã)")
    date: str = Field(..., description="Data da previsão")
    predicted_close: float = Field(..., description="Preço previsto")


class MultiDayPredictionResponse(BaseModel):
    """Resposta de previsão de múltiplos dias."""
    ticker: str
    predictions: List[MultiDayPrediction]
    warning: str = Field(
        default="Previsões mais distantes têm maior incerteza",
        description="Aviso sobre incerteza"
    )
    generated_at: str


class ModelInfoResponse(BaseModel):
    """Informações do modelo."""
    ticker: str
    stock_info: dict
    model_loaded: bool
    sequence_length: int
    features_used: List[str]


class HealthResponse(BaseModel):
    """Resposta de health check."""
    status: str
    timestamp: str
    available_models: List[str]


class ErrorResponse(BaseModel):
    """Resposta de erro."""
    error: str
    detail: str


def get_predictor(ticker: str) -> StockPredictor:
    """Obtém ou cria um predictor para o ticker."""
    ticker = ticker.upper()
    
    if ticker not in predictors:
        try:
            predictor = StockPredictor(ticker)
            predictor.load()
            predictors[ticker] = predictor
            logger.info(f"Predictor carregado para {ticker}")
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Modelo não encontrado para {ticker}. Execute o treinamento primeiro."
            )
        except Exception as e:
            logger.error(f"Erro ao carregar predictor: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao carregar modelo: {str(e)}"
            )
    
    return predictors[ticker]


def get_available_models() -> List[str]:
    """Lista modelos disponíveis."""
    models = []
    if MODELS_DIR.exists():
        for file in MODELS_DIR.glob("*_lstm_model.keras"):
            ticker = file.stem.replace("_lstm_model", "")
            models.append(ticker)
    return models


# Endpoints
@app.get(
    "/",
    summary="Página inicial",
    response_class=JSONResponse
)
async def root():
    """Página inicial da API."""
    return {
        "name": "Stock Price Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "/predict/{ticker}",
            "predict_days": "/predict/{ticker}/days/{n_days}",
            "model_info": "/model/{ticker}",
            "health": "/health"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check"
)
async def health_check():
    """Verifica status da API e modelos disponíveis."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        available_models=get_available_models()
    )


@app.get(
    "/predict/{ticker}",
    response_model=PredictionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Modelo não encontrado"},
        500: {"model": ErrorResponse, "description": "Erro interno"}
    },
    summary="Prever próximo dia"
)
async def predict_next_day(
    ticker: str = Path(
        ..., 
        description="Símbolo da ação (ex: NVDA, AAPL, GOOGL)",
        min_length=1,
        max_length=10
    )
):
    """
    Prevê o preço de fechamento do próximo dia útil.
    
    - **ticker**: Símbolo da ação (deve ter modelo treinado)
    
    Retorna previsão com variação esperada e direção.
    """
    try:
        predictor = get_predictor(ticker)
        result = predictor.predict_next_day()
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na previsão: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao fazer previsão: {str(e)}"
        )


@app.get(
    "/predict/{ticker}/days/{n_days}",
    response_model=MultiDayPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Parâmetros inválidos"},
        404: {"model": ErrorResponse, "description": "Modelo não encontrado"},
        500: {"model": ErrorResponse, "description": "Erro interno"}
    },
    summary="Prever múltiplos dias"
)
async def predict_multiple_days(
    ticker: str = Path(
        ..., 
        description="Símbolo da ação (ex: NVDA, AAPL, GOOGL)",
        min_length=1,
        max_length=10
    ),
    n_days: int = Path(
        ..., 
        ge=1,
        le=30,
        description="Número de dias para prever (1-30)"
    )
):
    """
    Prevê os preços de fechamento dos próximos N dias úteis.
    
    - **ticker**: Símbolo da ação
    - **n_days**: Número de dias (1-30)
    
    ⚠️ **Atenção**: Previsões mais distantes têm incerteza significativamente maior.
    """
    try:
        predictor = get_predictor(ticker)
        predictions = predictor.predict_n_days(n_days)
        
        return MultiDayPredictionResponse(
            ticker=ticker.upper(),
            predictions=[MultiDayPrediction(**p) for p in predictions],
            generated_at=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na previsão multi-dia: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao fazer previsão: {str(e)}"
        )


# @app.get(
#     "/model/{ticker}",
#     response_model=ModelInfoResponse,
#     responses={
#         404: {"model": ErrorResponse, "description": "Modelo não encontrado"}
#     },
#     summary="Informações do modelo"
# )
# async def get_model_info(ticker: str):
#     """
#     Retorna informações sobre o modelo treinado.
    
#     - **ticker**: Símbolo da ação
    
#     Inclui informações da ação, features usadas e configuração do modelo.
#     """
#     try:
#         predictor = get_predictor(ticker)
#         info = predictor.get_model_info()
#         return ModelInfoResponse(**info)
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Erro ao obter info do modelo: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Erro ao obter informações: {str(e)}"
#         )


# @app.get(
#     "/models",
#     summary="Listar modelos disponíveis"
# )
# async def list_models():
#     """Lista todos os modelos treinados disponíveis."""
#     models = get_available_models()
#     return {
#         "available_models": models,
#         "count": len(models)
#     }


# Eventos de startup/shutdown
@app.on_event("startup")
async def startup_event():
    """Evento de inicialização."""
    logger.info("🚀 Stock Predictor API iniciando...")
    models = get_available_models()
    logger.info(f"📊 Modelos disponíveis: {models}")


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de encerramento."""
    logger.info("👋 Stock Predictor API encerrando...")
    predictors.clear()


def main():
    """Inicia o servidor."""
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🚀 STOCK PREDICTOR API")
    print("=" * 60)
    print(f"\n📍 Servidor: http://{API_HOST}:{API_PORT}")
    print(f"📚 Documentação: http://localhost:{API_PORT}/docs")
    print(f"📖 ReDoc: http://localhost:{API_PORT}/redoc")
    print("\n" + "=" * 60)
    
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
