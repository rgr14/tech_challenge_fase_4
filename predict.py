"""
Script para fazer previsões com modelo LSTM treinado
"""
import argparse
import logging
from datetime import datetime, timedelta
from typing import Optional
import numpy as np

from config import DEFAULT_TICKER, SEQUENCE_LENGTH, FEATURES
from data_collector import StockDataCollector
from preprocessor import StockDataPreprocessor
from model import StockLSTMModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        self.collector = StockDataCollector(ticker)
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
        """
        Prevê o preço de fechamento do próximo dia útil.
        
        Returns:
            Dicionário com previsão e informações
        """
        if not self._is_loaded:
            self.load()
        
        # Coletar dados recentes
        logger.info("Coletando dados recentes...")
        df = self.collector.get_latest_data(days=SEQUENCE_LENGTH + 30)
        
        # Preparar features (com validação desabilitada para previsão rápida)
        self.preprocessor.validate_data = False  # Já foi validado no treino
        df_features = self.preprocessor.prepare_features(df, ticker=self.ticker)
        
        # Verificar se temos dados suficientes
        if len(df_features) < SEQUENCE_LENGTH:
            raise ValueError(
                f"Dados insuficientes. Necessário: {SEQUENCE_LENGTH}, "
                f"Disponível: {len(df_features)}"
            )
        
        # Preparar dados para previsão
        X = self.preprocessor.prepare_prediction_data(df_features)
        
        # Fazer previsão
        prediction_scaled = self.model.predict(X)
        prediction = self.preprocessor.inverse_transform_predictions(prediction_scaled)
        
        # Informações adicionais
        last_close = float(df_features['Close'].iloc[-1])
        predicted_price = float(prediction[0][0])
        change = predicted_price - last_close
        change_pct = (change / last_close) * 100
        
        # Data da previsão (próximo dia útil)
        last_date = df_features.index[-1]
        next_date = self._get_next_business_day(last_date)
        
        result = {
            "ticker": self.ticker,
            "prediction_date": next_date.strftime("%Y-%m-%d"),
            "last_close": round(last_close, 2),
            "last_close_date": last_date.strftime("%Y-%m-%d"),
            "predicted_close": round(predicted_price, 2),
            "expected_change": round(change, 2),
            "expected_change_pct": round(change_pct, 2),
            "direction": "UP" if change > 0 else "DOWN",
            "generated_at": datetime.now().isoformat()
        }
        
        return result
    
    def predict_n_days(self, n_days: int = 5) -> list:
        """
        Prevê os próximos N dias (previsão iterativa).
        
        NOTA: Previsões mais distantes têm incerteza crescente.
        
        Args:
            n_days: Número de dias para prever
        
        Returns:
            Lista de previsões
        """
        if not self._is_loaded:
            self.load()
        
        # Coletar dados recentes
        df = self.collector.get_latest_data(days=SEQUENCE_LENGTH + 30)
        
        # Desabilitar validação para previsão rápida
        self.preprocessor.validate_data = False
        df_features = self.preprocessor.prepare_features(df, ticker=self.ticker)
        
        predictions = []
        current_features = df_features.copy()
        last_date = current_features.index[-1]
        
        for i in range(n_days):
            # Preparar dados
            X = self.preprocessor.prepare_prediction_data(current_features)
            
            # Prever
            pred_scaled = self.model.predict(X)
            pred = self.preprocessor.inverse_transform_predictions(pred_scaled)[0][0]
            
            # Próximo dia útil
            next_date = self._get_next_business_day(last_date)
            
            predictions.append({
                "day": i + 1,
                "date": next_date.strftime("%Y-%m-%d"),
                "predicted_close": round(float(pred), 2),
                "uncertainty": "low" if i < 2 else "medium" if i < 4 else "high"
            })
            
            # Atualizar para próxima previsão (simplificado)
            # Na prática, precisaríamos de um modelo mais sofisticado
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
        
        # Pular fins de semana
        while next_day.weekday() >= 5:  # 5 = Sábado, 6 = Domingo
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
            "sequence_length": self.preprocessor.sequence_length,
            "features_used": self.preprocessor.fitted_features
        }


def main():
    """Entry point do script de previsão."""
    parser = argparse.ArgumentParser(
        description='Fazer previsões com modelo LSTM treinado'
    )
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        default=DEFAULT_TICKER,
        help=f'Símbolo da ação (default: {DEFAULT_TICKER})'
    )
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=1,
        help='Número de dias para prever (default: 1)'
    )
    
    args = parser.parse_args()
    
    predictor = StockPredictor(args.ticker)
    
    print("\n" + "=" * 60)
    print(f"PREVISÃO DE PREÇOS - {args.ticker}")
    print("=" * 60)
    
    if args.days == 1:
        result = predictor.predict_next_day()
        
        print(f"\nÚltimo fechamento: ${result['last_close']:.2f}")
        print(f"Data: {result['last_close_date']}")
        print(f"\n📈 Previsão para {result['prediction_date']}:")
        print(f"   Preço previsto: ${result['predicted_close']:.2f}")
        print(f"   Variação esperada: {result['expected_change']:+.2f} ({result['expected_change_pct']:+.2f}%)")
        print(f"   Direção: {result['direction']}")
    else:
        predictions = predictor.predict_n_days(args.days)
        
        print(f"\nPrevisões para os próximos {args.days} dias:")
        print("-" * 50)
        
        for pred in predictions:
            uncertainty_icon = "🟢" if pred['uncertainty'] == 'low' else "🟡" if pred['uncertainty'] == 'medium' else "🔴"
            print(f"Dia {pred['day']} ({pred['date']}): ${pred['predicted_close']:.2f} {uncertainty_icon}")
        
        print("\n⚠️  AVISO: Previsões mais distantes têm maior incerteza.")
    
    print("\n" + "=" * 60)
    print("⚠️  DISCLAIMER: Estas previsões são apenas para fins educacionais.")
    print("    Não constitui recomendação de investimento.")
    print("=" * 60)


if __name__ == "__main__":
    main()