"""
Módulo de coleta de dados de ações via Yahoo Finance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

from config import DATA_DIR, DEFAULT_TICKER, DEFAULT_START_DATE, DEFAULT_END_DATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """Classe para coleta e gerenciamento de dados de ações."""
    
    def __init__(self, ticker: str = DEFAULT_TICKER):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.data: Optional[pd.DataFrame] = None
    
    def fetch_historical_data(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Busca dados históricos da ação.
        
        Args:
            start_date: Data inicial (YYYY-MM-DD)
            end_date: Data final (YYYY-MM-DD), None para hoje
            interval: Intervalo dos dados (1d, 1wk, 1mo)
        
        Returns:
            DataFrame com dados históricos
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Buscando dados de {self.ticker} de {start_date} até {end_date}")
        
        self.data = self.stock.history(
            start=start_date,
            end=end_date,
            interval=interval
        )
        
        if self.data.empty:
            raise ValueError(f"Nenhum dado encontrado para {self.ticker}")
        
        # Limpar e preparar dados
        self.data = self._clean_data(self.data)
        
        logger.info(f"Coletados {len(self.data)} registros")
        return self.data
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e prepara os dados."""
        # Remover colunas desnecessárias se existirem
        cols_to_drop = ["Dividends", "Stock Splits"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # Garantir que o índice é datetime
        df.index = pd.to_datetime(df.index)
        
        # Remover timezone se existir
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Remover NaN
        df = df.dropna()
        
        # Ordenar por data
        df = df.sort_index()
        
        return df
    
    def get_latest_data(self, days: int = 60) -> pd.DataFrame:
        """
        Busca os últimos N dias de dados.
        
        Args:
            days: Número de dias para buscar
        
        Returns:
            DataFrame com dados recentes
        """
        end_date = datetime.now()
        # Adicionar margem para fins de semana/feriados
        start_date = end_date - timedelta(days=days + 30)
        
        data = self.fetch_historical_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        # Retornar apenas os últimos N dias úteis
        return data.tail(days)
    
    def save_data(self, filename: Optional[str] = None) -> str:
        """Salva os dados em CSV."""
        if self.data is None:
            raise ValueError("Nenhum dado para salvar. Execute fetch_historical_data primeiro.")
        
        if filename is None:
            filename = f"{self.ticker}_historical.csv"
        
        filepath = DATA_DIR / filename
        self.data.to_csv(filepath)
        logger.info(f"Dados salvos em {filepath}")
        return str(filepath)
    
    def load_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """Carrega dados de um CSV."""
        if filename is None:
            filename = f"{self.ticker}_historical.csv"
        
        filepath = DATA_DIR / filename
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Dados carregados de {filepath}")
        return self.data
    
    def get_stock_info(self) -> dict:
        """Retorna informações básicas da ação."""
        info = self.stock.info
        return {
            "ticker": self.ticker,
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
        }


def main():
    """Exemplo de uso do coletor de dados."""
    collector = StockDataCollector("NVDA")
    
    # Buscar dados históricos
    data = collector.fetch_historical_data()
    print(f"\nDados coletados: {len(data)} registros")
    print(f"\nPrimeiros registros:\n{data.head()}")
    print(f"\nÚltimos registros:\n{data.tail()}")
    print(f"\nEstatísticas:\n{data.describe()}")
    
    # Informações da ação
    info = collector.get_stock_info()
    print(f"\nInformações da ação: {info}")
    
    # Salvar dados
    collector.save_data()


if __name__ == "__main__":
    main()
