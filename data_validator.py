"""
Módulo de Validação e Limpeza de Dados para Séries Temporais Financeiras
=========================================================================
Responsável por garantir a qualidade dos dados antes do treinamento do LSTM.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Relatório de qualidade dos dados."""
    ticker: str
    total_records: int
    date_range: Tuple[str, str]
    
    # Dados faltantes
    missing_values: Dict[str, int] = field(default_factory=dict)
    missing_percentage: Dict[str, float] = field(default_factory=dict)
    total_missing: int = 0
    
    # Duplicatas
    duplicate_dates: int = 0
    duplicate_rows: int = 0
    
    # Gaps temporais
    missing_trading_days: List[str] = field(default_factory=list)
    total_gaps: int = 0
    
    # Outliers
    outliers_detected: Dict[str, int] = field(default_factory=dict)
    total_outliers: int = 0
    
    # Integridade
    integrity_issues: List[str] = field(default_factory=list)
    
    # Status
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Gera resumo textual do relatório."""
        lines = [
            "\n" + "=" * 60,
            f"📊 RELATÓRIO DE QUALIDADE - {self.ticker}",
            "=" * 60,
            f"\n📅 Período: {self.date_range[0]} até {self.date_range[1]}",
            f"📈 Total de registros: {self.total_records:,}",
            "\n--- Dados Faltantes ---"
        ]
        
        if self.total_missing == 0:
            lines.append("✅ Nenhum dado faltante encontrado")
        else:
            for col, count in self.missing_values.items():
                if count > 0:
                    pct = self.missing_percentage[col]
                    lines.append(f"  ⚠️  {col}: {count} ({pct:.2f}%)")
        
        lines.append("\n--- Duplicatas ---")
        if self.duplicate_dates == 0 and self.duplicate_rows == 0:
            lines.append("✅ Nenhuma duplicata encontrada")
        else:
            if self.duplicate_dates > 0:
                lines.append(f"  ⚠️  Datas duplicadas: {self.duplicate_dates}")
            if self.duplicate_rows > 0:
                lines.append(f"  ⚠️  Linhas duplicadas: {self.duplicate_rows}")
        
        # lines.append("\n--- Gaps Temporais ---")
        # if self.total_gaps == 0:
        #     lines.append("✅ Série temporal contínua (sem gaps)")
        # else:
        #     lines.append(f"  ⚠️  {self.total_gaps} dias de negociação faltando")
        #     if len(self.missing_trading_days) <= 10:
        #         for day in self.missing_trading_days:
        #             lines.append(f"      - {day}")
        #     else:
        #         for day in self.missing_trading_days[:5]:
        #             lines.append(f"      - {day}")
        #         lines.append(f"      ... e mais {len(self.missing_trading_days) - 5} dias")
        
        lines.append("\n--- Outliers ---")
        if self.total_outliers == 0:
            lines.append("✅ Nenhum outlier extremo detectado")
        else:
            for col, count in self.outliers_detected.items():
                if count > 0:
                    lines.append(f"  ⚠️  {col}: {count} outliers")
        
        # lines.append("\n--- Integridade dos Dados ---")
        # if not self.integrity_issues:
        #     lines.append("✅ Dados íntegros (High >= Low, etc.)")
        # else:
        #     for issue in self.integrity_issues[:10]:
        #         lines.append(f"  ❌ {issue}")
        
        # lines.append("\n" + "-" * 60)
        # status = "✅ VÁLIDO" if self.is_valid else "❌ INVÁLIDO"
        # lines.append(f"Status Final: {status}")
        
        # if self.warnings:
        #     lines.append(f"\n⚠️  {len(self.warnings)} avisos")
        # if self.errors:
        #     lines.append(f"❌ {len(self.errors)} erros críticos")
        
        # lines.append("=" * 60)
        
        return "\n".join(lines)


class DataValidator:
    """Classe para validação de dados financeiros."""
    
    def __init__(
        self,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        max_missing_pct: float = 5.0,
        check_trading_days: bool = True
    ):
        """
        Args:
            outlier_method: Método de detecção ('iqr', 'zscore', 'mad')
            outlier_threshold: Limiar para detecção de outliers
            max_missing_pct: Percentual máximo aceitável de dados faltantes
            check_trading_days: Se deve verificar dias de negociação faltantes
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.max_missing_pct = max_missing_pct
        self.check_trading_days = check_trading_days
        self.report: Optional[DataQualityReport] = None
    
    def validate(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> DataQualityReport:
        """
        Executa validação completa dos dados.
        
        Args:
            df: DataFrame com dados OHLCV
            ticker: Símbolo da ação
        
        Returns:
            Relatório de qualidade
        """
        logger.info(f"Iniciando validação de dados para {ticker}...")
        
        df = df.copy()
        
        # Garantir índice datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Inicializar relatório
        self.report = DataQualityReport(
            ticker=ticker,
            total_records=len(df),
            date_range=(
                df.index.min().strftime("%Y-%m-%d"),
                df.index.max().strftime("%Y-%m-%d")
            )
        )
        
        # Executar validações
        self._check_missing_values(df)
        self._check_duplicates(df)
        self._check_temporal_gaps(df)
        self._check_outliers(df)
        self._check_data_integrity(df)
        
        # Determinar status final
        self._determine_final_status()
        
        logger.info(f"Validação concluída. Status: {'VÁLIDO' if self.report.is_valid else 'INVÁLIDO'}")
        
        return self.report
    
    def _check_missing_values(self, df: pd.DataFrame) -> None:
        """Verifica dados faltantes."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in required_columns:
            if col in df.columns:
                missing = df[col].isna().sum()
                pct = (missing / len(df)) * 100
                
                self.report.missing_values[col] = missing
                self.report.missing_percentage[col] = pct
                self.report.total_missing += missing
                
                if pct > self.max_missing_pct:
                    self.report.warnings.append(
                        f"Coluna {col} tem {pct:.2f}% de dados faltantes"
                    )
            else:
                self.report.errors.append(f"Coluna obrigatória '{col}' não encontrada")
        
        # Verificar NaN infinitos
        for col in df.select_dtypes(include=[np.number]).columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                self.report.warnings.append(f"Coluna {col} contém {inf_count} valores infinitos")
    
    def _check_duplicates(self, df: pd.DataFrame) -> None:
        """Verifica duplicatas."""
        # Datas duplicadas
        self.report.duplicate_dates = df.index.duplicated().sum()
        
        # Linhas completamente duplicadas
        self.report.duplicate_rows = df.duplicated().sum()
        
        if self.report.duplicate_dates > 0:
            self.report.warnings.append(
                f"{self.report.duplicate_dates} datas duplicadas encontradas"
            )
    
    def _check_temporal_gaps(self, df: pd.DataFrame) -> None:
        """Verifica gaps na série temporal."""
        if not self.check_trading_days:
            return
        
        df_sorted = df.sort_index()
        
        # Gerar range de dias úteis esperados
        date_range = pd.date_range(
            start=df_sorted.index.min(),
            end=df_sorted.index.max(),
            freq='B'  # Business days
        )
        
        # Encontrar dias faltantes
        existing_dates = set(df_sorted.index.date)
        expected_dates = set(date_range.date)
        
        # Remover feriados conhecidos (simplificado - apenas fins de semana já são tratados)
        missing_dates = expected_dates - existing_dates
        
        # Filtrar apenas dias úteis realmente faltantes (tolerância para feriados)
        # Consideramos aceitável até 30 dias de "feriados" por ano
        if len(missing_dates) > 0:
            self.report.missing_trading_days = sorted([
                d.strftime("%Y-%m-%d") for d in missing_dates
            ])
            self.report.total_gaps = len(missing_dates)
            
            # Só avisar se muitos gaps
            years = (df_sorted.index.max() - df_sorted.index.min()).days / 365
            expected_holidays = int(years * 15)  # ~15 feriados por ano
            
            if self.report.total_gaps > expected_holidays:
                self.report.warnings.append(
                    f"{self.report.total_gaps} dias de negociação potencialmente faltando"
                )
    
    def _check_outliers(self, df: pd.DataFrame) -> None:
        """Detecta outliers nos dados."""
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            data = df[col].dropna()
            
            if self.outlier_method == "iqr":
                outliers = self._detect_outliers_iqr(data)
            elif self.outlier_method == "zscore":
                outliers = self._detect_outliers_zscore(data)
            elif self.outlier_method == "mad":
                outliers = self._detect_outliers_mad(data)
            else:
                outliers = pd.Series([False] * len(data))
            
            count = outliers.sum()
            self.report.outliers_detected[col] = count
            self.report.total_outliers += count
            
            if count > 0:
                pct = (count / len(data)) * 100
                if pct > 1:  # Mais de 1% de outliers
                    self.report.warnings.append(
                        f"Coluna {col} tem {count} outliers ({pct:.2f}%)"
                    )
    
    def _detect_outliers_iqr(self, data: pd.Series) -> pd.Series:
        """Detecta outliers usando IQR (Interquartile Range)."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - (self.outlier_threshold * IQR)
        upper = Q3 + (self.outlier_threshold * IQR)
        
        return (data < lower) | (data > upper)
    
    def _detect_outliers_zscore(self, data: pd.Series) -> pd.Series:
        """Detecta outliers usando Z-Score."""
        mean = data.mean()
        std = data.std()
        
        if std == 0:
            return pd.Series([False] * len(data))
        
        z_scores = np.abs((data - mean) / std)
        return z_scores > self.outlier_threshold
    
    def _detect_outliers_mad(self, data: pd.Series) -> pd.Series:
        """Detecta outliers usando MAD (Median Absolute Deviation)."""
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return pd.Series([False] * len(data))
        
        modified_z = 0.6745 * (data - median) / mad
        return np.abs(modified_z) > self.outlier_threshold
    
    def _check_data_integrity(self, df: pd.DataFrame) -> None:
        """Verifica integridade lógica dos dados OHLCV."""
        issues = []
        
        # High deve ser >= Low
        if 'High' in df.columns and 'Low' in df.columns:
            invalid = df[df['High'] < df['Low']]
            if len(invalid) > 0:
                issues.append(f"{len(invalid)} registros com High < Low")
                for idx in invalid.index[:5]:
                    issues.append(
                        f"  {idx.strftime('%Y-%m-%d')}: High={invalid.loc[idx, 'High']:.2f}, "
                        f"Low={invalid.loc[idx, 'Low']:.2f}"
                    )
        
        # High deve ser >= Open e Close
        if all(col in df.columns for col in ['High', 'Open', 'Close']):
            invalid_open = df[df['High'] < df['Open']]
            invalid_close = df[df['High'] < df['Close']]
            
            if len(invalid_open) > 0:
                issues.append(f"{len(invalid_open)} registros com High < Open")
            if len(invalid_close) > 0:
                issues.append(f"{len(invalid_close)} registros com High < Close")
        
        # Low deve ser <= Open e Close
        if all(col in df.columns for col in ['Low', 'Open', 'Close']):
            invalid_open = df[df['Low'] > df['Open']]
            invalid_close = df[df['Low'] > df['Close']]
            
            if len(invalid_open) > 0:
                issues.append(f"{len(invalid_open)} registros com Low > Open")
            if len(invalid_close) > 0:
                issues.append(f"{len(invalid_close)} registros com Low > Close")
        
        # Volume deve ser >= 0
        if 'Volume' in df.columns:
            invalid = df[df['Volume'] < 0]
            if len(invalid) > 0:
                issues.append(f"{len(invalid)} registros com Volume negativo")
        
        # Preços devem ser positivos
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                invalid = df[df[col] <= 0]
                if len(invalid) > 0:
                    issues.append(f"{len(invalid)} registros com {col} <= 0")
        
        # Variações extremas (> 50% em um dia)
        if 'Close' in df.columns:
            returns = df['Close'].pct_change().abs()
            extreme = returns[returns > 0.5]
            if len(extreme) > 0:
                issues.append(f"{len(extreme)} dias com variação > 50%")
                for idx in extreme.index[:3]:
                    issues.append(
                        f"  {idx.strftime('%Y-%m-%d')}: {extreme.loc[idx]*100:.1f}%"
                    )
        
        self.report.integrity_issues = issues
        
        if issues:
            self.report.errors.extend([
                issue for issue in issues 
                if not issue.startswith("  ")
            ])
    
    def _determine_final_status(self) -> None:
        """Determina status final da validação."""
        # Erros críticos invalidam os dados
        if self.report.errors:
            self.report.is_valid = False
            return
        
        # Muitos dados faltantes invalidam
        for col, pct in self.report.missing_percentage.items():
            if pct > 10:  # Mais de 10% faltando
                self.report.is_valid = False
                return
        
        # Problemas de integridade graves
        if len(self.report.integrity_issues) > 10:
            self.report.is_valid = False
            return


class DataCleaner:
    """Classe para limpeza e tratamento de dados."""
    
    def __init__(
        self,
        handle_missing: str = "interpolate",
        handle_outliers: str = "clip",
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0
    ):
        """
        Args:
            handle_missing: Método para dados faltantes ('drop', 'interpolate', 'ffill', 'bfill')
            handle_outliers: Método para outliers ('keep', 'clip', 'remove', 'median')
            outlier_method: Método de detecção de outliers
            outlier_threshold: Limiar para outliers
        """
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.cleaning_log: List[str] = []
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executa limpeza completa dos dados.
        
        Args:
            df: DataFrame com dados brutos
        
        Returns:
            DataFrame limpo
        """
        self.cleaning_log = []
        df_clean = df.copy()
        original_len = len(df_clean)
        
        logger.info("Iniciando limpeza de dados...")
        
        # 1. Remover duplicatas
        df_clean = self._remove_duplicates(df_clean)
        
        # 2. Ordenar por data
        df_clean = df_clean.sort_index()
        
        # 3. Tratar dados faltantes
        df_clean = self._handle_missing_values(df_clean)
        
        # 4. Tratar outliers
        df_clean = self._handle_outliers(df_clean)
        
        # 5. Corrigir integridade
        df_clean = self._fix_integrity(df_clean)
        
        # 6. Remover infinitos
        df_clean = self._remove_infinites(df_clean)
        
        # 7. Verificação final
        df_clean = df_clean.dropna()
        
        final_len = len(df_clean)
        removed = original_len - final_len
        
        self.cleaning_log.append(
            f"Limpeza concluída: {original_len} → {final_len} registros "
            f"({removed} removidos, {removed/original_len*100:.2f}%)"
        )
        
        logger.info(self.cleaning_log[-1])
        
        return df_clean
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicatas."""
        # Datas duplicadas - manter primeiro
        if df.index.duplicated().any():
            dups = df.index.duplicated().sum()
            df = df[~df.index.duplicated(keep='first')]
            self.cleaning_log.append(f"Removidas {dups} datas duplicadas")
        
        # Linhas duplicadas
        if df.duplicated().any():
            dups = df.duplicated().sum()
            df = df.drop_duplicates()
            self.cleaning_log.append(f"Removidas {dups} linhas duplicadas")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata dados faltantes."""
        missing_before = df.isna().sum().sum()
        
        if missing_before == 0:
            return df
        
        if self.handle_missing == "drop":
            df = df.dropna()
            
        elif self.handle_missing == "interpolate":
            # Interpolar colunas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(
                method='time',  # Considera o índice temporal
                limit_direction='both'
            )
            # Preencher extremidades se ainda houver NaN
            df = df.ffill().bfill()
            
        elif self.handle_missing == "ffill":
            df = df.ffill()
            
        elif self.handle_missing == "bfill":
            df = df.bfill()
        
        missing_after = df.isna().sum().sum()
        treated = missing_before - missing_after
        
        if treated > 0:
            self.cleaning_log.append(
                f"Tratados {treated} valores faltantes ({self.handle_missing})"
            )
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata outliers."""
        if self.handle_outliers == "keep":
            return df
        
        price_cols = ['Open', 'High', 'Low', 'Close']
        total_treated = 0
        
        for col in price_cols:
            if col not in df.columns:
                continue
            
            # Detectar outliers
            if self.outlier_method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - (self.outlier_threshold * IQR)
                upper = Q3 + (self.outlier_threshold * IQR)
            else:
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - (self.outlier_threshold * std)
                upper = mean + (self.outlier_threshold * std)
            
            outliers = (df[col] < lower) | (df[col] > upper)
            n_outliers = outliers.sum()
            
            if n_outliers == 0:
                continue
            
            if self.handle_outliers == "clip":
                df.loc[df[col] < lower, col] = lower
                df.loc[df[col] > upper, col] = upper
                
            elif self.handle_outliers == "remove":
                df = df[~outliers]
                
            elif self.handle_outliers == "median":
                median = df[col].median()
                df.loc[outliers, col] = median
            
            total_treated += n_outliers
        
        if total_treated > 0:
            self.cleaning_log.append(
                f"Tratados {total_treated} outliers ({self.handle_outliers})"
            )
        
        return df
    
    def _fix_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrige problemas de integridade."""
        fixes = 0
        
        # Garantir High >= Low
        if 'High' in df.columns and 'Low' in df.columns:
            invalid = df['High'] < df['Low']
            if invalid.any():
                # Swap valores
                df.loc[invalid, ['High', 'Low']] = df.loc[invalid, ['Low', 'High']].values
                fixes += invalid.sum()
        
        # Garantir High >= max(Open, Close)
        if all(col in df.columns for col in ['High', 'Open', 'Close']):
            max_oc = df[['Open', 'Close']].max(axis=1)
            invalid = df['High'] < max_oc
            if invalid.any():
                df.loc[invalid, 'High'] = max_oc[invalid]
                fixes += invalid.sum()
        
        # Garantir Low <= min(Open, Close)
        if all(col in df.columns for col in ['Low', 'Open', 'Close']):
            min_oc = df[['Open', 'Close']].min(axis=1)
            invalid = df['Low'] > min_oc
            if invalid.any():
                df.loc[invalid, 'Low'] = min_oc[invalid]
                fixes += invalid.sum()
        
        # Volume negativo -> 0
        if 'Volume' in df.columns:
            invalid = df['Volume'] < 0
            if invalid.any():
                df.loc[invalid, 'Volume'] = 0
                fixes += invalid.sum()
        
        if fixes > 0:
            self.cleaning_log.append(f"Corrigidos {fixes} problemas de integridade")
        
        return df
    
    def _remove_infinites(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove valores infinitos."""
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        
        if inf_count > 0:
            df = df.replace([np.inf, -np.inf], np.nan)
            self.cleaning_log.append(f"Removidos {inf_count} valores infinitos")
        
        return df
    
    def get_cleaning_report(self) -> str:
        """Retorna relatório de limpeza."""
        return "\n".join(["📋 Relatório de Limpeza:"] + self.cleaning_log)


def validate_and_clean(
    df: pd.DataFrame,
    ticker: str = "UNKNOWN",
    auto_clean: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Função de conveniência para validar e limpar dados.
    
    Args:
        df: DataFrame com dados brutos
        ticker: Símbolo da ação
        auto_clean: Se deve limpar automaticamente
        verbose: Se deve imprimir relatórios
    
    Returns:
        Tuple com DataFrame limpo e relatório de qualidade
    """
    # Validar
    validator = DataValidator()
    report = validator.validate(df, ticker)
    
    if verbose:
        print(report.summary())
    
    # Limpar se necessário
    if auto_clean:
        cleaner = DataCleaner()
        df_clean = cleaner.clean(df)
        
        if verbose:
            print("\n" + cleaner.get_cleaning_report())
        
        # Re-validar
        report_after = validator.validate(df_clean, ticker)
        
        if verbose:
            print("\n📊 Após limpeza:")
            print(f"   Registros: {len(df)} → {len(df_clean)}")
            print(f"   Status: {'✅ VÁLIDO' if report_after.is_valid else '❌ INVÁLIDO'}")
        
        return df_clean, report_after
    
    return df, report


# =============================================================================
# EXEMPLO DE USO
# =============================================================================
def main():
    """Demonstração do módulo de validação."""
    
    # Criar dados de exemplo com problemas
    dates = pd.date_range('2024-01-01', periods=100, freq='B')
    
    np.random.seed(42)
    df = pd.DataFrame({
        'Open': 100 + np.random.randn(100).cumsum(),
        'High': 102 + np.random.randn(100).cumsum(),
        'Low': 98 + np.random.randn(100).cumsum(),
        'Close': 100 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Introduzir problemas
    df.iloc[10, 0] = np.nan  # Missing value
    df.iloc[20, 1] = 50      # High < Low (problema de integridade)
    df.iloc[30, 4] = -1000   # Volume negativo
    df.iloc[50, 3] = 500     # Outlier
    
    print("Dados de exemplo criados com problemas propositais")
    print(f"Shape: {df.shape}")
    
    # Validar e limpar
    df_clean, report = validate_and_clean(df, ticker="TEST", verbose=True)
    
    print(f"\nDados finais: {df_clean.shape}")


if __name__ == "__main__":
    main()