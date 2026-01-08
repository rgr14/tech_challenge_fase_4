"""
Modulo de Validacao e Limpeza de Dados para Series Temporais Financeiras
=========================================================================
Responsavel por garantir a qualidade dos dados antes do treinamento do LSTM.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import warnings
from pathlib import Path

from src.config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Relatorio de qualidade dos dados."""
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
        """Gera resumo textual do relatorio."""
        lines = [
            "\n" + "=" * 60,
            f"RELATORIO DE QUALIDADE - {self.ticker}",
            "=" * 60,
            f"\nPeriodo: {self.date_range[0]} ate {self.date_range[1]}",
            f"Total de registros: {self.total_records:,}",
        ]

        lines.append("\n--- Outliers ---")
        if self.total_outliers == 0:
            lines.append("Nenhum outlier extremo detectado")
        else:
            for col, count in self.outliers_detected.items():
                if count > 0:
                    lines.append(f"  {col}: {count} outliers")

        return "\n".join(lines)


class DataValidator:
    """Classe para validacao de dados financeiros."""

    def __init__(
        self,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        max_missing_pct: float = 5.0,
        check_trading_days: bool = True
    ):
        """
        Args:
            outlier_method: Metodo de deteccao ('iqr', 'zscore', 'mad')
            outlier_threshold: Limiar para deteccao de outliers
            max_missing_pct: Percentual maximo aceitavel de dados faltantes
            check_trading_days: Se deve verificar dias de negociacao faltantes
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.max_missing_pct = max_missing_pct
        self.check_trading_days = check_trading_days
        self.report: Optional[DataQualityReport] = None

    def validate(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> DataQualityReport:
        """
        Executa validacao completa dos dados.

        Args:
            df: DataFrame com dados OHLCV
            ticker: Simbolo da acao

        Returns:
            Relatorio de qualidade
        """
        logger.info(f"Iniciando validacao de dados para {ticker}...")

        df = df.copy()

        # Garantir indice datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Inicializar relatorio
        self.report = DataQualityReport(
            ticker=ticker,
            total_records=len(df),
            date_range=(
                df.index.min().strftime("%Y-%m-%d"),
                df.index.max().strftime("%Y-%m-%d")
            )
        )

        # Executar validacoes
        self._check_missing_values(df)
        self._check_duplicates(df)
        self._check_temporal_gaps(df)
        self._check_outliers(df)
        self._check_data_integrity(df)

        # Determinar status final
        self._determine_final_status()

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
                self.report.errors.append(f"Coluna obrigatoria '{col}' nao encontrada")

        # Verificar NaN infinitos
        for col in df.select_dtypes(include=[np.number]).columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                self.report.warnings.append(f"Coluna {col} contem {inf_count} valores infinitos")

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
        """Verifica gaps na serie temporal."""
        if not self.check_trading_days:
            return

        df_sorted = df.sort_index()

        # Gerar range de dias uteis esperados
        date_range = pd.date_range(
            start=df_sorted.index.min(),
            end=df_sorted.index.max(),
            freq='B'  # Business days
        )

        # Encontrar dias faltantes
        existing_dates = set(df_sorted.index.date)
        expected_dates = set(date_range.date)

        # Remover feriados conhecidos (simplificado - apenas fins de semana ja sao tratados)
        missing_dates = expected_dates - existing_dates

        # Filtrar apenas dias uteis realmente faltantes (tolerancia para feriados)
        # Consideramos aceitavel ate 30 dias de "feriados" por ano
        if len(missing_dates) > 0:
            self.report.missing_trading_days = sorted([
                d.strftime("%Y-%m-%d") for d in missing_dates
            ])
            self.report.total_gaps = len(missing_dates)

            # So avisar se muitos gaps
            years = (df_sorted.index.max() - df_sorted.index.min()).days / 365
            expected_holidays = int(years * 15)  # ~15 feriados por ano

            if self.report.total_gaps > expected_holidays:
                self.report.warnings.append(
                    f"{self.report.total_gaps} dias de negociacao potencialmente faltando"
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
        """Verifica integridade logica dos dados OHLCV."""
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

        # Precos devem ser positivos
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                invalid = df[df[col] <= 0]
                if len(invalid) > 0:
                    issues.append(f"{len(invalid)} registros com {col} <= 0")

        # Variacoes extremas (> 50% em um dia)
        if 'Close' in df.columns:
            returns = df['Close'].pct_change().abs()
            extreme = returns[returns > 0.5]
            if len(extreme) > 0:
                issues.append(f"{len(extreme)} dias com variacao > 50%")
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
        """Determina status final da validacao."""
        # Erros criticos invalidam os dados
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
            handle_missing: Metodo para dados faltantes ('drop', 'interpolate', 'ffill', 'bfill')
            handle_outliers: Metodo para outliers ('keep', 'clip', 'remove', 'median')
            outlier_method: Metodo de deteccao de outliers
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

        # 7. Verificacao final
        df_clean = df_clean.dropna()

        final_len = len(df_clean)
        removed = original_len - final_len

        self.cleaning_log.append(
            f"Limpeza concluida: {original_len} -> {final_len} registros "
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
            # Interpolar colunas numericas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(
                method='time',  # Considera o indice temporal
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
        """Retorna relatorio de limpeza."""
        return "\n".join(["Relatorio de Limpeza:"] + self.cleaning_log)


def validate_and_clean(
    df: pd.DataFrame,
    ticker: str = "UNKNOWN",
    auto_clean: bool = True,
    verbose: bool = False
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Funcao de conveniencia para validar e limpar dados.

    Args:
        df: DataFrame com dados brutos
        ticker: Simbolo da acao
        auto_clean: Se deve limpar automaticamente
        verbose: Se deve imprimir relatorios

    Returns:
        Tuple com DataFrame limpo e relatorio de qualidade
    """
    # Validar
    validator = DataValidator()
    report = validator.validate(df, ticker)

    if verbose:
        print(report.summary())

    # Limpar se necessario
    if auto_clean:
        cleaner = DataCleaner()
        df_clean = cleaner.clean(df)

        # Re-validar
        report_after = validator.validate(df_clean, ticker)

        return df_clean, report_after

    return df, report


def generate_boxplots(
    df: pd.DataFrame,
    ticker: str = "STOCK",
    output_dir: Optional[Path] = None,
    show_plot: bool = False
) -> str:
    """
    Gera boxplots para todas as variaveis quantitativas do DataFrame.

    Args:
        df: DataFrame com dados OHLCV do yfinance
        ticker: Simbolo da acao para titulo
        output_dir: Diretorio para salvar o grafico (default: ./artifacts/models)
        show_plot: Se deve exibir o grafico interativamente

    Returns:
        Caminho do arquivo salvo
    """
    # Selecionar apenas colunas numericas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError("Nenhuma coluna numerica encontrada no DataFrame")

    # Separar Volume das outras colunas (escala diferente)
    price_cols = [col for col in numeric_cols if col != 'Volume']
    has_volume = 'Volume' in numeric_cols

    # Configurar layout
    if has_volume:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_prices = axes[0]
        ax_volume = axes[1]
    else:
        fig, ax_prices = plt.subplots(1, 1, figsize=(10, 6))

    # Boxplot das variaveis de preco
    if price_cols:
        bp1 = ax_prices.boxplot(
            [df[col].dropna() for col in price_cols],
            labels=price_cols,
            patch_artist=True,
            notch=True
        )

        # Cores para os boxes
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(price_cols)))
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)

        ax_prices.set_title(f'Boxplot - Variaveis de Preco ({ticker})', fontsize=12, fontweight='bold')
        ax_prices.set_ylabel('Valor (USD)', fontsize=10)
        ax_prices.grid(True, alpha=0.3, axis='y')

        # Adicionar estatisticas
        stats_text = []
        for col in price_cols:
            data = df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
            stats_text.append(f"{col}: {outliers} outliers")

        ax_prices.text(
            0.02, 0.98, '\n'.join(stats_text),
            transform=ax_prices.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    # Boxplot do Volume (escala separada)
    if has_volume:
        bp2 = ax_volume.boxplot(
            [df['Volume'].dropna()],
            labels=['Volume'],
            patch_artist=True,
            notch=True
        )
        bp2['boxes'][0].set_facecolor(plt.cm.Greens(0.6))

        ax_volume.set_title(f'Boxplot - Volume ({ticker})', fontsize=12, fontweight='bold')
        ax_volume.set_ylabel('Volume', fontsize=10)
        ax_volume.grid(True, alpha=0.3, axis='y')

        # Formatar eixo Y para milhoes
        ax_volume.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M')
        )

        # Estatisticas do Volume
        vol_data = df['Volume'].dropna()
        Q1 = vol_data.quantile(0.25)
        Q3 = vol_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((vol_data < Q1 - 1.5 * IQR) | (vol_data > Q3 + 1.5 * IQR)).sum()

        ax_volume.text(
            0.02, 0.98, f"Volume: {outliers} outliers",
            transform=ax_volume.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    # Titulo geral
    fig.suptitle(
        f'Analise Exploratoria - Deteccao de Outliers ({ticker})\n'
        f'Periodo: {df.index.min().strftime("%Y-%m-%d")} a {df.index.max().strftime("%Y-%m-%d")} | '
        f'N = {len(df)} registros',
        fontsize=11,
        y=1.02
    )

    plt.tight_layout()

    # Salvar grafico
    if output_dir is None:
        output_dir = MODELS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{ticker}_boxplots_eda.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Boxplots salvos em: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return str(output_path)


def print_outlier_summary(df: pd.DataFrame, ticker: str = "STOCK") -> None:
    """
    Imprime resumo de outliers para cada variavel quantitativa.

    Args:
        df: DataFrame com dados OHLCV
        ticker: Simbolo da acao
    """
    print("\n" + "=" * 60)
    print(f"RESUMO DE OUTLIERS - {ticker}")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_low = data[data < lower_bound]
        outliers_high = data[data > upper_bound]
        total_outliers = len(outliers_low) + len(outliers_high)
        pct = (total_outliers / len(data)) * 100

        print(f"\n{col}:")
        print(f"  Min: {data.min():.2f} | Max: {data.max():.2f}")
        print(f"  Q1: {Q1:.2f} | Mediana: {data.median():.2f} | Q3: {Q3:.2f}")
        print(f"  IQR: {IQR:.2f}")
        print(f"  Limites: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  Outliers: {total_outliers} ({pct:.2f}%)")
        if len(outliers_low) > 0:
            print(f"    - Abaixo: {len(outliers_low)}")
        if len(outliers_high) > 0:
            print(f"    - Acima: {len(outliers_high)}")

    print("\n" + "=" * 60)


def main():
    """Demonstracao do modulo de validacao e analise exploratoria."""
    import argparse

    parser = argparse.ArgumentParser(description='Validacao e EDA de dados financeiros')
    parser.add_argument('--ticker', type=str, default='NVDA', help='Simbolo da acao')
    parser.add_argument('--boxplot', action='store_true', help='Gerar boxplots')
    parser.add_argument('--show', action='store_true', help='Exibir graficos')
    args = parser.parse_args()

    from src.config import DATA_DIR

    # Tentar carregar dados reais
    data_path = DATA_DIR / f"{args.ticker}_historical.csv"

    if data_path.exists():
        print(f"Carregando dados de {data_path}")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        print(f"Arquivo {data_path} nao encontrado. Criando dados de exemplo...")
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        np.random.seed(42)
        df = pd.DataFrame({
            'Open': 100 + np.random.randn(100).cumsum(),
            'High': 102 + np.random.randn(100).cumsum(),
            'Low': 98 + np.random.randn(100).cumsum(),
            'Close': 100 + np.random.randn(100).cumsum(),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

    print(f"Shape: {df.shape}")

    # Validar e limpar
    df_clean, report = validate_and_clean(df, ticker=args.ticker, verbose=True)
    print(f"\nDados finais: {df_clean.shape}")

    # Gerar boxplots
    if args.boxplot:
        print("\nGerando boxplots...")
        output_path = generate_boxplots(df_clean, ticker=args.ticker, show_plot=args.show)
        print(f"Boxplots salvos em: {output_path}")

        # Imprimir resumo de outliers
        print_outlier_summary(df_clean, ticker=args.ticker)


if __name__ == "__main__":
    main()
