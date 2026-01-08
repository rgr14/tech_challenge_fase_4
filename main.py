#!/usr/bin/env python3
"""
Stock LSTM Predictor - Ponto de entrada principal
=================================================

Uso:
    python main.py train [--ticker NVDA] [--grid-search] [--train-best]
    python main.py predict [--ticker NVDA]
    python main.py api

Exemplos:
    # Treinar modelo com Grid Search
    python main.py train --grid-search --ticker NVDA

    # Treinar com melhores parametros encontrados
    python main.py train --train-best --ticker NVDA

    # Treinamento simples
    python main.py train --ticker NVDA

    # Fazer previsao
    python main.py predict --ticker NVDA

    # Iniciar API
    python main.py api
"""
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Stock LSTM Predictor - Previsao de precos de acoes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Comandos disponiveis')

    # Comando: train
    train_parser = subparsers.add_parser('train', help='Treinar modelo LSTM')
    train_parser.add_argument(
        '--ticker',
        type=str,
        default='NVDA',
        help='Simbolo da acao (default: NVDA)'
    )
    train_parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Executar Grid Search para encontrar melhores parametros'
    )
    train_parser.add_argument(
        '--train-best',
        action='store_true',
        help='Treinar com melhores parametros do Grid Search'
    )
    train_parser.add_argument(
        '--splits',
        type=int,
        default=3,
        help='Numero de splits para TimeSeriesSplit (default: 3)'
    )
    train_parser.add_argument(
        '--start-date',
        type=str,
        default='2022-01-01',
        help='Data inicial para coleta de dados (default: 2022-01-01)'
    )

    # Comando: predict
    predict_parser = subparsers.add_parser('predict', help='Fazer previsao')
    predict_parser.add_argument(
        '--ticker',
        type=str,
        default='NVDA',
        help='Simbolo da acao (default: NVDA)'
    )

    # Comando: api
    api_parser = subparsers.add_parser('api', help='Iniciar servidor API')
    api_parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host do servidor (default: 0.0.0.0)'
    )
    api_parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Porta do servidor (default: 8000)'
    )

    args = parser.parse_args()

    if args.command == 'train':
        from src.cli.train import run_grid_search, train_with_best_params, train_model, _save_grid_search_plots

        if args.grid_search:
            result = run_grid_search(
                ticker=args.ticker,
                start_date=args.start_date,
                n_splits=args.splits
            )
            if result["all_results"]:
                _save_grid_search_plots(result["all_results"], args.ticker)
            print(f"\nGrid Search concluido!")
            print(f"Para treinar o modelo final, execute:")
            print(f"  python main.py train --ticker {args.ticker} --train-best")

        elif args.train_best:
            train_with_best_params(
                ticker=args.ticker,
                start_date=args.start_date
            )

        else:
            train_model(ticker=args.ticker)

    elif args.command == 'predict':
        # Atualizar ticker global no modulo predict
        from src.cli import predict
        predict.TICKER = args.ticker
        predict.main()

    elif args.command == 'api':
        from src.api.app import main as api_main
        api_main()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
