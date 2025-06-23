# /app/cli.py

import argparse
import pandas as pd
import json

# 시스템의 모든 기능을 관장하는 메인 컨트롤러 클래스를 불러옵니다.
from app import QuantSystem

def setup_pandas_display():
    """Pandas DataFrame의 콘솔 출력 형식을 설정합니다."""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)

def main():
    """
    퀀트 자산 관리 시스템의 메인 CLI(Command-Line Interface) 실행 함수.
    """
    # --- 1. 최상위 파서(Parser) 생성 ---
    parser = argparse.ArgumentParser(
        description="퀀트 자산 관리 시스템 CLI. 분석, 랭킹, 리스크 평가를 수행합니다."
    )
    # 하위 명령어들을 담을 공간 생성
    subparsers = parser.add_subparsers(dest='command', help='실행할 명령어', required=True)

    # --- 2. 'rank' 명령어 파서 생성 ---
    parser_rank = subparsers.add_parser('rank', help='종합 점수 기반 상위 주식 랭킹을 조회합니다.')
    parser_rank.add_argument(
        '--top_n',
        type=int,
        default=20,
        help='조회할 상위 주식의 개수 (기본값: 20)'
    )
    parser_rank.add_argument(
        '--strategy',
        type=str,
        default='blend',
        choices=['value', 'growth', 'blend'],
        help="사용할 가중치 전략 ('value', 'growth', 'blend') (기본값: blend)"
    )
    parser_rank.add_argument(
        '--output',
        type=str,
        default=None,
        help='결과를 저장할 CSV 파일 경로 (예: top_stocks.csv)'
    )

    # --- 3. 'deepdive' 명령어 파서 생성 ---
    parser_deepdive = subparsers.add_parser('deepdive', help='단일 종목에 대한 심층 분석 리포트를 생성합니다.')
    parser_deepdive.add_argument(
        'ticker',
        type=str,
        help='분석할 주식 티커 (예: AAPL)'
    )
    parser_deepdive.add_argument(
        '--output',
        type=str,
        default=None,
        help='리포트를 저장할 JSON 파일 경로 (예: aapl_report.json)'
    )

    # --- 4. 'risk' 명령어 파서 생성 ---
    parser_risk = subparsers.add_parser('risk', help='주어진 포트폴리오의 종합 리스크를 평가합니다.')
    parser_risk.add_argument(
        'tickers',
        type=str,
        nargs='+', # 하나 이상의 티커를 리스트로 받음
        help='포트폴리오를 구성하는 주식 티커 목록 (예: AAPL MSFT GOOGL)'
    )
    parser_risk.add_argument(
        '--output',
        type=str,
        default=None,
        help='리스크 리포트를 저장할 JSON 파일 경로 (예: portfolio_risk.json)'
    )

    # --- 5. 입력된 명령어 파싱 ---
    args = parser.parse_args()

    # --- 6. 시스템 초기화 및 명령어 실행 ---
    print("퀀트 자산 관리 시스템을 초기화합니다...")
    system = QuantSystem()
    setup_pandas_display()
    print("-" * 50)

    if args.command == 'rank':
        print(f"'{args.strategy}' 전략 기반 Top {args.top_n} 주식 랭킹을 조회합니다...")
        ranked_stocks_df = system.get_top_ranked_stocks(top_n=args.top_n, strategy=args.strategy)
        
        if ranked_stocks_df is not None and not ranked_stocks_df.empty:
            print(ranked_stocks_df)
            if args.output:
                ranked_stocks_df.to_csv(args.output)
                print(f"\n결과가 '{args.output}' 파일에 저장되었습니다.")
        else:
            print("분석 결과를 가져오는 데 실패했습니다.")

    elif args.command == 'deepdive':
        print(f"'{args.ticker}'에 대한 심층 분석을 시작합니다...")
        report = system.get_stock_deep_dive_report(ticker=args.ticker)
        
        # 보기 좋게 출력
        print(json.dumps(report, indent=4, ensure_ascii=False))
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=4)
            print(f"\n리포트가 '{args.output}' 파일에 저장되었습니다.")

    elif args.command == 'risk':
        print(f"포트폴리오 {args.tickers}에 대한 리스크 분석을 시작합니다...")
        risk_dashboard_data = system.get_portfolio_risk_dashboard(portfolio_tickers=args.tickers)
        
        # 보기 좋게 출력
        print(json.dumps(risk_dashboard_data, indent=4, ensure_ascii=False))

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(risk_dashboard_data, f, ensure_ascii=False, indent=4)
            print(f"\n리스크 리포트가 '{args.output}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()
