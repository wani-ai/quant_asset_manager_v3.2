# /scripts/populate_database.py

import pandas as pd
import logging
import sys
import time
from tqdm import tqdm  # 진행 상황을 보여주는 라이브러리 (pip install tqdm)

# 시스템의 다른 모듈에서 핵심 클래스와 함수를 불러옵니다.
# 스크립트 실행 경로 문제를 방지하기 위해, 프로젝트의 최상위 루트에서
# 'python -m scripts.populate_database' 와 같이 실행하는 것을 권장합니다.
from data.database import get_database_engine, save_df_to_db
from data.connectors import SmartDataManager
import config

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def fetch_and_save_data(tickers: list, db_engine, data_manager: SmartDataManager):
    """
    주어진 티커 리스트에 대해 시가총액 및 재무 지표 데이터를 수집하고 DB에 저장합니다.
    """
    
    # --- 1. Historical Market Cap 데이터 수집 및 저장 ---
    logger.info("과거 시가총액 데이터 수집을 시작합니다...")
    all_market_caps = []
    
    for ticker in tqdm(tickers, desc="Fetching Market Caps"):
        try:
            mkt_cap_data = data_manager.get_historical_market_cap(ticker, limit=12) # 최근 12년치
            if mkt_cap_data:
                for item in mkt_cap_data:
                    all_market_caps.append({
                        'ticker': ticker,
                        'report_year': item.get('year'),
                        'market_cap': item.get('marketCap')
                    })
        except Exception as e:
            logger.error(f"'{ticker}'의 시가총액 데이터 수집 중 오류: {e}")
        # API 호출 제한 준수를 위해 각 호출 후 대기
        time.sleep(1) 

    if all_market_caps:
        mkt_cap_df = pd.DataFrame(all_market_caps).dropna()
        logger.info(f"총 {len(mkt_cap_df)}개의 시가총액 데이터를 'historical_market_cap' 테이블에 저장합니다.")
        save_df_to_db(mkt_cap_df, 'historical_market_cap', db_engine, if_exists='replace')
    else:
        logger.warning("수집된 시가총액 데이터가 없습니다.")

    # --- 2. Financial Metrics 데이터 수집 및 저장 ---
    logger.info("핵심 재무 지표 데이터(TTM 기준) 수집을 시작합니다...")
    all_financial_metrics = []

    for ticker in tqdm(tickers, desc="Fetching Financial Metrics"):
        try:
            metrics_data = data_manager.get_financial_ratios_ttm(ticker)
            if metrics_data:
                metrics_data['ticker'] = ticker
                all_financial_metrics.append(metrics_data)
        except Exception as e:
            logger.error(f"'{ticker}'의 재무 지표 데이터 수집 중 오류: {e}")
        # API 호출 제한 준수
        time.sleep(1) 
        
    if all_financial_metrics:
        metrics_df = pd.DataFrame(all_financial_metrics)
        # 분석에 필요한 컬럼만 선택 (config.py에서 관리)
        required_cols = ['ticker']
        if hasattr(config, 'QUALITY_BENCHMARK_COLS'):
            required_cols += config.QUALITY_BENCHMARK_COLS
        if hasattr(config, 'METRICS_FOR_SCORING'):
            required_cols += list(config.METRICS_FOR_SCORING.keys())
        
        # 중복된 컬럼명 제거
        required_cols = sorted(list(set(required_cols)))

        # 데이터프레임에 존재하는 컬럼만 최종적으로 필터링
        existing_cols = [col for col in required_cols if col in metrics_df.columns]
        metrics_df = metrics_df[existing_cols]

        logger.info(f"총 {len(metrics_df)}개 기업의 재무 지표를 'financial_metrics' 테이블에 저장합니다.")
        save_df_to_db(metrics_df, 'financial_metrics', db_engine, if_exists='replace')
    else:
        logger.warning("수집된 재무 지표 데이터가 없습니다.")


def main():
    """
    데이터베이스를 초기 데이터로 채우는 메인 실행 함수.
    """
    logger.info("--- 데이터베이스 채우기(Population) 스크립트를 시작합니다. ---")
    
    # 1. 시스템 모듈 초기화
    db_engine = get_database_engine()
    if db_engine is None:
        logger.error("데이터베이스 엔진 연결 실패. 작업을 중단합니다.")
        return
        
    data_manager = SmartDataManager()

    # 2. 분석 대상 티커 목록 가져오기
    try:
        logger.info("나스닥 전체 상장 기업 목록을 동적으로 가져옵니다...")
        # ✨✨✨ 핵심 변경: get_all_symbols 대신 get_nasdaq_symbols를 호출합니다. ✨✨✨
        target_tickers = data_manager.get_nasdaq_symbols()
    
        if not target_tickers:
            logger.warning("동적 티커 목록 로딩 실패. 작업을 중단합니다.")
            return

        # [주의!] 전체 티커 수는 수천 개에 달하므로, 초기 테스트 시에는 아래 코드의 주석을 해제하여
        # 일부만 사용하시는 것을 강력히 권장합니다.
        # target_tickers = target_tickers[:100] # 예: 상위 100개만 테스트

        logger.info(f"총 {len(target_tickers)}개의 티커를 대상으로 데이터 수집을 시작합니다.")
        logger.warning("전체 티커 데이터 수집은 API 호출 제한으로 인해 수 시간이 소요될 수 있습니다.")

    except Exception as e:
        logger.error(f"티커 목록을 가져오는 데 실패했습니다: {e}")
        return

    # 3. 데이터 수집 및 저장 실행
    fetch_and_save_data(target_tickers, db_engine, data_manager)

    logger.info("--- 모든 데이터 수집 및 저장 작업이 완료되었습니다. ---")

if __name__ == "__main__":
    main()
