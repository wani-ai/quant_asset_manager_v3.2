# /scripts/generate_quality_benchmark.py

import pandas as pd
import logging
import sys

# 시스템의 다른 모듈에서 DB 연결 및 유틸리티 함수를 불러옵니다.
# (스크립트 실행 경로를 고려하여 sys.path에 프로젝트 루트를 추가해야 할 수 있습니다.)
# import sys
# sys.path.append('.')
from data.database import get_database_engine, load_data_from_db, save_df_to_db
import config

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def generate_quality_benchmark():
    """
    '꾸준한 성장 기업'(챔피언 그룹)을 식별하고, 이들의 평균 재무 프로필을 계산하여
    'quality_benchmark' 테이블을 데이터베이스에 생성 또는 업데이트합니다.
    """

    logger.info("퀄리티 벤치마크 생성을 시작합니다...")

    db_engine = get_database_engine()
    if db_engine is None:
        logger.error("데이터베이스 엔진 연결에 실패했습니다. 작업을 중단합니다.")
        return

    try:
        # --- 1. 과거 10년+ 시가총액 데이터 로딩 ---
        logger.info("과거 시가총액 데이터를 로딩합니다...")
        # 가정: historical_market_cap 테이블에 ticker, report_year, market_cap 컬럼이 있음.
        query_mkt_cap = "SELECT ticker, report_year, market_cap FROM historical_market_cap ORDER BY ticker, report_year"
        mkt_cap_df = load_data_from_db(query_mkt_cap, db_engine)
        
        if mkt_cap_df.empty:
            logger.error("시가총액 데이터가 없습니다. 작업을 중단합니다.")
            return

        # --- 2. '꾸준한 성장 기업' 스크리닝 ---
        logger.info("'꾸준한 성장 기업'을 스크리닝합니다...")
        # 연간 시가총액 증가율 계산
        mkt_cap_df['growth_rate'] = mkt_cap_df.groupby('ticker')['market_cap'].pct_change()
        
        # 지난 10년간의 데이터만 필터링 (필요시)
        latest_year = mkt_cap_df['report_year'].max()
        ten_years_ago = latest_year - 10
        mkt_cap_df_10y = mkt_cap_df[mkt_cap_df['report_year'] > ten_years_ago]
        
        # 성장 연도 카운트
        positive_growth_years = mkt_cap_df_10y[mkt_cap_df_10y['growth_rate'] > 0].groupby('ticker').size()
        
        # 8회 이상 성장한 기업 필터링
        champion_tickers = positive_growth_years[positive_growth_years >= 8].index.tolist()
        
        if not champion_tickers:
            logger.warning("스크리닝 조건을 만족하는 '꾸준한 성장 기업'이 없습니다.")
            return
            
        logger.info(f"총 {len(champion_tickers)}개의 '꾸준한 성장 기업'(챔피언 그룹)을 식별했습니다.")

        # --- 3. 챔피언 그룹의 최신 재무 지표 로딩 ---
        logger.info("챔피언 그룹의 최신 재무 지표를 로딩합니다...")
        # SQL 쿼리에서 IN 절을 안전하게 사용하기 위해 튜플로 변환
        tickers_tuple = tuple(champion_tickers)

        if not tickers_tuple:
            logger.warning("챔피언 그룹이 비어 있어 재무 지표를 로딩할 수 없습니다.")
            return
          
        query_metrics = f"SELECT * FROM financial_metrics WHERE ticker IN {tickers_tuple}"
        champion_metrics_df = load_data_from_db(query_metrics, db_engine)

        if champion_metrics_df.empty:
            logger.error("챔피언 그룹의 재무 지표를 가져올 수 없습니다.")
            return

        # --- 4. 퀄리티 벤치마크 계산 ---
        logger.info("퀄리티 벤치마크를 계산합니다 (중앙값 기준)...")
        # 분석에 사용할 지표 목록 (config.py에서 관리)
        benchmark_cols = config.QUALITY_BENCHMARK_COLS
        
        # 데이터프레임에 존재하는 컬럼만으로 필터링
        existing_benchmark_cols = [col for col in benchmark_cols if col in champion_metrics_df.columns]
        

        # 이상치에 강한 중앙값(median)을 사용하여 벤치마크 계산
        quality_benchmark = champion_metrics_df[benchmark_cols].median()
        
        # 계산 결과를 DataFrame으로 변환하여 저장 준비
        benchmark_df = pd.DataFrame(quality_benchmark).T
        benchmark_df['updated_at'] = pd.Timestamp.now()

        logger.info("계산된 퀄리티 벤치마크:")
        print(benchmark_df.to_string())

        # --- 5. 데이터베이스에 저장 ---
        logger.info("계산된 벤치마크를 'quality_benchmark' 테이블에 저장합니다...")
        save_df_to_db(benchmark_df, 'quality_benchmark', db_engine, if_exists='replace')

        logger.info("퀄리티 벤치마크 생성이 성공적으로 완료되었습니다.")

    except Exception as e:
        logger.error(f"퀄리티 벤치마크 생성 중 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    # 이 스크립트를 커맨드 라인에서 직접 실행할 수 있도록 합니다.
    # 예: python -m scripts.generate_quality_benchmark
    generate_quality_benchmark()
