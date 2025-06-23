# /data/database.py

import logging
import pandas as pd
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.exc import SQLAlchemyError
import functools

# 시스템 전역 설정을 불러옵니다.
import config

# -----------------------------------------------------------------------------
# 데이터베이스 엔진을 한 번만 생성하여 재사용하기 위한 캐싱 데코레이터
# -----------------------------------------------------------------------------
@functools.lru_cache(max=None)
def get_database_engine() -> Engine | None:
    """
    시스템 전역에서 사용할 SQLAlchemy 데이터베이스 엔진을 생성하고 반환합니다.
    이 함수는 싱글톤(Singleton) 패턴처럼 동작하여, 애플리케이션 수명 동안
    단 하나의 엔진 인스턴스만 유지하도록 보장합니다.

    :return: 연결에 성공하면 SQLAlchemy Engine 객체를, 실패하면 None을 반환합니다.
    """
    logger = logging.getLogger(__name__)
    try:
        # config.py 또는 .env 파일에 저장된 DB 연결 정보를 불러옵니다.
        db_connection_str = config.DB_CONNECTION_STRING
        if not db_connection_str:
            logger.error("데이터베이스 연결 문자열(DB_CONNECTION_STRING)이 설정되지 않았습니다.")
            return None
            
        engine = create_engine(db_connection_str, pool_pre_ping=True)
        
        # 실제 연결을 테스트하여 엔진이 유효한지 확인합니다.
        with engine.connect() as connection:
            logger.info("데이터베이스 엔진 생성 및 연결 테스트 성공.")
        
        return engine

    except ImportError:
        logger.error("SQLAlchemy 라이브러리가 설치되지 않았습니다. 'pip install SQLAlchemy psycopg2-binary'를 실행해주세요.")
        return None
    except SQLAlchemyError as e:
        logger.error(f"데이터베이스 엔진 생성 중 오류 발생: {e}")
        return None

# -----------------------------------------------------------------------------
# 데이터베이스로부터 데이터를 로드하는 범용 함수
# -----------------------------------------------------------------------------
def load_data_from_db(query: str, db_engine: Engine) -> pd.DataFrame:
    """
    주어진 SQL 쿼리를 실행하여 그 결과를 pandas DataFrame으로 로드합니다.

    :param query: 실행할 SQL 쿼리 문자열.
    :param db_engine: 사용할 SQLAlchemy 엔진 객체.
    :return: 쿼리 결과를 담은 DataFrame. 오류 발생 시 빈 DataFrame 반환.
    """
    logger = logging.getLogger(__name__)
    
    if db_engine is None:
        logger.error("데이터베이스 엔진이 유효하지 않아 쿼리를 실행할 수 없습니다.")
        return pd.DataFrame()
        
    try:
        # SQL 인젝션 공격을 방지하기 위해 쿼리를 text()로 감쌉니다.
        with db_engine.connect() as connection:
            df = pd.read_sql(text(query), connection)
        return df
    except SQLAlchemyError as e:
        logger.error(f"데이터베이스 쿼리 실행 중 오류 발생: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"데이터 로딩 중 예기치 않은 오류 발생: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 데이터프레임을 데이터베이스 테이블에 저장하는 범용 함수 (추가 제안)
# -----------------------------------------------------------------------------
def save_df_to_db(df: pd.DataFrame, table_name: str, db_engine: Engine, if_exists: str = 'replace'):
    """
    pandas DataFrame을 지정된 데이터베이스 테이블에 저장(쓰기)합니다.

    :param df: 저장할 DataFrame.
    :param table_name: 데이터를 저장할 테이블의 이름.
    :param db_engine: 사용할 SQLAlchemy 엔진 객체.
    :param if_exists: 테이블이 이미 존재할 경우의 동작 ('fail', 'replace', 'append').
    """
    logger = logging.getLogger(__name__)

    if db_engine is None:
        logger.error("데이터베이스 엔진이 유효하지 않아 데이터를 저장할 수 없습니다.")
        return
        
    try:
        df.to_sql(table_name, db_engine, if_exists=if_exists, index=True)
        logger.info(f"DataFrame이 '{table_name}' 테이블에 성공적으로 저장되었습니다. ({len(df)} rows)")
    except SQLAlchemyError as e:
        logger.error(f"데이터베이스에 저장 중 오류 발생: {e}")
    except Exception as e:
        logger.error(f"데이터 저장 중 예기치 않은 오류 발생: {e}")

