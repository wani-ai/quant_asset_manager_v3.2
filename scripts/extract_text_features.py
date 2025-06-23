# /scripts/extract_text_features.py

import logging
import sys
import os
import re
from pathlib import Path

# PDF 처리를 위한 라이브러리 (pip install PyPDF2 필요)
import PyPDF2

# 시스템의 다른 모듈에서 DB 연결 및 유틸리티 함수를 불러옵니다.
# import sys
# sys.path.append('.')
from data.database import get_database_engine, save_df_to_db
import config
import pandas as pd

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

def extract_key_sections_from_filings():
    """
    /knowledge_base/data/ 디렉토리의 10-K 보고서(PDF)에서
    'Item 1A. Risk Factors'와 같은 핵심 섹션의 텍스트를 추출하여
    PostgreSQL 데이터베이스에 저장합니다.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- 10-K 보고서 핵심 섹션 추출을 시작합니다. ---")

    # --- 1. 설정 및 경로 불러오기 ---
    raw_data_dir = getattr(config, 'RAW_DATA_DIR', Path('knowledge_base/data'))
    db_engine = get_database_engine()

    if db_engine is None:
        logger.error("데이터베이스 엔진 연결에 실패했습니다. 작업을 중단합니다.")
        return
    
    if not raw_data_dir.exists() or not any(raw_data_dir.iterdir()):
        logger.warning(f"'{raw_data_dir}' 디렉토리가 비어있습니다. 분석할 PDF 파일을 추가해주세요.")
        return

    # --- 2. PDF 파일 순회 및 텍스트 추출 ---
    extracted_data = []
    
    for pdf_path in raw_data_dir.glob("*.pdf"):
        try:
            logger.info(f"파일 처리 중: {pdf_path.name}")
            
            # 파일명에서 티커와 연도 파싱 (예: AAPL_10K_2023.pdf)
            filename_parts = pdf_path.stem.split('_')
            if len(filename_parts) < 3:
                logger.warning(f"파일명 형식이 올바르지 않아 건너뜁니다: {pdf_path.name}")
                continue
            ticker, doc_type, year = filename_parts[0], filename_parts[1], int(filename_parts[2])

            if doc_type.upper() != '10K':
                logger.info(f"10-K 보고서가 아니므로 건너뜁니다: {pdf_path.name}")
                continue

            # PDF에서 텍스트 읽기
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                full_text = "".join(page.extract_text() for page in reader.pages)

            # --- 3. 정규 표현식을 사용한 'Risk Factors' 섹션 검색 ---
            # SEC 보고서의 "Item 1A." 패턴을 대소문자, 공백 변형을 고려하여 검색
            # (?s)는 줄바꿈 문자(newline)도 '.'에 포함되도록 하는 플래그
            pattern = re.compile(r"(?is)Item\s*1A\.(?:\s*|&nbsp;)*Risk\s*Factors\.(.*?)(?:Item\s*1B\.|Item\s*2\.)", re.DOTALL)
            match = pattern.search(full_text)
            
            risk_factors_text = ""
            if match:
                risk_factors_text = match.group(1).strip()
                logger.info(f"'{ticker}'의 Risk Factors 섹션을 성공적으로 추출했습니다.")
            else:
                logger.warning(f"'{ticker}' 파일에서 Risk Factors 섹션을 찾지 못했습니다.")

            extracted_data.append({
                'ticker': ticker,
                'year': year,
                'section': 'Risk Factors',
                'content': risk_factors_text
            })

        except Exception as e:
            logger.error(f"'{pdf_path.name}' 파일 처리 중 오류 발생: {e}")

    # --- 4. 데이터베이스에 저장 ---
    if not extracted_data:
        logger.info("추출된 데이터가 없습니다. 작업을 종료합니다.")
        return

    df = pd.DataFrame(extracted_data)
    
    logger.info("추출된 텍스트 데이터를 'sec_filing_sections' 테이블에 저장합니다...")
    save_df_to_db(df, 'sec_filing_sections', db_engine, if_exists='replace')

    logger.info("--- 핵심 섹션 추출 및 저장이 성공적으로 완료되었습니다. ---")

if __name__ == "__main__":
    extract_key_sections_from_filings()
