# /analysis/metrics/liquidity_analyzer.py

import pandas as pd
import logging
from sqlalchemy import Engine
from typing import Dict, Any

from data.database import load_data_from_db

def evaluate_liquidity(ticker: str, db_engine: Engine) -> Dict[str, Any]:
    """
    특정 기업의 유동성(Liquidity)을 종합적으로 평가합니다.
    주요 유동성 비율을 계산하고, 그에 대한 해석을 제공합니다.

    :param ticker: 분석할 주식 티커.
    :param db_engine: 데이터베이스 연결을 위한 SQLAlchemy 엔진.
    :return: 유동성 지표와 평가를 담은 딕셔너리.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"'{ticker}'의 유동성 분석을 시작합니다.")

    # --- 1. 데이터 로딩 ---
    # 가정: financial_metrics 테이블에 필요한 모든 지표가 저장되어 있음.
    query = f"SELECT * FROM financial_metrics WHERE ticker = '{ticker}' LIMIT 1"
    
    try:
        data = load_data_from_db(query, db_engine)
        if data.empty:
            logger.warning(f"'{ticker}'에 대한 재무 지표 데이터를 찾을 수 없습니다.")
            return {"error": "No financial metrics data found."}
        
        metrics = data.iloc[0]

        # --- 2. 유동성 지표 계산 및 평가 ---
        
        # 유동비율 (Current Ratio)
        current_ratio = metrics.get('currentRatio', 0)
        if current_ratio > 2.0:
            current_ratio_eval = "매우 우수 (단기 부채 상환 능력 충분)"
        elif current_ratio > 1.0:
            current_ratio_eval = "양호"
        else:
            current_ratio_eval = "위험 (유동자산이 유동부채보다 적음)"

        # 당좌비율 (Quick Ratio)
        quick_ratio = metrics.get('quickRatio', 0)
        if quick_ratio > 1.0:
            quick_ratio_eval = "우수 (재고 제외해도 단기 부채 상환 가능)"
        elif quick_ratio > 0.5:
            quick_ratio_eval = "양호"
        else:
            quick_ratio_eval = "주의 필요 (재고 의존도 높음)"
            
        # 현금비율 (Cash Ratio)
        cash_ratio = metrics.get('cashRatio', 0)
        if cash_ratio > 0.5:
            cash_ratio_eval = "최우수 (보유 현금만으로 단기 부채 절반 이상 감당)"
        elif cash_ratio > 0.2:
            cash_ratio_eval = "양호 (현금 보유량 안정적)"
        else:
            cash_ratio_eval = "검토 필요"

        # OCF 비율 (Operating Cash Flow to Current Liabilities Ratio)
        # FMP의 'operatingCashFlowSalesRatio'를 대체 지표로 사용하거나,
        # 원본 데이터(현금흐름표, 재무상태표)에서 직접 계산해야 함.
        # 여기서는 financial_metrics에 컬럼이 있다고 가정
        ocf_ratio = metrics.get('operatingCashFlowToCurrentLiabilities', 0)
        if ocf_ratio > 0.4:
             ocf_ratio_eval = "우수 (영업활동 현금흐름 풍부)"
        elif ocf_ratio > 0.2:
             ocf_ratio_eval = "양호"
        else:
             ocf_ratio_eval = "저조 또는 데이터 없음"

        # --- 3. 결과 종합 ---
        report = {
            "summary": "종합적인 유동성은 양호/우수/위험 합니다.", # 점수화 로직 추가 후 계산
            "current_ratio": {
                "value": f"{current_ratio:.2f}",
                "evaluation": current_ratio_eval,
                "question": "1년 내에 갚아야 할 빚을 갚을 능력이 되는가?"
            },
            "quick_ratio": {
                "value": f"{quick_ratio:.2f}",
                "evaluation": quick_ratio_eval,
                "question": "재고가 안 팔리는 최악의 경우에도 단기 부채를 갚을 수 있는가?"
            },
            "cash_ratio": {
                "value": f"{cash_ratio:.2f}",
                "evaluation": cash_ratio_eval,
                "question": "지금 당장 보유한 현금만으로 단기 부채를 얼마나 감당할 수 있는가?"
            },
            "ocf_ratio": {
                "value": f"{ocf_ratio:.2f}",
                "evaluation": ocf_ratio_eval,
                "question": "영업활동으로 벌어들인 현금으로 단기 부채를 얼마나 갚을 수 있는가?"
            }
        }
        
        # 간단한 종합 점수 계산
        eval_scores = {'최우수': 2, '매우 우수': 2, '우수': 2, '양호': 1, '검토 필요': 0, '저조 또는 데이터 없음': 0, '주의 필요': -1, '위험': -2}
        total_score = sum([
            eval_scores.get(report['current_ratio']['evaluation'], 0),
            eval_scores.get(report['quick_ratio']['evaluation'], 0),
            eval_scores.get(report['cash_ratio']['evaluation'], 0),
            eval_scores.get(report['ocf_ratio']['evaluation'], 0)
        ])
        
        if total_score >= 6:
            report['summary'] = "최우수: 단기 채무 상환 능력이 매우 뛰어나며, 유동성 위기 가능성이 매우 낮습니다."
        elif total_score >= 3:
            report['summary'] = "양호: 안정적인 단기 지급 능력을 보유하고 있습니다."
        else:
            report['summary'] = "주의 필요: 단기 부채 상환 능력이 다소 부족하거나 재고 의존도가 높아, 예상치 못한 현금 유출 시 리스크가 발생할 수 있습니다."

        return report

    except Exception as e:
        logger.error(f"'{ticker}' 유동성 분석 중 오류 발생: {e}", exc_info=True)
        return {"error": f"An error occurred during liquidity analysis for {ticker}."}

