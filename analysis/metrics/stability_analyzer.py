# /analysis/metrics/stability_analyzer.py

import pandas as pd
import logging
from sqlalchemy import Engine
from typing import Dict, Any

from data.database import load_data_from_db

def evaluate_stability(ticker: str, db_engine: Engine) -> Dict[str, Any]:
    """
    특정 기업의 재무 안정성(Financial Stability)을 종합적으로 평가합니다.
    주요 부채 비율 및 상환 능력 지표를 계산하고, 그에 대한 해석을 제공합니다.

    :param ticker: 분석할 주식 티커.
    :param db_engine: 데이터베이스 연결을 위한 SQLAlchemy 엔진.
    :return: 안정성 지표와 평가를 담은 딕셔너리.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"'{ticker}'의 재무 안정성 분석을 시작합니다.")

    # --- 1. 데이터 로딩 ---
    # 가정: financial_metrics 테이블에 필요한 모든 지표가 저장되어 있음.
    query = f"SELECT * FROM financial_metrics WHERE ticker = '{ticker}' LIMIT 1"
    
    try:
        data = load_data_from_db(query, db_engine)
        if data.empty:
            logger.warning(f"'{ticker}'에 대한 재무 지표 데이터를 찾을 수 없습니다.")
            return {"error": "No financial metrics data found."}
        
        metrics = data.iloc[0]

        # --- 2. 안정성 지표 계산 및 평가 ---
        
        # 부채비율 (Debt to Equity)
        debt_to_equity = metrics.get('debtEquityRatio', 0)
        if debt_to_equity < 0.5:
            debt_to_equity_eval = "매우 안전 (낮은 부채 의존도)"
        elif debt_to_equity < 1.0:
            debt_to_equity_eval = "양호"
        elif debt_to_equity < 2.0:
            debt_to_equity_eval = "주의 필요"
        else:
            debt_to_equity_eval = "위험 (과도한 부채)"

        # 이자보상배율 (Interest Coverage)
        interest_coverage = metrics.get('interestCoverage', 0)
        if interest_coverage > 5.0:
            interest_coverage_eval = "매우 우수 (이자 지급 능력 충분)"
        elif interest_coverage > 2.0:
            interest_coverage_eval = "양호"
        elif interest_coverage > 1.0:
            interest_coverage_eval = "주의 필요 (영업이익이 이자비용과 비슷한 수준)"
        else:
            interest_coverage_eval = "위험 (영업이익으로 이자 감당 불가)"
            
        # 총부채/EBITDA (Total Debt to EBITDA)
        # 이 지표는 직접 제공되지 않는 경우가 많아, 원본 데이터로 계산 필요
        # total_debt = ... ; ebitda = ...
        # 여기서는 financial_metrics에 컬럼이 있다고 가정
        total_debt_to_ebitda = metrics.get('totalDebtToEbitda', 0)
        if total_debt_to_ebitda > 0 and total_debt_to_ebitda < 3.0:
            total_debt_to_ebitda_eval = "양호 (현금창출력 대비 부채 수준 안정적)"
        elif total_debt_to_ebitda > 0 and total_debt_to_ebitda < 5.0:
            total_debt_to_ebitda_eval = "주의 필요"
        else:
             total_debt_to_ebitda_eval = "위험 또는 데이터 없음"

        # --- 3. 결과 종합 ---
        report = {
            "summary": "종합적인 재무 안정성은 양호/우수/위험 합니다.", # 점수화 로직 추가 후 계산
            "debt_to_equity": {
                "value": f"{debt_to_equity:.2f}",
                "evaluation": debt_to_equity_eval,
                "question": "회사의 재무구조가 타인 자본에 얼마나 의존하고 있는가?"
            },
            "interest_coverage": {
                "value": f"{interest_coverage:.2f}배",
                "evaluation": interest_coverage_eval,
                "question": "영업이익으로 이자를 감당할 능력이 충분한가?"
            },
            "total_debt_to_ebitda": {
                "value": f"{total_debt_to_ebitda:.2f}년",
                "evaluation": total_debt_to_ebitda_eval,
                "question": "회사가 벌어들인 현금으로 총부채를 갚는 데 몇 년이 걸리는가?"
            }
            # DSCR 등 다른 지표들도 동일한 방식으로 추가
        }
        
        # 간단한 종합 점수 계산
        eval_scores = {'매우 안전': 2, '매우 우수': 2, '양호': 1, '주의 필요': -1, '위험': -2, '위험 또는 데이터 없음': 0}
        total_score = sum([
            eval_scores.get(report['debt_to_equity']['evaluation'], 0),
            eval_scores.get(report['interest_coverage']['evaluation'], 0),
            eval_scores.get(report['total_debt_to_ebitda']['evaluation'], 0),
        ])
        
        if total_score >= 4:
            report['summary'] = "최우수: 부채 수준이 매우 낮고 상환 능력이 탁월하여 재무적으로 매우 안정적입니다."
        elif total_score >= 2:
            report['summary'] = "양호: 재무구조가 건전하며, 단기적인 위기 대응 능력을 갖추고 있습니다."
        else:
            report['summary'] = "주의 필요: 부채 의존도가 높거나 이자 상환 부담이 있어, 금리 인상 등 외부 충격에 취약할 수 있습니다."

        return report

    except Exception as e:
        logger.error(f"'{ticker}' 안정성 분석 중 오류 발생: {e}", exc_info=True)
        return {"error": f"An error occurred during stability analysis for {ticker}."}

