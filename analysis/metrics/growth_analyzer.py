# /analysis/metrics/growth_analyzer.py

import pandas as pd
import logging
from sqlalchemy import Engine
from typing import Dict, Any

from data.database import load_data_from_db

def evaluate_growth(ticker: str, db_engine: Engine) -> Dict[str, Any]:
    """
    특정 기업의 성장성(Growth)을 종합적으로 평가합니다.
    매출, 순이익, EPS의 성장률을 계산하고, 그에 대한 해석을 제공합니다.

    :param ticker: 분석할 주식 티커.
    :param db_engine: 데이터베이스 연결을 위한 SQLAlchemy 엔진.
    :return: 성장성 지표와 평가를 담은 딕셔너리.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"'{ticker}'의 성장성 분석을 시작합니다.")

    # --- 1. 데이터 로딩 ---
    # 가정: financial_metrics 테이블에 TTM 기준 성장률 지표가 저장되어 있음.
    query = f"SELECT * FROM financial_metrics WHERE ticker = '{ticker}' LIMIT 1"
    
    try:
        data = load_data_from_db(query, db_engine)
        if data.empty:
            logger.warning(f"'{ticker}'에 대한 재무 지표 데이터를 찾을 수 없습니다.")
            return {"error": "No financial metrics data found."}
        
        metrics = data.iloc[0]

        # --- 2. 성장성 지표 계산 및 평가 ---
        
        # 매출액증가율 (Revenue Growth YoY)
        revenue_growth = metrics.get('revenueGrowth', 0)
        if revenue_growth > 0.15:
            revenue_growth_eval = "최우수 (매우 빠른 외형 성장)"
        elif revenue_growth > 0.05:
            revenue_growth_eval = "양호 (안정적인 성장세)"
        elif revenue_growth > 0:
            revenue_growth_eval = "정체 (성장 둔화)"
        else:
            revenue_growth_eval = "역성장"

        # 순이익증가율 (Net Income Growth YoY)
        net_income_growth = metrics.get('netIncomeGrowth', 0)
        if net_income_growth > 0.20:
            net_income_growth_eval = "최우수 (이익 창출 능력 급증)"
        elif net_income_growth > 0.05:
            net_income_growth_eval = "양호 (수익성 개선 중)"
        else:
            net_income_growth_eval = "주의 필요 (이익 감소 또는 정체)"
            
        # EPS 성장률 (EPS Growth YoY)
        eps_growth = metrics.get('epsGrowth', 0)
        if eps_growth > 0.20:
            eps_growth_eval = "최우수 (주주가치 매우 빠르게 증가)"
        elif eps_growth > 0.05:
            eps_growth_eval = "양호"
        else:
            eps_growth_eval = "주의 필요 (주당 순이익 정체 또는 감소)"

        # --- 3. 결과 종합 ---
        report = {
            "summary": "종합적인 성장성은 양호/우수/저조 합니다.", # 점수화 로직 추가 후 계산
            "revenue_growth": {
                "value": f"{revenue_growth:.2%}",
                "evaluation": revenue_growth_eval,
                "question": "회사의 외형(규모)이 얼마나 빠르게 커지고 있는가?"
            },
            "net_income_growth": {
                "value": f"{net_income_growth:.2%}",
                "evaluation": net_income_growth_eval,
                "question": "회사의 이익이 얼마나 빠르게 증가하고 있는가?"
            },
            "eps_growth": {
                "value": f"{eps_growth:.2%}",
                "evaluation": eps_growth_eval,
                "question": "주주 한 명에게 돌아가는 이익이 얼마나 빠르게 증가하고 있는가?"
            }
        }
        
        # 간단한 종합 점수 계산
        eval_scores = {'최우수': 2, '양호': 1, '정체': 0, '주의 필요': -1, '역성장': -2}
        total_score = sum([
            eval_scores.get(report['revenue_growth']['evaluation'], 0),
            eval_scores.get(report['net_income_growth']['evaluation'], 0),
            eval_scores.get(report['eps_growth']['evaluation'], 0)
        ])
        
        if total_score >= 5:
            report['summary'] = "최우수: 매출과 이익이 모두 폭발적으로 성장하고 있으며, 강력한 성장 모멘텀을 보유하고 있습니다."
        elif total_score >= 2:
            report['summary'] = "양호: 안정적인 성장세를 보이고 있으며, 향후 성장이 기대됩니다."
        else:
            report['summary'] = "주의 필요: 성장성이 정체되거나 역성장하고 있어, 성장 동력에 대한 점검이 필요합니다."

        return report

    except Exception as e:
        logger.error(f"'{ticker}' 성장성 분석 중 오류 발생: {e}", exc_info=True)
        return {"error": f"An error occurred during growth analysis for {ticker}."}

