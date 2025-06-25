# /analysis/metrics/valuation_analyzer.py

import pandas as pd
import logging
from sqlalchemy import Engine
from typing import Dict, Any

from data.database import load_data_from_db

def evaluate_valuation(ticker: str, db_engine: Engine) -> Dict[str, Any]:
    """
    특정 기업의 가치평가(Valuation) 수준을 종합적으로 평가합니다.
    주요 가치평가 지표를 계산하고, 그에 대한 해석을 제공합니다.

    :param ticker: 분석할 주식 티커.
    :param db_engine: 데이터베이스 연결을 위한 SQLAlchemy 엔진.
    :return: 가치평가 지표와 평가를 담은 딕셔너리.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"'{ticker}'의 가치평가 분석을 시작합니다.")

    # --- 1. 데이터 로딩 ---
    # 가정: financial_metrics 테이블에 필요한 모든 지표가 저장되어 있음.
    query = f"SELECT * FROM financial_metrics WHERE ticker = '{ticker}' LIMIT 1"
    
    try:
        data = load_data_from_db(query, db_engine)
        if data.empty:
            logger.warning(f"'{ticker}'에 대한 재무 지표 데이터를 찾을 수 없습니다.")
            return {"error": "No financial metrics data found."}
        
        metrics = data.iloc[0]

        # --- 2. 가치평가 지표 계산 및 평가 ---
        # 이 지표들은 산업별 편차가 매우 크므로, 상대 평가가 필수적입니다.
        # 여기서는 일반적인 기준으로 '고평가/저평가' 경향을 판단합니다.
        
        # PER (주가수익비율)
        per = metrics.get('peRatio', 0)
        if per > 0 and per < 15:
            per_eval = "저평가 국면"
        elif per >= 15 and per < 25:
            per_eval = "적정 주가"
        elif per >= 25:
            per_eval = "고평가 국면"
        else:
            per_eval = "N/A (적자 기업)"

        # PBR (주가순자산비율)
        pbr = metrics.get('pbRatio', 0)
        if pbr > 0 and pbr < 1:
            pbr_eval = "매우 저평가 (청산가치 이하)"
        elif pbr >= 1 and pbr < 2:
            pbr_eval = "저평가 경향"
        elif pbr >=2 and pbr < 4:
            pbr_eval = "적정 또는 고평가 경향"
        else:
            pbr_eval = "고평가 또는 N/A"

        # PSR (주가매출비율)
        psr = metrics.get('psRatio', 0)
        if psr > 0 and psr < 1:
            psr_eval = "매우 저평가"
        elif psr >= 1 and psr < 2:
            psr_eval = "저평가 경향"
        else:
            psr_eval = "고평가 또는 N/A"
            
        # EV/EBITDA
        ev_to_ebitda = metrics.get('enterpriseValueOverEBITDA', 0)
        if ev_to_ebitda > 0 and ev_to_ebitda < 10:
            ev_to_ebitda_eval = "저평가 경향 (M&A 매력도 높음)"
        elif ev_to_ebitda >= 10 and ev_to_ebitda < 20:
            ev_to_ebitda_eval = "적정 주가"
        else:
            ev_to_ebitda_eval = "고평가 또는 N/A"

        # --- 3. 결과 종합 ---
        report = {
            "summary": "종합적인 가치평가 수준은 고평가/저평가 입니다.", # 점수화 로직 추가 후 계산
            "per": {
                "value": f"{per:.2f}",
                "evaluation": per_eval,
                "question": "현재 주가는 회사가 벌어들이는 이익의 몇 배 수준인가?"
            },
            "pbr": {
                "value": f"{pbr:.2f}",
                "evaluation": pbr_eval,
                "question": "현재 주가는 회사의 순자산(청산가치) 대비 몇 배 수준인가?"
            },
            "psr": {
                "value": f"{psr:.2f}",
                "evaluation": psr_eval,
                "question": "현재 주가는 회사 매출액의 몇 배 수준인가?"
            },
            "ev_to_ebitda": {
                "value": f"{ev_to_ebitda:.2f}",
                "evaluation": ev_to_ebitda_eval,
                "question": "회사를 통째로 사는 데 드는 돈이, 회사가 창출하는 현금의 몇 배인가?"
            }
            # PFCF, PEG 등 다른 지표들도 동일한 방식으로 추가
        }
        
        # 간단한 종합 점수 계산 (낮을수록 좋으므로 점수를 반대로 부여)
        eval_scores = {'매우 저평가': 2, '저평가 국면': 2, '저평가 경향': 1, '적정 주가': 0, '고평가 경향': -1, '고평가 국면': -2, '고평가 또는 N/A': -2, 'N/A (적자 기업)': 0}
        total_score = sum([
            eval_scores.get(report['per']['evaluation'], 0),
            eval_scores.get(report['pbr']['evaluation'], 0),
            eval_scores.get(report['psr']['evaluation'], 0),
            eval_scores.get(report['ev_to_ebitda']['evaluation'], 0)
        ])
        
        if total_score >= 4:
            report['summary'] = "저평가: 현재 주가는 기업의 본질가치 대비 매우 매력적인 수준으로 판단됩니다."
        elif total_score >= 1:
            report['summary'] = "적정 평가: 현재 주가는 합리적인 수준에서 거래되고 있습니다."
        else:
            report['summary'] = "고평가: 현재 주가에는 미래 성장에 대한 높은 기대감이 반영되어 있어, 주의가 필요한 구간입니다."

        return report

    except Exception as e:
        logger.error(f"'{ticker}' 가치평가 분석 중 오류 발생: {e}", exc_info=True)
        return {"error": f"An error occurred during valuation analysis for {ticker}."}

