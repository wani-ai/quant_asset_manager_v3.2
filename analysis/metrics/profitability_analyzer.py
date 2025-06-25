# /analysis/metrics/profitability_analyzer.py

import pandas as pd
import logging
from sqlalchemy import Engine
from typing import Dict, Any

from data.database import load_data_from_db

def evaluate_profitability(ticker: str, db_engine: Engine) -> Dict[str, Any]:
    """
    특정 기업의 수익성(Profitability)을 종합적으로 평가합니다.
    주요 마진 지표와 자본 수익률 지표를 계산하고, 그에 대한 해석을 제공합니다.

    :param ticker: 분석할 주식 티커.
    :param db_engine: 데이터베이스 연결을 위한 SQLAlchemy 엔진.
    :return: 수익성 지표와 평가를 담은 딕셔너리.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"'{ticker}'의 수익성 분석을 시작합니다.")

    # --- 1. 데이터 로딩 ---
    # 가정: financial_metrics 테이블에 필요한 모든 지표가 저장되어 있음.
    # 실제 구현에서는 더 많은 원본 데이터(예: 재무제표)가 필요할 수 있습니다.
    query = f"SELECT * FROM financial_metrics WHERE ticker = '{ticker}' LIMIT 1"
    
    try:
        data = load_data_from_db(query, db_engine)
        if data.empty:
            logger.warning(f"'{ticker}'에 대한 재무 지표 데이터를 찾을 수 없습니다.")
            return {"error": "No financial metrics data found."}
        
        metrics = data.iloc[0]

        # --- 2. 수익성 지표 계산 및 평가 ---
        # 제공된 지표 목록을 기반으로 평가를 수행합니다.
        
        # 매출총이익률 (Gross Margin)
        gross_margin = metrics.get('grossProfitMargin', 0)
        if gross_margin > 0.4:
            gross_margin_eval = "우수 (강력한 제품/서비스 경쟁력)"
        elif gross_margin > 0.2:
            gross_margin_eval = "양호"
        else:
            gross_margin_eval = "검토 필요"

        # 영업이익률 (Operating Margin)
        op_margin = metrics.get('operatingMargin', 0)
        if op_margin > 0.15:
            op_margin_eval = "우수 (효율적인 주력 사업 운영)"
        elif op_margin > 0.05:
            op_margin_eval = "양호"
        else:
            op_margin_eval = "저조 (수익성 개선 필요)"

        # 순이익률 (Net Margin)
        net_margin = metrics.get('netProfitMargin', 0)
        if net_margin > 0.10:
            net_margin_eval = "우수 (뛰어난 최종 수익성)"
        elif net_margin > 0.05:
            net_margin_eval = "양호"
        else:
            net_margin_eval = "저조"

        # ROE (자기자본이익률)
        roe = metrics.get('roe', 0)
        if roe > 0.15:
            roe_eval = "우수 (주주 자본 활용 능력 탁월)"
        elif roe > 0.08:
            roe_eval = "양호"
        else:
            roe_eval = "저조 (자본 효율성 개선 필요)"

        # ROA (총자산이익률)
        roa = metrics.get('roa', 0)
        if roa > 0.05:
            roa_eval = "우수 (효율적인 자산 활용)"
        else:
            roa_eval = "검토 필요"
            
        # --- 3. 결과 종합 ---
        report = {
            "summary": "종합적인 수익성은 양호/우수/저조 합니다.", # 이 부분은 점수화 로직 추가 후 계산
            "gross_margin": {
                "value": f"{gross_margin:.2%}",
                "evaluation": gross_margin_eval,
                "question": "제품/서비스 자체의 기본적인 수익성은 높은가?"
            },
            "operating_margin": {
                "value": f"{op_margin:.2%}",
                "evaluation": op_margin_eval,
                "question": "주력 사업으로 돈을 얼마나 잘 벌고 있는가?"
            },
            "net_margin": {
                "value": f"{net_margin:.2%}",
                "evaluation": net_margin_eval,
                "question": "모든 비용을 제외하고 주주에게 최종적으로 남는 이익은 얼마나 되는가?"
            },
            "roe": {
                "value": f"{roe:.2%}",
                "evaluation": roe_eval,
                "question": "주주의 돈으로 얼마나 효율적으로 이익을 창출하고 있는가?"
            },
            "roa": {
                "value": f"{roa:.2%}",
                "evaluation": roa_eval,
                "question": "회사가 가진 모든 자산을 활용해 얼마나 효율적으로 이익을 내는가?"
            }
            # EBITDA Margin, ROIC 등 다른 지표들도 동일한 방식으로 추가
        }
        
        # 간단한 종합 점수 계산
        eval_scores = {'우수': 2, '양호': 1, '검토 필요': 0, '저조': 0}
        total_score = sum([
            eval_scores.get(report['gross_margin']['evaluation'], 0),
            eval_scores.get(report['operating_margin']['evaluation'], 0),
            eval_scores.get(report['net_margin']['evaluation'], 0),
            eval_scores.get(report['roe']['evaluation'], 0),
            eval_scores.get(report['roa']['evaluation'], 0)
        ])
        
        if total_score >= 8:
            report['summary'] = "최우수: 모든 수익성 지표가 업계 최고 수준입니다."
        elif total_score >= 5:
            report['summary'] = "양호: 전반적으로 안정적인 수익 창출 능력을 보유하고 있습니다."
        else:
            report['summary'] = "검토 필요: 일부 수익성 지표에 대한 개선이 필요합니다."

        return report

    except Exception as e:
        logger.error(f"'{ticker}' 수익성 분석 중 오류 발생: {e}", exc_info=True)
        return {"error": f"An error occurred during profitability analysis for {ticker}."}

