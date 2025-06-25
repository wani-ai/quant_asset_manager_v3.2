# /analysis/metrics/efficiency_analyzer.py

import pandas as pd
import logging
from sqlalchemy import Engine
from typing import Dict, Any

from data.database import load_data_from_db

def evaluate_efficiency(ticker: str, db_engine: Engine) -> Dict[str, Any]:
    """
    특정 기업의 운영 효율성(Efficiency)을 종합적으로 평가합니다.
    주요 자산 회전율 지표를 계산하고, 그에 대한 해석을 제공합니다.

    :param ticker: 분석할 주식 티커.
    :param db_engine: 데이터베이스 연결을 위한 SQLAlchemy 엔진.
    :return: 효율성 지표와 평가를 담은 딕셔너리.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"'{ticker}'의 운영 효율성 분석을 시작합니다.")

    # --- 1. 데이터 로딩 ---
    # 가정: financial_metrics 테이블에 필요한 모든 지표가 저장되어 있음.
    query = f"SELECT * FROM financial_metrics WHERE ticker = '{ticker}' LIMIT 1"
    
    try:
        data = load_data_from_db(query, db_engine)
        if data.empty:
            logger.warning(f"'{ticker}'에 대한 재무 지표 데이터를 찾을 수 없습니다.")
            return {"error": "No financial metrics data found."}
        
        metrics = data.iloc[0]

        # --- 2. 효율성 지표 계산 및 평가 ---
        # 이 지표들은 산업별 편차가 매우 크므로, 상대 평가가 중요합니다.
        # 여기서는 일반적인 기준으로 평가합니다.
        
        # 총자산회전율 (Asset Turnover)
        asset_turnover = metrics.get('assetTurnover', 0)
        if asset_turnover > 1.0:
            asset_turnover_eval = "우수 (자산을 매우 효율적으로 사용하여 매출 발생)"
        elif asset_turnover > 0.5:
            asset_turnover_eval = "양호"
        else:
            asset_turnover_eval = "저조 (자본 집약적 산업이거나, 자산 활용도 개선 필요)"

        # 재고자산회전율 (Inventory Turnover)
        inventory_turnover = metrics.get('inventoryTurnover', 0)
        if inventory_turnover > 10.0:
            inventory_turnover_eval = "최우수 (재고가 매우 빠르게 판매됨)"
        elif inventory_turnover > 4.0:
            inventory_turnover_eval = "양호 (효율적인 재고 관리)"
        else:
            inventory_turnover_eval = "주의 필요 (재고 과다 또는 판매 부진 가능성)"
            
        # 매출채권회전율 (Receivables Turnover)
        receivables_turnover = metrics.get('receivablesTurnover', 0)
        if receivables_turnover > 12.0:
            receivables_turnover_eval = "최우수 (외상대금 회수 매우 빠름, 현금흐름에 긍정적)"
        elif receivables_turnover > 6.0:
            receivables_turnover_eval = "양호"
        else:
            receivables_turnover_eval = "검토 필요 (대금 회수 기간이 길어 현금흐름에 부담)"

        # --- 3. 결과 종합 ---
        report = {
            "summary": "종합적인 운영 효율성은 양호/우수/저조 합니다.", # 점수화 로직 추가 후 계산
            "asset_turnover": {
                "value": f"{asset_turnover:.2f}회",
                "evaluation": asset_turnover_eval,
                "question": "자산을 얼마나 빨리 굴려서 매출을 발생시키고 있는가?"
            },
            "inventory_turnover": {
                "value": f"{inventory_turnover:.2f}회",
                "evaluation": inventory_turnover_eval,
                "question": "재고가 얼마나 효율적으로 관리되고 빨리 팔려나가는가?"
            },
            "cash_ratio": { # Key in the original document was cash_ratio, but it should be receivables_turnover
                "value": f"{receivables_turnover:.2f}회",
                "evaluation": receivables_turnover_eval,
                "question": "외상으로 판 물건 대금을 얼마나 빨리 회수하고 있는가?"
            }
        }
        
        # 간단한 종합 점수 계산
        eval_scores = {'최우수': 2, '우수': 2, '양호': 1, '검토 필요': 0, '주의 필요': 0, '저조': -1}
        total_score = sum([
            eval_scores.get(report['asset_turnover']['evaluation'], 0),
            eval_scores.get(report['inventory_turnover']['evaluation'], 0),
            eval_scores.get(report['cash_ratio']['evaluation'], 0) # Key in the original document was cash_ratio, but it should be receivables_turnover
        ])
        
        if total_score >= 5:
            report['summary'] = "최우수: 자산, 재고, 매출채권 등 모든 측면에서 매우 효율적인 운영 능력을 보여줍니다."
        elif total_score >= 3:
            report['summary'] = "양호: 전반적으로 안정적인 운영 효율성을 유지하고 있습니다."
        else:
            report['summary'] = "개선 필요: 특정 영역(재고, 매출채권 등)에서 운영 효율성을 개선할 여지가 있습니다."

        return report

    except Exception as e:
        logger.error(f"'{ticker}' 효율성 분석 중 오류 발생: {e}", exc_info=True)
        return {"error": f"An error occurred during efficiency analysis for {ticker}."}

