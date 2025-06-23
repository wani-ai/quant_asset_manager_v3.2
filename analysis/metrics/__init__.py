# /analysis/metrics/__init__.py

"""
The metrics sub-package contains specialized modules for the detailed analysis
of each financial metric category.

Each module provides functions to evaluate a company's performance based on
specific aspects like profitability, stability, liquidity, etc.

This __init__.py file aggregates the main evaluation functions from each module,
providing a single, convenient access point for the rest of the system,
such as the main QuantSystem controller.
"""

# -----------------------------------------------------------------------------
# 각 전문 분석 모듈에서 핵심 평가 함수(function)들을 불러옵니다.
# 이를 통해 외부에서는 'from analysis.metrics import evaluate_profitability'와 같이
# 간결하게 함수를 호출할 수 있습니다.
# (실제 구현 시, 각 함수는 이 __init__.py와 동일한 디렉토리 내에
# profitability_analyzer.py와 같은 별도 파일로 정의되어야 합니다.)
# -----------------------------------------------------------------------------
from .profitability_analyzer import evaluate_profitability
from .stability_analyzer import evaluate_stability
from .liquidity_analyzer import evaluate_liquidity
from .efficiency_analyzer import evaluate_efficiency
from .growth_analyzer import evaluate_growth
from .valuation_analyzer import evaluate_valuation


# -----------------------------------------------------------------------------
# __all__ 변수는 'from analysis.metrics import *' 구문을 사용할 때,
# 어떤 이름들을 외부로 공개할지 명시적으로 정의하는 파이썬의 표준 방식입니다.
# 이를 통해 패키지의 공개 API를 명확하게 관리할 수 있습니다.
# -----------------------------------------------------------------------------
__all__ = [
    'evaluate_profitability',
    'evaluate_stability',
    'evaluate_liquidity',
    'evaluate_efficiency',
    'evaluate_growth',
    'evaluate_valuation',
]
