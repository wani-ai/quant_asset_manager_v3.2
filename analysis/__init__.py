# /analysis/__init__.py

"""
The analysis package serves as the core feature engineering and quantitative
analysis engine of the system.

This package contains modules for:
- Relative valuation against dynamic peer groups (relative_valuator).
- Technical attractiveness scoring (technical_scorer).
- Comprehensive portfolio risk assessment (portfolio_risk).
- A sub-package 'metrics' for detailed analysis of individual financial metric categories.

This __init__.py file makes the main analyzer classes available for convenient
import by higher-level modules, such as the main QuantSystem controller.
"""

# -----------------------------------------------------------------------------
# 각 전문 분석 모듈에서 핵심 클래스(Class)들을 불러옵니다.
# 이를 통해 외부에서는 'from analysis import RelativeValuator'와 같이
# 간결하게 클래스를 호출할 수 있습니다.
# -----------------------------------------------------------------------------
from .relative_valuator import RelativeValuator
from .technical_scorer import TechnicalScorer
from .portfolio_risk import PortfolioRiskAnalyzer

# -----------------------------------------------------------------------------
# __all__ 변수는 'from analysis import *' 구문을 사용할 때,
# 어떤 이름들을 외부로 공개할지 명시적으로 정의하는 파이썬의 표준 방식입니다.
# 이를 통해 패키지의 공개 API를 명확하게 관리할 수 있습니다.
# -----------------------------------------------------------------------------
__all__ = [
    'RelativeValuator',
    'TechnicalScorer',
    'PortfolioRiskAnalyzer',
]
