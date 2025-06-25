# /backtesting/strategies/__init__.py

"""
The strategies sub-package contains various investment strategy classes
that are designed to be run by the backtrader engine.

Each strategy class inherits from backtrader.Strategy and defines the
specific logic for generating buy/sell signals and managing positions.

This __init__.py file exposes the main strategy classes for easy import
by the backtesting engine.
"""

# 머신러닝 랭킹 모델의 예측을 기반으로 하는 투자 전략
from .ml_ranking_strategy import ML_Ranking_Strategy


__all__ = [
    'ML_Ranking_Strategy',
]

