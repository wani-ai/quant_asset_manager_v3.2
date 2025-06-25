# /ml_models/prediction/__init__.py

"""
The prediction sub-package contains pipelines for loading pre-trained models
and using them to make predictions on new, unseen data.

These scripts are the "inference" part of the MLOps cycle, taking in
freshly processed features and outputting actionable predictions like
stock rankings or metric forecasts.

This __init__.py file exposes the main prediction functions for easy use
by the application layer (e.g., the QuantSystem controller).
"""

# 주식 랭킹 모델을 사용하여 예측을 생성하는 함수
from .predict_stock_returns import predict_with_ranking_model

# 시계열 모델을 사용하여 미래 지표를 예측하는 함수
from .predict_timeseries import predict_future_metrics


# __all__ 변수는 패키지의 공개 API를 명시적으로 정의합니다.
__all__ = [
    'predict_with_ranking_model',
    'predict_future_metrics',
]
