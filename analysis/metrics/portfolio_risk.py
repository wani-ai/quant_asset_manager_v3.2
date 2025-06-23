# /analysis/portfolio_risk.py

import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
from sqlalchemy import Engine

# 시스템의 다른 모듈에서 클래스와 함수를 불러옵니다.
from data.connectors import SmartDataManager
import config

class PortfolioRiskAnalyzer:
    """
    주어진 포트폴리오의 종합적인 리스크를 평가하는 분석 모듈.
    하방 리스크, 거시경제 리스크, 시스템적 팩터 리스크를 정량화하고,
    하나의 '종합 리스크 점수'로 집계합니다.
    """
    def __init__(self, db_engine: Engine, data_manager: SmartDataManager):
        """
        PortfolioRiskAnalyzer의 생성자.

        :param db_engine: 데이터베이스 연결을 위한 SQLAlchemy 엔진.
        :param data_manager: 외부 API 데이터를 가져오기 위한 SmartDataManager 인스턴스.
        """
        self.db_engine = db_engine
        self.data_manager = data_manager
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_portfolio_returns(self, tickers: list, period: str = "1y", weights: list = None) -> pd.Series:
        """포트폴리오의 일일 수익률 시계열을 계산합니다."""
        all_prices = []
        for ticker in tickers:
            # SmartDataManager를 통해 개별 종목의 수정 종가 데이터를 가져옵니다.
            # 가정: data_manager에 해당 기능이 구현되어 있음.
            prices = self.data_manager.get_historical_prices(ticker, period=period)
            if prices is not None:
                all_prices.append(prices['adjClose'].rename(ticker))
        
        if not all_prices:
            raise ValueError("포트폴리오를 구성하는 종목의 가격 데이터를 가져올 수 없습니다.")

        prices_df = pd.concat(all_prices, axis=1)
        returns_df = prices_df.pct_change().dropna()

        # 가중치가 주어지지 않으면 동일 가중으로 계산
        if weights is None:
            weights = np.array([1/len(tickers)] * len(tickers))
        
        portfolio_returns = returns_df.dot(weights)
        return portfolio_returns

    def _calculate_cvar(self, returns: pd.Series, alpha: float = 0.99) -> float:
        """역사적 시뮬레이션 기반으로 Conditional Value at Risk (CVaR)를 계산합니다."""
        if returns.empty:
            return 0.0
        
        var_level = returns.quantile(1 - alpha)
        cvar = returns[returns <= var_level].mean()
        return cvar

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """최대 낙폭(Maximum Drawdown)을 계산합니다."""
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """샤프 지수(Sharpe Ratio)를 계산합니다."""
        excess_returns = returns - risk_free_rate / 252 # 일일 무위험 수익률로 변환
        # 연율화 (일일 데이터 기준)
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return sharpe_ratio

    def _calculate_macro_exposure(self, portfolio_returns: pd.Series) -> dict:
        """포트폴리오의 거시경제 리스크(금리, 인플레이션) 노출도를 분석합니다."""
        # 가정: data_manager를 통해 거시경제 데이터를 가져올 수 있음.
        # macro_data = self.data_manager.get_macro_data(['DGS10', 'CPIAUCSL'])
        # 이 예제에서는 가상 데이터로 대체합니다.
        macro_data = pd.DataFrame({
            'interest_rate_change': np.random.randn(len(portfolio_returns)) * 0.001,
            'inflation': np.random.randn(len(portfolio_returns)) * 0.0001 + 0.0002
        }, index=portfolio_returns.index)

        # 금리 리스크 (금리 베타)
        X = sm.add_constant(macro_data['interest_rate_change'])
        y = portfolio_returns
        model = sm.OLS(y, X).fit()
        interest_rate_beta = model.params.get('interest_rate_change', 0)

        # 인플레이션 리스크 (실질 수익률)
        real_returns = portfolio_returns - macro_data['inflation']
        avg_real_return = real_returns.mean() * 252 # 연율화

        return {
            'interest_rate_beta': interest_rate_beta,
            'annualized_real_return': avg_real_return
        }

    def _calculate_fama_french_exposure(self, portfolio_returns: pd.Series) -> dict:
        """Fama-French 3-Factor 모델을 사용하여 시스템적 팩터 리스크를 분석합니다."""
        # 가정: data_manager를 통해 Fama-French 팩터 데이터를 가져올 수 있음.
        # ff_factors = self.data_manager.get_fama_french_factors()
        # 이 예제에서는 가상 데이터로 대체합니다.
        ff_factors = pd.DataFrame({
            'Mkt-RF': np.random.randn(len(portfolio_returns)) * 0.01,
            'SMB': np.random.randn(len(portfolio_returns)) * 0.005,
            'HML': np.random.randn(len(portfolio_returns)) * 0.005,
        }, index=portfolio_returns.index)

        y = portfolio_returns
        X = ff_factors[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        return {
            'alpha': model.params.get('const', 0) * 252, # 연율화
            'beta_mkt': model.params.get('Mkt-RF', 0),
            'beta_smb': model.params.get('SMB', 0),
            'beta_hml': model.params.get('HML', 0),
            'r_squared': model.rsquared
        }

    def _normalize_score(self, value, min_val, max_val, higher_is_better=False):
        """값을 0-100점 척도로 정규화합니다. 높은 점수 = 낮은 리스크"""
        if higher_is_better:
            score = 100 * (value - min_val) / (max_val - min_val)
        else:
            score = 100 * (1 - (value - min_val) / (max_val - min_val))
        return np.clip(score, 0, 100)
    
    def generate_full_report(self, tickers: list, weights: list = None) -> dict:
        """
        주어진 포트폴리오에 대한 종합 리스크 리포트를 생성합니다.

        :param tickers: 포트폴리오를 구성하는 주식 티커 리스트.
        :param weights: 각 주식의 가중치. None일 경우 동일 가중.
        :return: 모든 리스크 지표와 점수를 포함하는 딕셔너리.
        """
        self.logger.info(f"포트폴리오 {tickers}에 대한 종합 리스크 분석을 시작합니다.")
        try:
            portfolio_returns = self._get_portfolio_returns(tickers, weights=weights)

            # 1. 핵심 리스크 지표 계산
            cvar_99 = self._calculate_cvar(portfolio_returns, alpha=0.99)
            max_dd = self._calculate_max_drawdown(portfolio_returns)
            sharpe = self._calculate_sharpe_ratio(portfolio_returns)
            macro_exposure = self._calculate_macro_exposure(portfolio_returns)
            ff_exposure = self._calculate_fama_french_exposure(portfolio_returns)
            
            # 2. 각 지표 점수화
            cvar_score = self._normalize_score(cvar_99, min_val=-0.05, max_val=0) # CVaR -5% = 0점, 0% = 100점
            mdd_score = self._normalize_score(max_dd, min_val=-0.30, max_val=0)  # MDD -30% = 0점, 0% = 100점
            sharpe_score = self._normalize_score(sharpe, min_val=0, max_val=2.0, higher_is_better=True) # 샤프 0 = 0점, 2 = 100점
            # ... 기타 리스크 지표에 대한 점수화 로직 추가 ...

            # 3. 종합 리스크 점수 산출
            # 이 가중치는 config.py에서 관리해야 합니다.
            risk_weights = self.config.RISK_SCORE_WEIGHTS
            composite_risk_score = (
                cvar_score * risk_weights.get('cvar', 0.4) +
                mdd_score * risk_weights.get('mdd', 0.3) +
                sharpe_score * risk_weights.get('sharpe', 0.3)
            )

            report = {
                'composite_risk_score': composite_risk_score,
                'scores': {
                    'cvar_score': cvar_score,
                    'mdd_score': mdd_score,
                    'sharpe_score': sharpe_score
                },
                'raw_metrics': {
                    'cvar_99': cvar_99,
                    'max_drawdown': max_dd,
                    'sharpe_ratio': sharpe,
                    'macro_exposure': macro_exposure,
                    'fama_french_exposure': ff_exposure
                }
            }
            return report

        except Exception as e:
            self.logger.error(f"포트폴리오 리스크 분석 중 오류 발생: {e}")
            return {'error': str(e)}

