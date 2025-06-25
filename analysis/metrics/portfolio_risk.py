# /analysis/portfolio_risk.py

import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
from sqlalchemy import Engine
from typing import List, Dict, Any, Optional
import pandas_ta as ta  # ✨✨✨ ATR 계산을 위해 추가합니다. ✨✨✨

# 시스템의 다른 모듈에서 클래스와 함수를 불러옵니다.
from data.connectors import SmartDataManager
from data.database import load_data_from_db
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

    def _get_portfolio_returns(self, tickers: List[str], period: str = "1y", weights: Optional[List[float]] = None) -> Optional[pd.Series]:
        """포트폴리오의 일일 수익률 시계열을 계산합니다."""
        all_prices = []
        for ticker in tickers:
            # ✨✨✨ 핵심 수정: 변수명을 prices에서 prices_df로 변경하여 오류를 해결합니다. ✨✨✨
            prices_df = self.data_manager.get_historical_prices(ticker, period=period)
            if prices_df is not None and not prices_df.empty:
                all_prices.append(prices_df['close'].rename(ticker)) # 수정종가(adjClose)가 없다면 종가(close) 사용
        
        if not all_prices:
            self.logger.error("포트폴리오를 구성하는 종목의 가격 데이터를 가져올 수 없습니다.")
            return None

        prices_df = pd.concat(all_prices, axis=1).fillna(method='ffill')
        returns_df = prices_df.pct_change().dropna()

        if weights is None:
            weights = np.array([1/len(tickers)] * len(tickers))
        
        portfolio_returns = returns_df.dot(weights)
        return portfolio_returns

    def _calculate_cvar(self, returns: pd.Series, alpha: float = 0.99) -> float:
        """역사적 시뮬레이션 기반으로 Conditional Value at Risk (CVaR)를 계산합니다."""
        if returns is None or returns.empty: return 0.0
        var_level = returns.quantile(1 - alpha)
        cvar = returns[returns <= var_level].mean()
        return cvar if pd.notna(cvar) else 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """최대 낙폭(Maximum Drawdown)을 계산합니다."""
        if returns is None or returns.empty: return 0.0
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min() if pd.notna(drawdown.min()) else 0.0

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """연율화된 샤프 지수(Sharpe Ratio)를 계산합니다."""
        if returns is None or returns.empty or returns.std() == 0: return 0.0
        # 가정: 무위험 수익률은 0으로 간주 (또는 FRED에서 가져와야 함)
        risk_free_rate = 0.0
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return sharpe_ratio if pd.notna(sharpe_ratio) else 0.0
        
    def get_single_stock_volatility(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """단일 종목의 변동성 관련 지표를 계산합니다."""
        self.logger.info(f"'{ticker}' 종목의 변동성 분석을 시작합니다.")
        prices_df = self.data_manager.get_historical_prices(ticker, period=period)
        if prices_df is None or prices_df.empty:
            return {'error': '가격 데이터를 가져올 수 없습니다.'}
        
        returns = prices_df['close'].pct_change()
        # Historical Volatility (연율화)
        historical_volatility = returns.std() * np.sqrt(252)
        
        # ATR (Average True Range) 계산
        if hasattr(prices_df, 'ta'):
             prices_df.ta.atr(append=True)
             atr = prices_df.iloc[-1].get(f"ATRr_14", 0) # 기본값 14
        else:
            atr = 0

        return {
            'historical_volatility_annualized': historical_volatility,
            'atr_14': atr
            # Implied Volatility(IV)는 옵션 데이터가 필요하므로 여기서는 생략
        }

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
        if max_val == min_val: return 50.0 # 분모가 0이 되는 경우 방지
        
        # 값의 범위를 [0, 1]로 클리핑
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_value = np.clip(scaled_value, 0, 1)

        if higher_is_better:
            return scaled_value * 100
        else:
            return (1 - scaled_value) * 100
    
    def generate_full_report(self, tickers: List[str], weights: Optional[List[float]] = None) -> Optional[Dict[str, Any]]:
        """주어진 포트폴리오에 대한 종합 리스크 리포트를 생성합니다."""
        self.logger.info(f"포트폴리오 {tickers}에 대한 종합 리스크 분석을 시작합니다.")
        
        try:
            portfolio_returns = self._get_portfolio_returns(tickers, weights=weights)
            if portfolio_returns is None:
                return None

            # 1. 핵심 리스크 지표 계산
            cvar_99 = self._calculate_cvar(portfolio_returns, alpha=0.99)
            max_dd = self._calculate_max_drawdown(portfolio_returns)
            sharpe = self._calculate_sharpe_ratio(portfolio_returns)
            
            # (향후 확장) 거시경제 및 파마-프렌치 분석 로직 호출
            macro_exposure = self._calculate_macro_exposure(portfolio_returns)
            ff_exposure = self._calculate_fama_french_exposure(portfolio_returns)

            # 2. 각 지표 점수화
            # 점수화 범위와 기준은 config.py에서 관리
            cvar_score = self._normalize_score(cvar_99, -0.05, 0) # CVaR -5% = 0점, 0% = 100점
            mdd_score = self._normalize_score(max_dd, -0.30, 0)  # MDD -30% = 0점, 0% = 100점
            sharpe_score = self._normalize_score(sharpe, 0, 2.0, higher_is_better=True) # 샤프 0 = 0점, 2 = 100점

            # 3. 종합 리스크 점수 산출
            risk_weights = getattr(self.config, 'RISK_SCORE_WEIGHTS', {})
            composite_risk_score = (
                cvar_score * risk_weights.get('cvar', 0.4) +
                mdd_score * risk_weights.get('mdd', 0.3) +
                sharpe_score * risk_weights.get('sharpe', 0.3)
            )

            report = {
                'composite_risk_score': composite_risk_score,
                'scores': {'cvar_score': cvar_score, 'mdd_score': mdd_score, 'sharpe_score': sharpe_score},
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
            self.logger.error(f"포트폴리오 리스크 분석 중 오류 발생: {e}", exc_info=True)
            return {'error': str(e)}

