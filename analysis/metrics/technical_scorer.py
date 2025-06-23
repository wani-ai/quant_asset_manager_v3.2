# /analysis/technical_scorer.py

import pandas as pd
import numpy as np
import logging
import pandas_ta as ta
from typing import List, Dict, Any

from data.connectors import SmartDataManager
import config

class TechnicalScorer:
    """
    다양한 기술적 지표를 계산하고, 이를 표준화 및 가중 평균하여
    개별 종목의 '기술적 매력도 점수'를 산출하는 모듈.
    """
    def __init__(self, data_manager: SmartDataManager):
        """
        TechnicalScorer의 생성자.

        :param data_manager: 외부 API 데이터를 가져오기 위한 SmartDataManager 인스턴스.
        """
        self.data_manager = data_manager
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        # 기술적 분석에 필요한 파라미터를 config 파일에서 로드
        self.params = self.config.TECHNICAL_ANALYSIS_PARAMS

    def _get_historical_data(self, ticker: str) -> pd.DataFrame:
        """지정된 티커의 시계열 OHLCV 데이터를 가져옵니다."""
        # 가정: data_manager에 해당 기능이 구현되어 있음.
        # 데이터는 최소 200일 이상을 가져와야 장기 이평선 계산이 가능합니다.
        prices_df = self.data_manager.get_historical_prices(ticker, period="1y")
        if prices_df is None or prices_df.empty:
            self.logger.warning(f"'{ticker}'의 시계열 데이터를 가져올 수 없습니다.")
            return pd.DataFrame()
        return prices_df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        pandas-ta 라이브러리를 사용하여 모든 기술적 지표를 계산합니다.
        """
        if df.empty:
            return df

        # Strategy 객체를 사용하여 여러 지표를 한 번에 계산
        strategy = ta.Strategy(
            name="QuantSystem Standard",
            description="Trend, Momentum, Volume, and Volatility Indicators",
            ta=[
                {"kind": "sma", "length": self.params['sma_short']},
                {"kind": "sma", "length": self.params['sma_long']},
                {"kind": "ema", "length": self.params['ema_short']},
                {"kind": "macd", "fast": self.params['macd_fast'], "slow": self.params['macd_slow'], "signal": self.params['macd_signal']},
                {"kind": "rsi", "length": self.params['rsi_period']},
                {"kind": "bbands", "length": self.params['bbands_period']},
                {"kind": "obv"},
                {"kind": "ad"},
            ]
        )
        df.ta.strategy(strategy)
        return df

    def _calculate_z_scores(self, df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        각 지표의 Z-점수를 롤링 윈도우 기반으로 계산하여 표준화합니다.
        """
        # Z-점수 계산 대상 지표 목록
        z_score_cols = self.params['z_score_columns']
        
        for col in z_score_cols:
            if col in df.columns:
                rolling_mean = df[col].rolling(window=window, min_periods=window//2).mean()
                rolling_std = df[col].rolling(window=window, min_periods=window//2).std()
                
                # std가 0일 경우 Z-점수는 0
                df[f'{col}_zscore'] = (df[col] - rolling_mean) / rolling_std
                df[f'{col}_zscore'].fillna(0, inplace=True)
        return df

    def get_summary_for_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        단일 종목에 대한 최신 기술적 지표 및 분석 요약을 반환합니다.
        'deepdive' 기능에서 사용됩니다.
        """
        self.logger.info(f"'{ticker}'에 대한 기술적 분석 요약을 생성합니다.")
        df = self._get_historical_data(ticker)
        if df.empty:
            return {"error": "데이터를 가져올 수 없습니다."}

        df_with_indicators = self._calculate_indicators(df)
        latest_data = df_with_indicators.iloc[-1]

        # 주요 지표를 딕셔너리로 정리
        summary = {
            "price": latest_data.get('Close'),
            "trend": {
                f"sma_{self.params['sma_short']}": latest_data.get(f"SMA_{self.params['sma_short']}"),
                f"sma_{self.params['sma_long']}": latest_data.get(f"SMA_{self.params['sma_long']}"),
                "status": "상승 추세" if latest_data.get(f"SMA_{self.params['sma_short']}") > latest_data.get(f"SMA_{self.params['sma_long']}") else "하락 추세"
            },
            "momentum": {
                "rsi": latest_data.get(f"RSI_{self.params['rsi_period']}"),
                "macd_line": latest_data.get(f"MACD_{self.params['macd_fast']}_{self.params['macd_slow']}_{self.params['macd_signal']}"),
                "macd_signal": latest_data.get(f"MACDs_{self.params['macd_fast']}_{self.params['macd_slow']}_{self.params['macd_signal']}")
            },
            "volume": {
                "obv": latest_data.get("OBV"),
                "ad_line": latest_data.get("AD")
            },
            "volatility": {
                "bollinger_upper": latest_data.get(f"BBU_{self.params['bbands_period']}_2.0"),
                "bollinger_lower": latest_data.get(f"BBL_{self.params['bbands_period']}_2.0")
            }
        }
        return summary

    def get_scores_for_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """
        여러 종목에 대해 '기술적 매력도 점수'를 계산하여 반환합니다.
        """
        self.logger.info(f"{len(tickers)}개 종목의 기술적 매력도 점수를 계산합니다.")
        all_scores = []
        
        for ticker in tickers:
            df = self._get_historical_data(ticker)
            if df.empty:
                continue

            df = self._calculate_indicators(df)
            df = self._calculate_z_scores(df)

            # 최신 Z-점수 가져오기
            latest_z_scores = df.iloc[-1]
            
            # --- 종합 점수 산출 로직 ---
            weights = self.config.TECHNICAL_SCORE_WEIGHTS # config 파일에서 가중치 로드
            categories = self.config.INDICATOR_CATEGORIES
            
            composite_score = 0
            for category, details in categories.items():
                category_score = 0
                total_weight_in_cat = 0
                for indicator_z, weight, higher_is_better in details:
                    if indicator_z in latest_z_scores:
                        z_score = latest_z_scores[indicator_z]
                        # 방향성 통일 (높을수록 좋게)
                        score_direction = 1 if higher_is_better else -1
                        category_score += z_score * weight * score_direction
                        total_weight_in_cat += weight

                if total_weight_in_cat > 0:
                    # 카테고리 내에서 정규화 후, 카테고리 가중치 적용
                    composite_score += (category_score / total_weight_in_cat) * weights.get(category, 0)

            all_scores.append({'ticker': ticker, 'technical_score': composite_score})

        if not all_scores:
            return pd.DataFrame(columns=['ticker', 'technical_score']).set_index('ticker')
            
        score_df = pd.DataFrame(all_scores).set_index('ticker')
        
        # 최종 점수를 0-100 척도로 변환 (Min-Max Scaling)
        min_score = score_df['technical_score'].min()
        max_score = score_df['technical_score'].max()
        if max_score > min_score:
            score_df['technical_score'] = 100 * (score_df['technical_score'] - min_score) / (max_score - min_score)
        else:
            score_df['technical_score'] = 50 # 모든 점수가 동일할 경우
            
        return score_df
