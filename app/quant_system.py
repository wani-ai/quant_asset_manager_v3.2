# /app/quant_system.py

import logging
import joblib
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional

# -----------------------------------------------------------------------------
# 시스템의 모든 핵심 모듈을 불러옵니다.
# -----------------------------------------------------------------------------
import config

# 데이터 계층
from data.connectors import SmartDataManager
from data.database import get_database_engine, load_data_from_db

# 분석 및 특징 공학 계층
from analysis.relative_valuator import RelativeValuator
from analysis.technical_scorer import TechnicalScorer
from analysis.portfolio_risk import PortfolioRiskAnalyzer
# 상세 지표 분석을 위한 개별 모듈 임포트
from analysis.metrics.profitability_analyzer import evaluate_profitability
from analysis.metrics.stability_analyzer import evaluate_stability
from analysis.metrics.liquidity_analyzer import evaluate_liquidity
from analysis.metrics.efficiency_analyzer import evaluate_efficiency
from analysis.metrics.growth_analyzer import evaluate_growth
from analysis.metrics.valuation_analyzer import evaluate_valuation

# 머신러닝 모델 계층
from ml_models.prediction.predict_stock_returns import predict_with_ranking_model

# -----------------------------------------------------------------------------
# 시스템의 모든 기능을 관장하는 메인 클래스
# -----------------------------------------------------------------------------
class QuantSystem:
    """
    퀀트 자산 관리 시스템의 메인 컨트롤러 클래스.
    데이터 수집, 분석, 모델 예측, 리포팅 등 모든 워크플로우를 관장합니다.
    """
    def __init__(self):
        """
        시스템 생성자: 모든 필수 구성 요소를 초기화하고 로드합니다.
        """
        self._setup_logging()
        self.logger.info("퀀트 자산 관리 시스템(QuantSystem) 초기화를 시작합니다.")

        # 설정 및 데이터베이스 엔진 로드
        self.config = config
        self.db_engine = get_database_engine()

        # 핵심 모듈 인스턴스화 (의존성 주입)
        self.data_manager = SmartDataManager()
        self.relative_valuator = RelativeValuator(db_engine=self.db_engine)
        self.technical_scorer = TechnicalScorer(data_manager=self.data_manager)
        self.portfolio_risk_analyzer = PortfolioRiskAnalyzer(db_engine=self.db_engine, data_manager=self.data_manager)
        
        # 훈련된 머신러닝 모델 로드
        self.models = self._load_models()
        
        self.logger.info("QuantSystem 초기화 완료. 시스템이 준비되었습니다.")

    def _setup_logging(self):
        """시스템 전반에 사용될 로거를 설정하는 내부 헬퍼 함수."""
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _load_models(self) -> dict:
        """'saved_models' 디렉토리에서 훈련된 모델 파일들을 로드합니다."""
        models = {}
        model_path = Path(getattr(self.config, 'ML_MODELS_DIR', 'ml_models')) / 'saved_models'
        if not model_path.exists():
            self.logger.error(f"모델 디렉토리가 존재하지 않습니다: {model_path}")
            return models

        ranking_model_file = model_path / 'stock_ranking_model.pkl'
        if ranking_model_file.exists():
            try:
                models['ranking_model'] = joblib.load(ranking_model_file)
                self.logger.info(f"'{ranking_model_file.name}' 모델 로드 성공.")
            except Exception as e:
                self.logger.error(f"'{ranking_model_file.name}' 모델 로드 실패: {e}")
        return models

    def get_top_ranked_stocks(self, top_n: int = 100, strategy: str = 'blend') -> Optional[pd.DataFrame]:
        """
        전체 투자 유니버스를 대상으로 종합 점수를 산출하고,
        상위 N개 주식의 랭킹을 반환합니다. 이 메소드는 시스템의 모든 분석 역량을 종합합니다.
        """
        self.logger.info(f"'{strategy}' 전략 기반 Top {top_n} 랭킹 분석을 시작합니다.")
        
        try:
            # 1. 펀더멘털 기반 듀얼 평가 실행
            fundamental_df = self.relative_valuator.run_full_valuation(strategy=strategy)
            if fundamental_df.empty:
                self.logger.error("펀더멘털 점수 계산에 실패했습니다.")
                return None

            # 2. 기술적 매력도 점수 추가
            tickers_to_score = fundamental_df.index.tolist()
            technical_df = self.technical_scorer.get_scores_for_tickers(tickers_to_score)
            
            # 두 데이터프레임 병합
            combined_df = fundamental_df.join(technical_df, how='left').fillna(50) # 점수 없으면 중간값(50) 부여

            # 3. ML 모델 예측 점수 추가
            if 'ranking_model' in self.models:
                # config 파일에 모델이 학습한 특징(feature) 목록이 정의되어 있어야 함
                model_features = getattr(self.config, 'MODEL_FEATURES', [])
                # 점수 계산에 필요한 특징이 데이터프레임에 모두 있는지 확인
                feature_scores = [f for f in model_features if f in combined_df.columns]
                
                if feature_scores:
                    ml_predictions = predict_with_ranking_model(
                        model=self.models['ranking_model'],
                        features_df=combined_df[feature_scores]
                    )
                    if ml_predictions is not None:
                        combined_df = combined_df.join(ml_predictions, how='left')

            # 4. 최종 종합 점수 계산 및 랭킹
            final_weights = getattr(self.config, 'FINAL_SCORE_WEIGHTS', {})
            combined_df['final_score'] = (
                combined_df.get('fundamental_score', 0) * final_weights.get('fundamental', 0.5) +
                combined_df.get('technical_score', 0) * final_weights.get('technical', 0.3) +
                combined_df.get('prediction_score', 0) * final_weights.get('ml', 0.2)
            )
            
            combined_df['final_rank'] = combined_df['final_score'].rank(ascending=False, method='min')
            final_df = combined_df.sort_values('final_rank')
            
            return final_df.head(top_n)

        except Exception as e:
            self.logger.error(f"Top 랭킹 분석 중 오류 발생: {e}", exc_info=True)
            return pd.DataFrame()


    def get_stock_deep_dive_report(self, ticker: str) -> Dict[str, Any]:
        """
        단일 종목에 대한 모든 평가 항목을 포함한 심층 분석 보고서를 생성합니다.
        """
        self.logger.info(f"'{ticker}'에 대한 심층 분석을 시작합니다.")
        
        report = {
            'ticker': ticker,
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # 1. 재무 분석 (6개 카테고리)
            'profitability': evaluate_profitability(ticker, self.db_engine),
            'stability': evaluate_stability(ticker, self.db_engine),
            'liquidity': evaluate_liquidity(ticker, self.db_engine),
            'efficiency': evaluate_efficiency(ticker, self.db_engine),
            'growth': evaluate_growth(ticker, self.db_engine),
            'valuation': evaluate_valuation(ticker, self.db_engine),

            # 2. 기술적 분석
            'technical_analysis': self.technical_scorer.get_summary_for_ticker(ticker),
            
            # 3. 리스크 평가 (개별 종목 관점)
            'volatility_analysis': self.portfolio_risk_analyzer.get_single_stock_volatility(ticker),
        }
        return report

    def get_portfolio_risk_dashboard(self, portfolio_tickers: List[str]) -> Optional[Dict[str, Any]]:
        """
        주어진 포트폴리오에 대한 종합 리스크 대시보드 데이터를 생성합니다.
        """
        self.logger.info("포트폴리오 리스크 분석을 시작합니다.")
        risk_data = self.portfolio_risk_analyzer.generate_full_report(portfolio_tickers)
        return risk_data
